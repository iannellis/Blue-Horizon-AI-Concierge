"""Long-lived resources for the booking agent.

Owns two async connection pools -- a read-only pool used exclusively by the
model-facing `search_rooms` tool, and a read-write pool used by write_ops,
list_bookings, and the customers/bookings API endpoints -- plus the rendered
system prompt, the search argument model, and the in-process proposal store.

The read-only pool authenticates as the `bh_agent_ro` Postgres role, which can
read only `rooms` and `room_availability` (see
`blue_horizon/load_data/regrant_booking_agent_role.sql`). The model never
supplies SQL: `search_rooms` runs one of the fixed queries in `search.py`
with arguments validated against the search argument model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from psycopg_pool import AsyncConnectionPool

from blue_horizon.agents._lifecycle import require
from blue_horizon.agents.booking import write_ops
from blue_horizon.agents.booking.config import render_system_prompt
from blue_horizon.agents.booking.db_utils import (
    _tool_error_message_for_model,
    assert_pool_is_read_only,
    check_credentials,
    fetch_rooms_metadata,
)
from blue_horizon.agents.booking.proposals import ProposalStore
from blue_horizon.agents.booking.search import (
    RoomsMetadata,
    build_search_args_model,
)
from blue_horizon.agents.booking.search_runner import search_rooms_with_retry
from blue_horizon.agents.exceptions import ConfigurationError, OperationalError
from blue_horizon.agents.prompt_utils import load_prompt_template, prompt_resource_path

if TYPE_CHECKING:
    import psycopg

    from blue_horizon.agents.booking.search import SearchArgsBase
    from blue_horizon.config import BookingSqlConfig

# How each URL is named in startup error messages. By role, not environment
# variable: the URLs may come from the application's variables or from an
# eval override, and this module is not told which.
_RO_LABEL = "read-only (bh_agent_ro)"
_RW_LABEL = "read-write (bh_agent_rw)"


class BookingSqlResources:
    """Own long-lived resources for the booking agent.

    Both pools, the rendered system prompt, the search argument model, and the
    proposal store live here.

    Attributes:
        config: Parsed configuration.
        pgsql_ro_db_url: Read-only database URL (`bh_agent_ro`), used only by
            `search_rooms`.
        pgsql_rw_db_url: Read-write database URL (`bh_agent_rw`), used by
            write_ops, list_bookings, and the propose_* tools.
        pool: Async connection pool for `pgsql_ro_db_url`.
        write_pool: Async connection pool for `pgsql_rw_db_url`.
        proposals: In-process store of pending booking proposals.
        system_prompt: Rendered system prompt used to build the agent.
        rooms_metadata: Search vocabulary and bounds read at startup.

    """

    config: BookingSqlConfig
    pgsql_ro_db_url: str
    pgsql_rw_db_url: str
    pool: AsyncConnectionPool[Any] | None
    write_pool: AsyncConnectionPool[Any] | None
    proposals: ProposalStore
    system_prompt: str | None
    rooms_metadata: RoomsMetadata | None
    _search_args_model: type[SearchArgsBase] | None
    _system_prompt_resource: str

    def __init__(
        self,
        *,
        config: BookingSqlConfig,
        pgsql_ro_db_url: str,
        pgsql_rw_db_url: str,
    ) -> None:
        """Construct booking SQL resources.

        This initializer performs only lightweight setup and validation:
        - Store configuration and both database URLs.
        - Resolve the prompts directory and system prompt template path.
        - Construct the (empty) proposal store.

        Call ``await startup_check()`` to open both pools and render the
        prompt.

        Args:
            config: Parsed configuration loaded from TOML.
            pgsql_ro_db_url: Read-only database URL (`bh_agent_ro`).
            pgsql_rw_db_url: Read-write database URL (`bh_agent_rw`).

        Raises:
            RuntimeError: If the prompt template folder or template file is missing.

        """
        self.config = config
        self.pgsql_ro_db_url = pgsql_ro_db_url
        self.pgsql_rw_db_url = pgsql_rw_db_url
        self.pool: AsyncConnectionPool[Any] | None = None
        self.write_pool: AsyncConnectionPool[Any] | None = None
        self.room_night_limit = write_ops.RoomNightLimit(
            max_room_nights=config.limits.max_room_nights_per_guest,
            window_hours=config.limits.room_night_window_hours,
        )
        self.proposals = ProposalStore(
            ttl_s=config.proposals.ttl_s, room_night_limit=self.room_night_limit,
        )
        self.system_prompt = None
        self.rooms_metadata = None
        self._search_args_model = None

        self._system_prompt_resource = prompt_resource_path(
            self.config.prompts.folder, self.config.prompts.system_prompt_filename,
        )

    async def startup_check(self) -> None:
        """Initialize resources and validate readiness.

        Opens both pools, proves the read-only pool's role really cannot
        write, fetches the rooms metadata, and from it builds the search
        argument model and renders the final system prompt.

        A blank URL never reaches here: `AppConfig` and the eval overrides
        reject one when configuration loads.

        Raises:
            ConfigurationError: If either database URL has its password
                rejected, or if the read-only pool's role is not actually
                read-only -- treated as a fatal misconfiguration rather than
                a warning, since a silently-writable "read-only" pool is
                exactly the guarantee this design depends on. Neither
                resolves by retrying, so both are kept out of
                `OperationalError`.
            OperationalError: If resources cannot be initialized for a
                transient reason (e.g. the database is unreachable).

        """
        try:
            timeout_s = self.config.db.pool.timeout_s
            await check_credentials(
                self.pgsql_ro_db_url, _RO_LABEL, timeout_s=timeout_s,
            )
            await check_credentials(
                self.pgsql_rw_db_url, _RW_LABEL, timeout_s=timeout_s,
            )
            await self._open_pools()
            await self._assert_read_pool_is_read_only()
            await self._render_system_prompt()
        except ConfigurationError:
            raise
        except OperationalError:
            raise
        except Exception as exc:
            msg = "Booking SQL resources failed during startup"
            raise OperationalError(msg) from exc

    def get_system_prompt(self) -> str:
        """Get the rendered system prompt.

        Returns:
            The rendered system prompt.

        Raises:
            RuntimeError: If the system prompt is not available because
                ``startup_check()`` has not been called or failed.

        """
        return require(self.system_prompt, "BookingSqlResources")

    def get_search_args_model(self) -> type[SearchArgsBase]:
        """Get the `search_rooms` argument model built at startup.

        Returns:
            The Pydantic model whose instances `search_rooms` accepts.

        Raises:
            RuntimeError: If ``startup_check()`` has not been called or failed.

        """
        return require(self._search_args_model, "BookingSqlResources")

    def get_read_pool(self) -> AsyncConnectionPool[Any]:
        """Get the read-only connection pool, narrowed to non-optional.

        Callers outside this module (the API, the LangGraph tool factory, the
        eval harness, notebooks) need a plain `AsyncConnectionPool`, not the
        `AsyncConnectionPool | None` this attribute is typed as before
        `startup_check()` has run. Routing through this method gives them
        that without each call site repeating its own None-check or `# type:
        ignore`.

        Returns:
            AsyncConnectionPool[Any]: The open read-only pool.

        Raises:
            RuntimeError: If `startup_check()` has not been called or failed.

        """
        return require(self.pool, "BookingSqlResources")

    def get_write_pool(self) -> AsyncConnectionPool[Any]:
        """Get the read-write connection pool, narrowed to non-optional.

        Callers outside this module (the API, the LangGraph tool factory, the
        eval harness, notebooks) need a plain `AsyncConnectionPool`, not the
        `AsyncConnectionPool | None` this attribute is typed as before
        `startup_check()` has run. Routing through this method gives them
        that without each call site repeating its own None-check or `# type:
        ignore`.

        Returns:
            AsyncConnectionPool[Any]: The open read-write pool.

        Raises:
            RuntimeError: If `startup_check()` has not been called or failed.

        """
        return require(self.write_pool, "BookingSqlResources")

    async def aclose(self) -> None:
        """Close resources owned by this instance.

        This method is idempotent.

        """
        if self.pool is not None:
            await self.pool.close()
        if self.write_pool is not None:
            await self.write_pool.close()
        self.pool = None
        self.write_pool = None
        self.system_prompt = None

    async def search_rooms(self, args: SearchArgsBase) -> dict[str, Any]:
        """Run a room search and return a bounded result.

        Delegates to `search_runner.search_rooms_with_retry` with this
        instance's read-only pool and configuration, and the
        `DATABASE_UNAVAILABLE` failure text the booking prompt keys off.

        Args:
            args: An instance of `get_search_args_model()`.

        Returns:
            On success, `run_room_search`'s result: ``status="ok"``,
            ``matching_count``, ``rooms``, and ``limit_note`` when the
            requested ``limit`` was clamped. On failure,
            ``status="error"`` with ``error`` and ``error_kind``.

        Raises:
            RuntimeError: If resources were not initialized.

        """
        return await search_rooms_with_retry(
            self.get_read_pool(),
            args,
            retry=self.config.db.retry,
            checkout_timeout_s=self.config.db.pool.timeout_s,
            failure_message=_tool_error_message_for_model(),
        )

    async def _open_pools(self) -> None:
        """Open the read-only and read-write async connection pools.

        ``statement_timeout`` is set at the database role level
        (``ALTER ROLE … SET …``) so it applies consistently under PgBouncer
        transaction pooling, where a per-connection ``SET`` would not
        reliably survive. ``regrant_booking_agent_role.sql`` sets it on
        Parent, and a branch reset carries it to child branches. It takes
        effect only on newly established connections, so an already open
        pool keeps the old value until its connections are recycled.
        ``search_path`` is left at the PostgreSQL default of
        ``"$user", public``, which resolves to ``public``; the paths that need
        certainty set it explicitly on their own connections.

        Each pool is built with ``check=AsyncConnectionPool.check_connection``,
        so psycopg_pool runs a health-check (an empty-query round trip) each
        time a connection is checked out, and discards a connection that fails
        it before it reaches a caller. Combined with ``max_idle``, this ensures
        neither pool ever hands out a connection that Neon has already
        dropped due to compute suspension.

        Raises:
            OperationalError: If either pool cannot be opened.

        """
        if self.pool is not None and self.write_pool is not None:
            return

        async def configure_connection(conn: psycopg.AsyncConnection[Any]) -> None:
            """Enable autocommit on each new connection.

            Args:
                conn: Newly created async psycopg connection.

            """
            await conn.set_autocommit(True)

        pool_cfg = self.config.db.pool
        try:
            self.pool = AsyncConnectionPool(
                conninfo=self.pgsql_ro_db_url,
                min_size=pool_cfg.min_size,
                max_size=pool_cfg.max_size,
                timeout=pool_cfg.timeout_s,
                max_idle=pool_cfg.max_idle_s,
                reconnect_timeout=pool_cfg.reconnect_timeout_s,
                configure=configure_connection,
                check=AsyncConnectionPool.check_connection,
                open=False,
            )
            await self.pool.open()

            self.write_pool = AsyncConnectionPool(
                conninfo=self.pgsql_rw_db_url,
                min_size=pool_cfg.min_size,
                max_size=pool_cfg.max_size,
                timeout=pool_cfg.timeout_s,
                max_idle=pool_cfg.max_idle_s,
                reconnect_timeout=pool_cfg.reconnect_timeout_s,
                configure=configure_connection,
                check=AsyncConnectionPool.check_connection,
                open=False,
            )
            await self.write_pool.open()
        except Exception as exc:
            msg = "Failed to open booking DB pools"
            raise OperationalError(msg) from exc

    async def _assert_read_pool_is_read_only(self) -> None:
        """Prove the read-only pool's role cannot write.

        Raises:
            OperationalError: If this is called before `_open_pools()`, or if
                the probe itself cannot be run (e.g. the database is
                unreachable); `startup_check()` wraps the latter.
            ConfigurationError: If the probe write is not refused; see
                `db_utils.assert_pool_is_read_only`.

        """
        if self.pool is None:
            msg = "_assert_read_pool_is_read_only() called before _open_pools()"
            raise OperationalError(msg)
        await assert_pool_is_read_only(self.pool)

    async def _render_system_prompt(self) -> None:
        """Load rooms metadata, then build the search model and system prompt.

        Both come from the same metadata, so the values the prompt describes
        and the values the tool accepts cannot drift apart.

        Raises:
            ConfigurationError: If the packaged prompt template is missing or
                unreadable -- propagated from `load_prompt_template()`
                unchanged, since wrapping it as `OperationalError` would
                make a missing file look retryable when it is not.
            OperationalError: If prompt rendering fails for any other
                reason (e.g. the database is unreachable for metadata).

        """
        try:
            meta = await fetch_rooms_metadata(self.pgsql_ro_db_url)
            agent_cfg = self.config.agent
            self._search_args_model = build_search_args_model(
                meta,
                max_room_numbers=agent_cfg.max_search_room_numbers,
                default_results=agent_cfg.default_search_results,
                max_results=agent_cfg.max_search_results,
            )
            self.rooms_metadata = meta

            template = load_prompt_template(self._system_prompt_resource)
            self.system_prompt = render_system_prompt(template=template, meta=meta)

        except ConfigurationError:
            raise
        except Exception as exc:
            msg = "Failed to render booking system prompt"
            raise OperationalError(msg) from exc
