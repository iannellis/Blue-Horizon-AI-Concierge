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

import logging
import time
from typing import TYPE_CHECKING, Any, Literal

import psycopg
from psycopg_pool import AsyncConnectionPool, PoolTimeout
from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from blue_horizon.agents._lifecycle import require
from blue_horizon.agents.booking.config import render_system_prompt
from blue_horizon.agents.booking.db_utils import (
    _tool_error_message_for_model,
    fetch_rooms_metadata,
    is_transient_conn_error,
)
from blue_horizon.agents.booking.proposals import ProposalStore
from blue_horizon.agents.booking.search import (
    RoomsMetadata,
    build_search_args_model,
    run_room_search,
)
from blue_horizon.agents.exceptions import ConfigurationError, OperationalError
from blue_horizon.agents.prompt_utils import load_prompt_template, prompt_resource_path

if TYPE_CHECKING:
    from pydantic import BaseModel

    from blue_horizon.config import BookingSqlConfig

logger = logging.getLogger(__name__)

# A no-op UPDATE that matches no row: used only to prove the read-only pool's
# role is actually refused write privileges at startup. Never mutates data
# even if the assertion this guards against has already failed.
_READ_ONLY_PROBE_SQL = "UPDATE room_availability SET status = status WHERE id = -1"

# Coarse classification of a search_rooms failure, independent of message
# text, so evaluators (see eval/stress/workload.py) can assert on structure
# rather than matching prose that guest-facing or model-facing copy changes
# could move. "invalid_arguments" is produced by the tool layer, which rejects
# arguments before this module sees them.
SqlErrorKind = Literal["unavailable", "invalid_arguments", "unexpected"]

# libpq reports a rejected password with no SQLSTATE at connection time, so
# this message fragment is the only signal available.
_AUTH_REJECTED_FRAGMENT = "password authentication failed"


def _require_url(url: str, env_var: str) -> None:
    """Raise if a required database URL is missing or blank.

    An empty string passes ordinary Pydantic `str` validation, so this
    catches the case that gets through config loading but can never
    establish a connection. Deliberately raised as `ConfigurationError`,
    not `OperationalError`: this process's environment will not change
    without an operator restarting it, so retrying is pointless.

    Args:
        url: The URL value to check.
        env_var: Name of the environment variable it came from, for the
            error message.

    Raises:
        ConfigurationError: If `url` is empty or all whitespace.

    """
    if not url or not url.strip():
        msg = f"{env_var} is missing or blank."
        raise ConfigurationError(msg)


async def _check_credentials(url: str, env_var: str, *, timeout_s: float) -> None:
    """Raise `ConfigurationError` if the database rejects a URL's password.

    Opens one direct connection instead of going through the pool, because
    the pool retries a rejected password internally and a checkout only ever
    reports `PoolTimeout`, which is indistinguishable from an outage. Any
    other failure propagates unchanged, for `startup_check` to treat as
    transient.

    Args:
        url: The database URL to authenticate with.
        env_var: Name of the environment variable it came from, for the
            error message.
        timeout_s: Connection timeout in seconds.

    Raises:
        ConfigurationError: If the server rejects the password.
        psycopg.OperationalError: If the connection fails for any other
            reason, such as the database being unreachable.

    """
    try:
        conn = await psycopg.AsyncConnection.connect(
            url, connect_timeout=max(1, int(timeout_s)),
        )
    except psycopg.OperationalError as exc:
        if _AUTH_REJECTED_FRAGMENT in str(exc):
            msg = f"{env_var} was rejected: password authentication failed."
            raise ConfigurationError(msg) from exc
        raise
    await conn.close()


def _sql_error_result(error: str, *, error_kind: SqlErrorKind) -> dict[str, Any]:
    """Build a standard `search_rooms` error result dict.

    Args:
        error: The error message to include in the result.
        error_kind: Coarse, message-independent classification of the
            failure.

    Returns:
        A result dict with ``status="error"`` and no rooms.

    """
    return {
        "status": "error",
        "matching_count": 0,
        "rooms": [],
        "error": error,
        "error_kind": error_kind,
    }


def _timing(started: float) -> dict[str, int]:
    """Measure the time since `started`, as a log record's extra fields.

    Args:
        started: `time.perf_counter` reading when the work began.

    Returns:
        dict[str, int]: ``duration_ms``, whole milliseconds elapsed.

    """
    return {"duration_ms": round((time.perf_counter() - started) * 1000)}


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
    _search_args_model: type[BaseModel] | None
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
        self.proposals = ProposalStore(ttl_s=config.proposals.ttl_s)
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

        Raises:
            ConfigurationError: If either database URL is missing, blank, or
                has its password rejected, or if the read-only pool's role is
                not actually read-only --
                treated as a fatal misconfiguration rather than a warning,
                since a silently-writable "read-only" pool is exactly the
                guarantee this design depends on. None of these resolve by
                retrying, so they are kept out of `OperationalError`.
            OperationalError: If resources cannot be initialized for a
                transient reason (e.g. the database is unreachable).

        """
        try:
            _require_url(self.pgsql_ro_db_url, "PGSQL_RO_DB_URL")
            _require_url(self.pgsql_rw_db_url, "PGSQL_RW_DB_URL")
            timeout_s = self.config.db.pool.timeout_s
            await _check_credentials(
                self.pgsql_ro_db_url, "PGSQL_RO_DB_URL", timeout_s=timeout_s,
            )
            await _check_credentials(
                self.pgsql_rw_db_url, "PGSQL_RW_DB_URL", timeout_s=timeout_s,
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

    def get_search_args_model(self) -> type[BaseModel]:
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

    async def search_rooms(self, args: BaseModel) -> dict[str, Any]:
        """Run a room search and return a bounded result.

        Each attempt borrows a connection from the read-only pool and runs the
        fixed query `search.run_room_search` selects for *args*. A search is a
        read, so it is always safe to retry on a transient connection error.

        Args:
            args: An instance of `get_search_args_model()`.

        Returns:
            On success, `run_room_search`'s result: ``status="ok"``,
            ``matching_count``, and at most ``top_k`` ``rooms``. On failure,
            ``status="error"`` with ``error`` and ``error_kind``.

        Raises:
            RuntimeError: If resources were not initialized.

        """
        require(self.pool, "BookingSqlResources")
        retry_cfg = self.config.db.retry

        _conn_errors = (
            psycopg.OperationalError,
            psycopg.InterfaceError,
            PoolTimeout,
            TimeoutError,
        )

        def _is_retryable(exc: BaseException) -> bool:
            return isinstance(exc, _conn_errors) and is_transient_conn_error(exc)

        # Spans every attempt and backoff, so a cold Neon start shows up here.
        started = time.perf_counter()
        try:
            async for attempt in AsyncRetrying(
                retry=retry_if_exception(_is_retryable),
                stop=stop_after_attempt(retry_cfg.max_transient_retries + 1),
                wait=wait_exponential(multiplier=retry_cfg.transient_retry_backoff_s),
                before_sleep=before_sleep_log(logger, logging.WARNING),
                reraise=True,
            ):
                with attempt:
                    result = await self._search_once(args)
                    # The arguments, never the rooms: the counts are all an
                    # audit needs to see what the model looked at.
                    timing = _timing(started)
                    logger.info(
                        "search_rooms ok: matching_count=%s returned=%s "
                        "duration_ms=%s args=%r",
                        result["matching_count"],
                        len(result["rooms"]),
                        timing["duration_ms"],
                        args,
                        extra=timing,
                    )
                    return result

        except _conn_errors as exc:
            # One line, no traceback: the stack under a pool checkout is
            # psycopg_pool internals and says nothing the exception does not.
            timing = _timing(started)
            logger.warning(
                "search_rooms connection error after retries: duration_ms=%s %r",
                timing["duration_ms"],
                exc,
                extra=timing,
            )
            return _sql_error_result(
                _tool_error_message_for_model(), error_kind="unavailable",
            )

        except Exception:
            # The SQL is fixed and the arguments validated, so anything else
            # (a psycopg.Error included) is a bug, not something the model
            # can fix by searching differently.
            timing = _timing(started)
            logger.exception(
                "search_rooms unexpected failure: duration_ms=%s args=%r",
                timing["duration_ms"],
                args,
                extra=timing,
            )
            return _sql_error_result(
                _tool_error_message_for_model(), error_kind="unexpected",
            )

        # Unreachable: AsyncRetrying either returns from the loop body or
        # reraises. Kept so a broken assumption fails closed, not silently.
        return _sql_error_result(
            _tool_error_message_for_model(), error_kind="unexpected",
        )

    async def _search_once(self, args: BaseModel) -> dict[str, Any]:
        """Run the search exactly once without any retry logic.

        Args:
            args: Validated search arguments.

        Returns:
            `run_room_search`'s result.

        Raises:
            psycopg.OperationalError: On connection-level failures.
            psycopg.InterfaceError: On connection-level failures.
            psycopg_pool.PoolTimeout: When a pool connection cannot be acquired.
            TimeoutError: On network timeout.
            psycopg.Error: On any other database error.

        """
        async with (
            self.get_read_pool().connection(
                timeout=self.config.db.pool.timeout_s,
            ) as conn,
            conn.transaction(),
        ):
            # Belt-and-braces: with a genuinely read-only role this can never
            # engage, but it costs nothing and covers the window between a
            # misconfiguration and the next restart's startup assertion.
            await conn.execute("SET TRANSACTION READ ONLY")
            return await run_room_search(conn, args, top_k=self.config.agent.top_k)

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

        A health-check (``SELECT 1``) is run each time a connection is
        checked out from either pool so stale connections are discarded
        before they reach a caller. Combined with ``max_idle``, this ensures
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

        Guards against `PGSQL_RO_DB_URL` being misconfigured to point at the
        same (writable) role as `PGSQL_RW_DB_URL` -- everything would
        otherwise work, and the guarantee this whole design depends on would
        be gone with nothing to notice.

        Raises:
            OperationalError: If this is called before `_open_pools()`, or if
                the probe itself cannot be run (e.g. the database is
                unreachable) -- both transient, from this method's point of
                view.
            ConfigurationError: If the probe write is not refused. This is
                `PGSQL_RO_DB_URL` pointed at a writable role, which no retry
                fixes; only a corrected environment variable and a restart
                do.

        """
        if self.pool is None:
            msg = "_assert_read_pool_is_read_only() called before _open_pools()"
            raise OperationalError(msg)
        try:
            async with self.pool.connection() as conn, conn.transaction():
                await conn.execute(_READ_ONLY_PROBE_SQL)
        except (
            psycopg.errors.ReadOnlySqlTransaction,
            psycopg.errors.InsufficientPrivilege,
        ):
            return
        msg = (
            "PGSQL_RO_DB_URL permitted a write. It must authenticate as a "
            "role with no write privileges (bh_agent_ro) -- refusing to "
            "start with the read-only guarantee unverified."
        )
        raise ConfigurationError(msg)

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
            self._search_args_model = build_search_args_model(
                meta, max_room_numbers=self.config.agent.max_search_room_numbers,
            )
            self.rooms_metadata = meta

            template = load_prompt_template(self._system_prompt_resource)
            self.system_prompt = render_system_prompt(
                template=template, top_k=self.config.agent.top_k, meta=meta,
            )

        except ConfigurationError:
            raise
        except Exception as exc:
            msg = "Failed to render booking system prompt"
            raise OperationalError(msg) from exc
