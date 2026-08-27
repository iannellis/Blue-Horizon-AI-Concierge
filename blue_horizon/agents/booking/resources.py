"""Long-lived resources for the booking SQL agent.

Owns two async connection pools -- a read-only pool used exclusively by the
model-facing `run_sql` tool, and a read-write pool used by write_ops,
list_bookings, and the customers/bookings API endpoints -- plus the rendered
system prompt and the in-process proposal store.

The read-only pool authenticates as the `bh_agent_ro` Postgres role, whose
grants are exactly the guardrail's table allowlist (see
`blue_horizon/load_data/regrant_booking_agent_role.sql`). This is the actual
enforcement boundary: the AST guardrail in `guardrails.py` is a redundant,
code-level restatement of the same rule, not the thing keeping the model from
writing.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, LiteralString, cast

import psycopg
from psycopg.rows import dict_row
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
    _is_transient_conn_error,
    _truncate_rows,
    _user_facing_db_message,
    fetch_rooms_metadata,
)
from blue_horizon.agents.booking.guardrails import validate_sql
from blue_horizon.agents.booking.proposals import ProposalStore
from blue_horizon.agents.exceptions import OperationalError
from blue_horizon.agents.prompt_utils import load_prompt_template, prompt_resource_path

if TYPE_CHECKING:
    from blue_horizon.config import BookingSqlConfig

logger = logging.getLogger(__name__)

# A no-op UPDATE that matches no row: used only to prove the read-only pool's
# role is actually refused write privileges at startup. Never mutates data
# even if the assertion this guards against has already failed.
_READ_ONLY_PROBE_SQL = "UPDATE room_availability SET status = status WHERE id = -1"


def _sql_error_result(error: str) -> dict[str, Any]:
    """Build a standard SQL tool error result dict.

    Args:
        error: The error message to include in the result.

    Returns:
        A result dict with ``status="error"`` and zero rows.

    """
    return {
        "status": "error",
        "rowcount": 0,
        "rows": [],
        "truncated": False,
        "error": error,
    }


class BookingSqlResources:
    """Own long-lived resources for the booking SQL agent.

    Both pools, the rendered system prompt, and the proposal store live here.

    Attributes:
        config: Parsed configuration.
        pgsql_ro_db_url: Read-only database URL (`bh_agent_ro`), used only by
            `run_sql`.
        pgsql_rw_db_url: Read-write database URL (`bh_agent_rw`), used by
            write_ops, list_bookings, and the propose_* tools.
        pool: Async connection pool for `pgsql_ro_db_url`.
        write_pool: Async connection pool for `pgsql_rw_db_url`.
        proposals: In-process store of pending booking proposals.
        system_prompt: Rendered system prompt used to build the agent.

    """

    config: BookingSqlConfig
    pgsql_ro_db_url: str
    pgsql_rw_db_url: str
    pool: AsyncConnectionPool[Any] | None
    write_pool: AsyncConnectionPool[Any] | None
    proposals: ProposalStore
    system_prompt: str | None
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

        self._system_prompt_resource = prompt_resource_path(
            self.config.prompts.folder, self.config.prompts.system_prompt_filename,
        )

    async def startup_check(self) -> None:
        """Initialize resources and validate readiness.

        Opens both pools, proves the read-only pool's role really cannot
        write, fetches metadata used by the system prompt template, and
        renders the final system prompt.

        Raises:
            OperationalError: If resources cannot be initialized, or if the
                read-only pool's role is not actually read-only -- treated as
                a fatal misconfiguration rather than a warning, since a
                silently-writable "read-only" pool is exactly the guarantee
                this design depends on.

        """
        try:
            await self._open_pools()
            await self._assert_read_pool_is_read_only()
            await self._render_system_prompt()
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

    async def execute_sql(self, query: str) -> dict[str, Any]:
        """Execute a single read-only SQL statement and return rows.

        Each attempt borrows a connection from the read-only pool
        (``search_path`` and ``statement_timeout`` are expected to be set at
        the database role level rather than in code) and executes *query*
        directly. All SQL through this path is a read, so it is always safe
        to retry on a transient connection error.

        Args:
            query: One SQL statement (no semicolons). Must be SELECT.

        Returns:
            Dict with keys:

                - status: str ("ok" or "error")
                - rows: list[dict[str, Any]]
                - truncated: bool
                - rowcount: int
                - error: str (only present on failure)

        Raises:
            RuntimeError: If resources were not initialized.

        """
        require(self.pool, "BookingSqlResources")

        try:
            validate_sql(
                query,
                allow_only_hotel_tables=self.config.db.guardrails.allow_only_hotel_tables,
            )
        except ValueError as exc:
            msg = str(exc)
            logger.info("run_sql rejected by guardrails: %s", msg)
            return _sql_error_result(msg)

        retry_cfg = self.config.db.retry

        _conn_errors = (
            psycopg.OperationalError,
            psycopg.InterfaceError,
            PoolTimeout,
            TimeoutError,
        )
        _privilege_errors = (
            psycopg.errors.ReadOnlySqlTransaction,
            psycopg.errors.InsufficientPrivilege,
        )

        def _is_retryable(exc: BaseException) -> bool:
            return isinstance(exc, _conn_errors) and _is_transient_conn_error(exc)

        # Default only reached if AsyncRetrying's loop completes without
        # returning or raising, which tenacity does not do in practice -- this
        # guards against a silent UnboundLocalError if that assumption ever
        # breaks.
        error_message = _user_facing_db_message()
        try:
            async for attempt in AsyncRetrying(
                retry=retry_if_exception(_is_retryable),
                stop=stop_after_attempt(retry_cfg.max_transient_retries + 1),
                wait=wait_exponential(multiplier=retry_cfg.transient_retry_backoff_s),
                before_sleep=before_sleep_log(logger, logging.WARNING),
                reraise=True,
            ):
                with attempt:
                    return await self._execute_once(query)

        except _conn_errors:
            logger.warning("run_sql connection error after retries", exc_info=True)
            error_message = _user_facing_db_message()

        except _privilege_errors as exc:
            # A write blocked by the read-only role or the belt-and-braces
            # READ ONLY transaction wrapper. This is not a SQL error to
            # diagnose and retry -- writes are simply not available on this
            # path -- so it is kept out of the generic psycopg.Error branch
            # below, which the prompt's retry instructions would otherwise
            # treat as "rewrite the query and try again".
            logger.info("run_sql blocked a write attempt: %s", exc)
            error_message = (
                "Writes are not available through this tool. Use the propose "
                "tools to book, cancel, or modify a reservation."
            )

        except psycopg.Error as exc:
            # SQL-level errors (type mismatches, syntax errors, constraint
            # violations, etc.) — expose the error text so the agent can
            # diagnose and rewrite the query per its retry instructions.
            logger.warning("run_sql SQL error: %s", exc)
            error_message = f"SQL error: {exc}"

        except Exception:
            logger.exception("run_sql unexpected failure")
            error_message = _user_facing_db_message()

        return _sql_error_result(error_message)

    async def _execute_once(self, query: str) -> dict[str, Any]:
        """Execute the SQL statement exactly once without any retry logic.

        Args:
            query: SQL statement that has already passed ``validate_sql``.

        Returns:
            Dict with keys ``status``, ``rows``, ``truncated``, and ``rowcount``.

        Raises:
            psycopg.OperationalError: On connection-level failures.
            psycopg.InterfaceError: On connection-level failures.
            psycopg_pool.PoolTimeout: When a pool connection cannot be acquired.
            TimeoutError: On network timeout.
            psycopg.Error: On SQL-level errors (syntax, constraints, types),
                including a blocked write.
            Exception: On any other unexpected failure.

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

            async with conn.cursor(row_factory=dict_row) as cur:
                # NOTE: psycopg's type stubs expect a LiteralString for
                # `execute()`. This cast is only to satisfy static type
                # checkers. Runtime safety is provided by `validate_sql(...)`
                # above and by the database role's own grants.
                await cur.execute(cast("LiteralString", query))

                if cur.description is None:
                    return {
                        "status": "ok",
                        "rows": [],
                        "truncated": False,
                        "rowcount": cur.rowcount,
                    }

                rows_raw = await cur.fetchall()
                rows = [dict(r) for r in rows_raw]
                rows, truncated = _truncate_rows(
                    rows,
                    max_rows=self.config.db.guardrails.max_rows,
                )
                return {
                    "status": "ok",
                    "rows": rows,
                    "truncated": truncated,
                    "rowcount": len(rows),
                }

    async def _open_pools(self) -> None:
        """Open the read-only and read-write async connection pools.

        ``search_path`` and ``statement_timeout`` are expected to be set at
        the database role level (``ALTER ROLE … SET …``) so they apply
        consistently under PgBouncer transaction pooling without any
        per-connection ``SET`` commands.

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
            OperationalError: If the probe write is not refused, or if this
                is called before `_open_pools()`.

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
        raise OperationalError(msg)

    async def _render_system_prompt(self) -> None:
        """Render and store the system prompt.

        Raises:
            OperationalError: If prompt rendering fails.

        """
        try:
            (
                enum_values,
                basic_amenities,
                additional_amenities,
                view_types,
            ) = await fetch_rooms_metadata(self.pgsql_ro_db_url)

            template = load_prompt_template(self._system_prompt_resource)
            self.system_prompt = render_system_prompt(
                template=template,
                top_k=self.config.agent.top_k,
                enum_values=enum_values,
                basic_amenities=basic_amenities,
                additional_amenities=additional_amenities,
                view_types=view_types,
            )

        except Exception as exc:
            msg = "Failed to render booking system prompt"
            raise OperationalError(msg) from exc
