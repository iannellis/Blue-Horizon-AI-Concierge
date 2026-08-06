"""Long-lived resources for the booking SQL agent.

Owns the async connection pool, rendered system prompt, and SQL execution
with guardrails and retry logic.
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

from blue_horizon.agents.booking.config import render_system_prompt
from blue_horizon.agents.booking.db_utils import (
    _is_transient_conn_error,
    _truncate_rows,
    _user_facing_db_message,
    fetch_rooms_metadata,
)
from blue_horizon.agents.booking.guardrails import _is_write_sql, validate_sql
from blue_horizon.agents.exceptions import OperationalError
from blue_horizon.agents.prompt_utils import load_prompt_template

if TYPE_CHECKING:
    from blue_horizon.config import BookingSqlConfig

logger = logging.getLogger(__name__)


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

    The pool, rendered system prompt, and helper methods live here.

    Attributes:
        config: Parsed configuration.
        pgsql_db_url: Database URL.
        pool: Async connection pool used for SQL execution.
        system_prompt: Rendered system prompt used to build the agent.

    """

    __slots__ = (
        "_system_prompt_resource",
        "config",
        "pgsql_db_url",
        "pool",
        "system_prompt",
    )

    config: BookingSqlConfig
    pgsql_db_url: str
    pool: AsyncConnectionPool[Any] | None
    system_prompt: str | None
    _system_prompt_resource: str

    def __init__(self, *, config: BookingSqlConfig, pgsql_db_url: str) -> None:
        """Construct booking SQL resources.

        This initializer performs only lightweight setup and validation:
        - Store configuration and the database URL.
        - Resolve the prompts directory and system prompt template path.

        Call ``await startup_check()`` to open the pool and render the prompt.

        Args:
            config: Parsed configuration loaded from TOML.
            pgsql_db_url: Database URL.

        Raises:
            RuntimeError: If the prompt template folder or template file is missing.

        """
        self.config = config
        self.pgsql_db_url = pgsql_db_url
        self.pool: AsyncConnectionPool[Any] | None = None
        self.system_prompt = None

        prompts_folder = self.config.prompts.folder.strip("/")
        if prompts_folder:
            self._system_prompt_resource = (
                f"{prompts_folder}/{self.config.prompts.system_prompt_filename}"
            )
        else:
            self._system_prompt_resource = self.config.prompts.system_prompt_filename

    async def startup_check(self) -> None:
        """Initialize resources and validate readiness.

        This method opens the database pool, fetches metadata used by the system
        prompt template, and renders the final system prompt.

        Raises:
            OperationalError: If resources cannot be initialized.

        """
        try:
            await self._open_pool()
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
        if self.system_prompt is None:
            msg = (
                "BookingSqlResources not initialized; call await startup_check() first"
            )
            raise RuntimeError(msg)
        return self.system_prompt

    async def aclose(self) -> None:
        """Close resources owned by this instance.

        This method is idempotent.

        """
        if self.pool is not None:
            await self.pool.close()
        self.pool = None
        self.system_prompt = None

    async def execute_sql(self, query: str) -> dict[str, Any]:
        """Execute a single SQL statement and return rows.

        Each attempt borrows a connection from the pool (``search_path`` and
        ``statement_timeout`` are expected to be set at the database role
        level rather than in code) and executes *query* directly.

        Transient connection errors are retried with exponential backoff.
        Write queries are not retried by default because the pool connection
        may have committed the write before the error was surfaced.  Set
        ``retry_writes_on_transient_errors = true`` in config to override.

        Args:
            query: One SQL statement (no semicolons). May be SELECT or DML. If DML
                uses RETURNING, returned rows are provided.

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
        if self.pool is None:
            msg = (
                "BookingSqlResources not initialized; call await startup_check() first"
            )
            raise RuntimeError(msg)

        try:
            validate_sql(
                query,
                allow_only_hotel_tables=self.config.db.guardrails.allow_only_hotel_tables,
            )
        except ValueError as exc:
            msg = str(exc)
            logger.info("run_sql rejected by guardrails: %s", msg)
            return {
                "status": "error",
                "rowcount": 0,
                "rows": [],
                "truncated": False,
                "error": msg,
            }

        is_write = _is_write_sql(query)
        retry_cfg = self.config.db.retry

        _conn_errors = (
            psycopg.OperationalError,
            psycopg.InterfaceError,
            PoolTimeout,
            TimeoutError,
        )

        def _is_retryable(exc: BaseException) -> bool:
            return (
                isinstance(exc, _conn_errors)
                and _is_transient_conn_error(exc)
                and (not is_write or retry_cfg.retry_writes_on_transient_errors)
            )

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
            return _sql_error_result(_user_facing_db_message())

        except psycopg.Error as exc:
            # SQL-level errors (type mismatches, syntax errors, constraint
            # violations, etc.) — expose the error text so the agent can
            # diagnose and rewrite the query per its retry instructions.
            logger.warning("run_sql SQL error: %s", exc)
            return _sql_error_result(f"SQL error: {exc}")

        except Exception:
            logger.exception("run_sql unexpected failure")
            return _sql_error_result(_user_facing_db_message())

        # Unreachable: AsyncRetrying always raises or returns inside the loop.
        return _sql_error_result(_user_facing_db_message())  # pragma: no cover

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
            psycopg.Error: On SQL-level errors (syntax, constraints, types).
            Exception: On any other unexpected failure.

        """
        async with (
            self.pool.connection(timeout=self.config.db.pool.timeout_s) as conn,  # type: ignore[union-attr]
            conn.cursor(row_factory=dict_row) as cur,
        ):
            # NOTE: psycopg's type stubs expect a LiteralString for `execute()`.
            # This cast is only to satisfy static type checkers. Runtime safety
            # is provided by `validate_sql(...)` above.
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

    async def _open_pool(self) -> None:
        """Open the async connection pool.

        ``search_path`` and ``statement_timeout`` are expected to be set at
        the database role level (``ALTER ROLE … SET …``) so they apply
        consistently under PgBouncer transaction pooling without any
        per-connection ``SET`` commands.

        A health-check (``SELECT 1``) is run each time a connection is
        checked out from the pool so stale connections are discarded before
        they reach ``execute_sql``.  Combined with ``max_idle``, this
        ensures the pool never hands out connections that Neon has already
        dropped due to compute suspension.

        Raises:
            OperationalError: If the pool cannot be opened.

        """
        if self.pool is not None:
            return

        async def configure_connection(conn: psycopg.AsyncConnection[Any]) -> None:
            """Enable autocommit on each new connection.

            Args:
                conn: Newly created async psycopg connection.

            """
            await conn.set_autocommit(True)

        try:
            pool = AsyncConnectionPool(
                conninfo=self.pgsql_db_url,
                min_size=self.config.db.pool.min_size,
                max_size=self.config.db.pool.max_size,
                timeout=self.config.db.pool.timeout_s,
                max_idle=self.config.db.pool.max_idle_s,
                configure=configure_connection,
                check=AsyncConnectionPool.check_connection,
                open=False,
            )
            await pool.open()
        except Exception as exc:
            msg = "Failed to open booking DB pool"
            raise OperationalError(msg) from exc

        self.pool = pool

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
            ) = await fetch_rooms_metadata(self.pgsql_db_url)

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
