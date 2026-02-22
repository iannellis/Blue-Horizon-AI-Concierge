"""Long-lived resources for the rooms SQL agent.

Owns the async connection pool, rendered system prompt, and SQL execution
with guardrails and retry logic.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, LiteralString, cast

import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool, PoolTimeout

from blue_horizon.agents.exceptions import OperationalError
from blue_horizon.agents.rooms.config import (
    load_prompt_template_for_rooms,
    render_system_prompt,
)
from blue_horizon.agents.rooms.db_utils import (
    _is_transient_conn_error,
    _sleep_backoff,
    _truncate_rows,
    _user_facing_db_message,
    fetch_rooms_metadata,
)
from blue_horizon.agents.rooms.guardrails import _is_write_sql, validate_sql

if TYPE_CHECKING:
    from blue_horizon.config import RoomsSqlConfig

logger = logging.getLogger(__name__)


class RoomsSqlResources:
    """Own long-lived resources for the rooms SQL agent.

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

    config: RoomsSqlConfig
    pgsql_db_url: str
    pool: AsyncConnectionPool[Any] | None
    system_prompt: str | None
    _system_prompt_resource: str

    def __init__(self, *, config: RoomsSqlConfig, pgsql_db_url: str) -> None:
        """Construct rooms SQL resources.

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
            msg = "Rooms SQL resources failed during startup"
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
            msg = "RoomsSqlResources not initialized; call await startup_check() first"
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
            msg = "RoomsSqlResources not initialized; call await startup_check() first"
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
        attempts = self.config.db.retry.max_transient_retries + 1

        for attempt in range(attempts):
            try:
                async with (
                    self.pool.connection(timeout=self.config.db.pool.timeout_s) as conn,
                    conn.cursor(row_factory=dict_row) as cur,
                ):
                    await cur.execute("SET search_path TO public;")

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

            except (
                psycopg.OperationalError,
                psycopg.InterfaceError,
                PoolTimeout,
                TimeoutError,
            ) as exc:
                can_retry = (
                    _is_transient_conn_error(exc)
                    and attempt < attempts - 1
                    and (
                        (not is_write)
                        or self.config.db.retry.retry_writes_on_transient_errors
                    )
                )

                if can_retry:
                    logger.warning(
                        "run_sql transient DB/pool error; retrying (attempt=%s/%s)",
                        attempt + 1,
                        attempts,
                        exc_info=True,
                    )
                    await _sleep_backoff(
                        self.config.db.retry.transient_retry_backoff_s,
                        attempt,
                    )
                    continue

                logger.warning(
                    "run_sql DB/pool operation failed (attempt=%s/%s)",
                    attempt + 1,
                    attempts,
                    exc_info=True,
                )
                return {
                    "status": "error",
                    "rowcount": 0,
                    "rows": [],
                    "truncated": False,
                    "error": _user_facing_db_message(),
                }

            except Exception:
                logger.exception("run_sql unexpected failure")
                return {
                    "status": "error",
                    "rowcount": 0,
                    "rows": [],
                    "truncated": False,
                    "error": _user_facing_db_message(),
                }

        logger.warning("run_sql failed after retries")
        return {
            "status": "error",
            "rowcount": 0,
            "rows": [],
            "truncated": False,
            "error": _user_facing_db_message(),
        }

    async def _open_pool(self) -> None:
        """Open the async connection pool.

        Raises:
            OperationalError: If the pool cannot be opened.

        """
        if self.pool is not None:
            return

        timeout_ms = self.config.db.timeouts.statement_timeout_ms

        async def configure_connection(conn: psycopg.AsyncConnection[Any]) -> None:
            """Configure each new connection on checkout.

            Args:
                conn: Newly created async psycopg connection.

            """
            await conn.set_autocommit(True)

            if timeout_ms <= 0:
                return

            async with conn.cursor() as cur:
                await cur.execute(
                    sql.SQL("SET statement_timeout = {}").format(
                        sql.Literal(timeout_ms),
                    ),
                )

        try:
            pool = AsyncConnectionPool(
                conninfo=self.pgsql_db_url,
                min_size=self.config.db.pool.min_size,
                max_size=self.config.db.pool.max_size,
                timeout=self.config.db.pool.timeout_s,
                configure=configure_connection,
                open=False,
            )
            await pool.open()
        except Exception as exc:
            msg = "Failed to open rooms DB pool"
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

            template = load_prompt_template_for_rooms(self._system_prompt_resource)
            self.system_prompt = render_system_prompt(
                template=template,
                top_k=self.config.agent.top_k,
                enum_values=enum_values,
                basic_amenities=basic_amenities,
                additional_amenities=additional_amenities,
                view_types=view_types,
            )

        except Exception as exc:
            msg = "Failed to render rooms system prompt"
            raise OperationalError(msg) from exc
