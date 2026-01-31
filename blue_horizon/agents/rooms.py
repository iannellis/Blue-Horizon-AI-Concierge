"""Provide a rooms SQL agent module.

This module defines an async LangGraph/LangChain agent backed by a PostgreSQL
 database. It follows the same separation of concerns as
`information_agent.py`:

- A resources class owns long-lived dependencies and performs startup checks.
- A factory class builds the agent and binds tools that call into resources.

Configuration is loaded from a TOML file into typed dataclasses. Secrets (e.g.,
PGSQL_DB_URL, API keys) come from environment variables.

FastAPI integration (suggested):
- Create `RoomsSqlResources` at startup and call `await resources.startup_check()`.
- Build the agent once with `RoomsAgentFactory(...).build()` and store it on
  `app.state`.
- Close resources at shutdown with `await resources.aclose()`.

"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from contextvars import ContextVar, Token
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Final, LiteralString, cast

import psycopg
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from psycopg import sql
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool, PoolTimeout

from blue_horizon.agents.exceptions import OperationalError
from blue_horizon.agents.prompt_utils import load_prompt_template
from blue_horizon.config import RoomsSqlConfig, load_app_config

if TYPE_CHECKING:
    from pathlib import Path
    from string import Template

    from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)

# Schemas for evaluating the agent
EVAL_SCHEMA: ContextVar[str | None] = ContextVar("EVAL_SCHEMA", default=None)

def set_eval_schema(schema: str | None) -> Token[str | None]:
    """Assign a schema to the evaluation ContextVar.

    Args:
        schema: Schema name to apply, or ``None`` to unset.

    Returns:
        Token used to reset the ContextVar to its prior state.

    """
    return EVAL_SCHEMA.set(schema)

def reset_eval_schema(token: Token[str | None]) -> None:
    """Restore the evaluation schema ContextVar to a previous value.

    Args:
        token: Token returned from :func:`set_eval_schema` representing the
            previous value.

    """
    try:
        EVAL_SCHEMA.reset(token)
    except ValueError:
        logger.warning(
            "Eval schema ContextVar reset failed due to context mismatch; "
            "clearing schema instead.",
        )
        EVAL_SCHEMA.set(None)

# Enums used in SQL
ENUM_TYPES: Final[tuple[str, ...]] = (
    "availability_status_type",
    "room_bed_type",
    "room_status_type",
    "room_type",
)


# ============================
# Settings (loaded from TOML config)
# ============================


def load_rooms_config(config_path: Path | str | None = None) -> RoomsSqlConfig:
    """Load the rooms SQL configuration section.

    For when using the rooms agent standalone.

    Args:
        config_path: Optional path to override the packaged config. If unset,
            ``app_config.toml`` from the package resources is used.

    Returns:
        RoomsSqlConfig: Parsed configuration for the rooms agent.

    """
    app_config = load_app_config(path=config_path)
    return app_config.rooms


# ============================
# Environment and user-facing messages
# ============================


@lru_cache(maxsize=1)
def get_pgsql_db_url() -> str:
    """Get the PostgreSQL connection URL from the environment.

    Returns:
        PostgreSQL connection URL.

    Raises:
        RuntimeError: If PGSQL_DB_URL is not set.

    """
    url = os.getenv("PGSQL_DB_URL")
    if not url:
        msg = "PGSQL_DB_URL is not set"
        raise RuntimeError(msg)
    return url


def _user_facing_db_message() -> str:
    """Return a user-facing message for database operational failures.

    Returns:
        A short message suitable for returning to end users when the database is
        unavailable.

    """
    return (
        "The booking system is temporarily unavailable. Please try again in a moment."
    )


# ============================
# Prompt loading
# ============================


def render_system_prompt(  # noqa: PLR0913
    *,
    template: Template,
    top_k: int,
    enum_values: dict[str, list[str]],
    basic_amenities: list[str],
    additional_amenities: list[str],
    view_types: list[str],
) -> str:
    """Render the system prompt template with runtime substitutions."""
    return template.safe_substitute(
        top_k=top_k,
        room_type=enum_values.get("room_type", []),
        basic_amenities=basic_amenities,
        additional_amenities=additional_amenities,
        room_bed_type=enum_values.get("room_bed_type", []),
        view_types=view_types,
        room_status_type=enum_values.get("room_status_type", []),
        availability_status_type=enum_values.get("availability_status_type", []),
    )


# ============================
# DB metadata fetch (async)
# ============================


async def fetch_rooms_metadata(
    pgsql_db_url: str,
) -> tuple[dict[str, list[str]], list[str], list[str], list[str]]:
    """Fetch database metadata used to fill the system prompt.

    The system prompt template is populated with:
      - Enum values for known enum types.
      - Distinct values from array columns in the rooms table.

    Args:
        pgsql_db_url: Database URL.

    Returns:
        Tuple of (enum_values, basic_amenities, additional_amenities, view_types).

    Raises:
        OperationalError: If metadata queries fail.

    """
    enum_values: dict[str, list[str]] = {}

    try:
        async with (
            await psycopg.AsyncConnection.connect(pgsql_db_url) as conn,
            conn.cursor() as cur,
        ):
            await cur.execute("SET search_path TO public;")
            for enum_type in ENUM_TYPES:
                query = sql.SQL("SELECT enum_range(NULL::{});").format(
                    sql.Identifier(enum_type),
                )
                await cur.execute(query)
                row = await cur.fetchone()
                if not row or row[0] is None:
                    enum_values[enum_type] = []
                    continue

                raw = str(row[0]).strip("{}")
                enum_values[enum_type] = [
                    v.strip().strip('"').strip("'") for v in raw.split(",") if v.strip()
                ]

            await cur.execute("SELECT DISTINCT unnest(basic_amenities) FROM rooms;")
            basic_amenities = [r[0] for r in await cur.fetchall()]

            await cur.execute(
                "SELECT DISTINCT unnest(additional_amenities) FROM rooms;",
            )
            additional_amenities = [r[0] for r in await cur.fetchall()]

            await cur.execute("SELECT DISTINCT unnest(view_type) FROM rooms;")
            view_types = [r[0] for r in await cur.fetchall()]

    except Exception as exc:
        msg = "Failed to fetch rooms metadata from the database"
        raise OperationalError(msg) from exc

    else:
        return enum_values, basic_amenities, additional_amenities, view_types


# ============================
# SQL guardrails (supports booking/cancel writes)
# ============================

_FORBIDDEN_TOKENS: Final[tuple[str, ...]] = (
    "pg_sleep",
    "information_schema",
    "pg_catalog",
    "dblink",
)

_FORBIDDEN_KEYWORDS = re.compile(
    r"\b("
    r"alter|create|drop|truncate|grant|revoke|vacuum|analyze|"
    r"copy|do|call|execute"
    r")\b",
    re.IGNORECASE,
)

_ALLOWED_TABLES: Final[set[str]] = {"rooms", "room_availability"}

_IDENTIFIER = r'(?:"[^"]+"|[a-zA-Z_][a-zA-Z0-9_]*)'
_TABLE_REF = re.compile(
    rf"\b(from|join|update|into)\s+({_IDENTIFIER})(?:\s*\.\s*({_IDENTIFIER}))?\b",
    re.IGNORECASE,
)


def _normalize_sql(query: str) -> str:
    """Normalize SQL text for lightweight validation.

    Args:
        query: Raw SQL string.

    Returns:
        SQL with collapsed whitespace.

    """
    return " ".join(query.strip().split())


def _normalize_identifier(identifier: str) -> str:
    """Normalize a SQL identifier for comparisons.

    Args:
        identifier: Identifier token, possibly quoted.

    Returns:
        Normalized identifier name.

    """
    token = identifier.strip()
    if token.startswith('"') and token.endswith('"') and len(token) >= 2:  # noqa: PLR2004
        token = token[1:-1]
    return token.lower()


def _contains_statement_separator(query: str) -> bool:  # noqa: C901, PLR0912, PLR0915
    """Detect whether a SQL string contains a statement separator.

    A semicolon is treated as a statement separator only when it appears outside
    of:
    - Single-quoted string literals.
    - Double-quoted identifiers.
    - Line comments (starting with `--`).
    - Block comments (between `/*` and `*/`).

    This is a lightweight scanner intended to avoid false rejections (e.g., a
    semicolon inside a string literal) while still enforcing the single-statement
    rule.

    Args:
        query: Raw SQL string.

    Returns:
        True if an unquoted/uncommented semicolon is present; otherwise False.

    """
    in_single = False
    in_double = False
    in_line_comment = False
    in_block_comment = False

    i = 0
    n = len(query)
    while i < n:
        ch = query[i]
        nxt = query[i + 1] if i + 1 < n else ""

        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
            i += 1
            continue

        if in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
                i += 2
            else:
                i += 1
            continue

        if not in_single and not in_double:
            if ch == "-" and nxt == "-":
                in_line_comment = True
                i += 2
                continue
            if ch == "/" and nxt == "*":
                in_block_comment = True
                i += 2
                continue

        if in_single:
            if ch == "'":
                if nxt == "'":
                    i += 2
                    continue
                in_single = False
            i += 1
            continue

        if in_double:
            if ch == '"':
                if nxt == '"':
                    i += 2
                    continue
                in_double = False
            i += 1
            continue

        if ch == "'":
            in_single = True
            i += 1
            continue

        if ch == '"':
            in_double = True
            i += 1
            continue

        if ch == ";":
            return True

        i += 1

    return False


def validate_sql(query: str, *, allow_only_hotel_tables: bool) -> None:
    """Validate a SQL statement against conservative guardrails.

    Guardrails enforced:
      - Exactly one statement per call (no statement separators).
      - Blocks common catalog/extension escape hatches.
      - Blocks DDL and privileged commands.
      - Optionally restricts table references to an allowlist.

    Args:
        query: SQL statement to validate.
        allow_only_hotel_tables: Whether to enforce a table allowlist.

    Raises:
        ValueError: If the SQL violates any guardrail.

    """
    if _contains_statement_separator(query):
        msg = (
            "Only a single SQL statement is allowed per call (no semicolons). "
            "For multi-step workflows, call the tool multiple times."
        )
        raise ValueError(msg)

    q = _normalize_sql(query)
    q_lower = q.lower()

    for token in _FORBIDDEN_TOKENS:
        if token in q_lower:
            msg = f"Forbidden SQL token detected: {token}"
            raise ValueError(msg)

    if _FORBIDDEN_KEYWORDS.search(q):
        msg = "DDL/privileged SQL statements are not allowed."
        raise ValueError(msg)

    if allow_only_hotel_tables:
        for _, part1, part2 in _TABLE_REF.findall(q):
            table_token = part2 or part1
            table = _normalize_identifier(table_token)
            if table not in _ALLOWED_TABLES:
                msg = f"Table not allowed: {table_token}"
                raise ValueError(msg)


# ============================
# Resources (long-lived dependencies)
# ============================


async def _sleep_backoff(base_s: float, attempt: int) -> None:
    """Sleep using exponential backoff.

    Args:
        base_s: Base backoff in seconds.
        attempt: Zero-based attempt index.

    """
    await asyncio.sleep(base_s * (2**attempt))


def _is_write_sql(query: str) -> bool:
    """Determine whether the SQL statement looks like a write (DML).

    Args:
        query: SQL statement.

    Returns:
        True if the statement contains INSERT, UPDATE, or DELETE.

    """
    q = _normalize_sql(query).lower()
    return bool(re.search(r"\b(insert|update|delete)\b", q))


def _is_transient_conn_error(exc: BaseException) -> bool:
    """Determine whether an exception looks like a transient failure.

    Args:
        exc: Exception raised by psycopg/psycopg_pool.

    Returns:
        True if the exception looks retryable (e.g., SSL close, pool timeout).

    """
    if isinstance(exc, (PoolTimeout, TimeoutError)):
        return True

    msg = str(exc).lower()
    patterns = (
        "ssl connection has been closed unexpectedly",
        "server closed the connection unexpectedly",
        "connection is closed",
        "connection not open",
        "terminating connection",
    )
    return any(p in msg for p in patterns)


def _truncate_rows(
    rows: list[dict[str, Any]],
    *,
    max_rows: int,
) -> tuple[list[dict[str, Any]], bool]:
    """Truncate DB result rows to a maximum length.

    Args:
        rows: Result rows.
        max_rows: Maximum number of rows to keep.

    Returns:
        Tuple of (rows_out, truncated).

    """
    if len(rows) <= max_rows:
        return rows, False
    return rows[:max_rows], True


class RoomsSqlResources:
    """Own long-lived resources for the rooms SQL agent.

    This mirrors the resources pattern used by `information_agent.py`. The pool,
    rendered system prompt, and helper methods live here.

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

        Call `await startup_check()` to open the pool and render the prompt.

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
                `startup_check()` has not been called or failed.

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
            logger.info(
                "run_sql rejected by guardrails: %s",
                msg,
            )
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
                    schema = EVAL_SCHEMA.get()
                    if schema:
                        await cur.execute(
                            sql.SQL("SET search_path TO {}").format(
                                sql.Identifier(schema),
                            ),
                        )
                    else:
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
            # Ensure DML (booking/cancel) commits for single-statement tool calls.
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
            msg = "Failed to render rooms system prompt"
            raise OperationalError(msg) from exc


# ============================
# Agent factory (builder)
# ============================


class RoomsAgentFactory:
    """Build the rooms agent using initialized resources.

    This mirrors the `AgentFactory` pattern used by `information_agent.py`.

    Attributes:
        config: Parsed configuration.
        resources: Initialized resources instance.

    """

    __slots__ = ("config", "resources")

    config: RoomsSqlConfig
    resources: RoomsSqlResources

    def __init__(self, *, config: RoomsSqlConfig, resources: RoomsSqlResources) -> None:
        """Construct the rooms agent factory.

        Args:
            config: Parsed configuration loaded from TOML.
            resources: Initialized resources instance.

        """
        self.config = config
        self.resources = resources

    def build(self) -> CompiledStateGraph:
        """Build and return a compiled rooms agent.

        Returns:
            Compiled LangGraph agent.

        Raises:
            RuntimeError: If resources were not initialized.

        """
        system_prompt = self.resources.get_system_prompt()

        llm = ChatOpenAI(
            model=self.config.llm.model,
            temperature=self.config.llm.temperature,
            reasoning={"effort": self.config.llm.reasoning_effort},
            timeout=self.config.llm.timeout_s,
            max_retries=self.config.llm.max_retries,
        )

        @tool(parse_docstring=True)
        async def run_sql(query: str) -> dict[str, Any]:
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

            """
            return await self.resources.execute_sql(query)

        return create_agent(
            model=llm,
            tools=[run_sql],
            system_prompt=system_prompt,
        )
