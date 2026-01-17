"""Provide a rooms SQL agent module.

This module defines an async LangGraph/LangChain agent backed by a PostgreSQL
(Neon) database. The agent uses a single tool, `run_sql`, to execute exactly one
SQL statement per tool call (no semicolons).

The `run_sql` tool applies lightweight guardrails to:
- Block DDL/privileged statements (e.g., DROP, ALTER, GRANT).
- Block common catalog/extension escape hatches (e.g., pg_catalog).
- Optionally restrict table references to a hotel table allowlist.
- Truncate large result sets for model safety.

Configuration is loaded from a TOML file into typed dataclasses. Secrets (e.g.,
NEON_DB_URL, API keys) come from environment variables.

FastAPI integration:
- Optionally call `initialize_rooms_agent()` at startup.
- In request handlers, call `await get_rooms_agent()` and then `agent.ainvoke(...)`.
- Call `shutdown_rooms_agent()` at shutdown.

"""

import asyncio
import logging
import os
import re
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from string import Template as StringTemplate
from typing import Any, Final, LiteralString, cast

import psycopg
from langchain.agents import create_agent
from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph
from psycopg import sql
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool, PoolTimeout

logger = logging.getLogger(__name__)


class OperationalError(RuntimeError):
    """Raise an operational error for expected failures.

    This exception type is used for failures that can occur during normal operation
    (e.g., transient connectivity issues, retrieval failures). Callers should
    generally log the error and return a safe, user-friendly response rather than
    crashing the request.

    """


ENUM_TYPES: Final[tuple[str, ...]] = (
    "availability_status_type",
    "room_bed_type",
    "room_status_type",
    "room_type",
)


# ============================
# Settings (loaded from TOML config)
# ============================


@dataclass(frozen=True, slots=True)
class LlmConfig:
    """Configure the ChatOpenAI client.

    Attributes:
        model: Model identifier passed to ChatOpenAI.
        temperature: Sampling temperature.
        reasoning_effort: Provider-specific reasoning control.
        timeout_s: Per-request network timeout in seconds.
        max_retries: Retry count for transient provider/network errors.

    """

    model: str
    temperature: float
    reasoning_effort: str
    timeout_s: float
    max_retries: int


@dataclass(frozen=True, slots=True)
class AgentConfig:
    """Configure agent prompting.

    Attributes:
        dialect: SQL dialect label used in the system prompt.
        top_k: Prompt parameter used by the system prompt template.

    """

    dialect: str
    top_k: int


@dataclass(frozen=True, slots=True)
class PromptsConfig:
    """Configure prompt template files.

    Attributes:
        folder: Folder (relative to this module) containing prompt templates.
        system_prompt: System prompt template filename (no path).

    """

    folder: str
    system_prompt: str


@dataclass(frozen=True, slots=True)
class DbPoolConfig:
    """Configure the client-side async connection pool.

    Notes:
        Neon may also pool on the server side. This pool controls concurrency and
        connection reuse within the application process.

    Attributes:
        min_size: Minimum number of connections to keep in the pool.
        max_size: Maximum number of connections to allow in the pool.
        timeout_s: Timeout (seconds) for acquiring a connection.

    """

    min_size: int
    max_size: int
    timeout_s: float


@dataclass(frozen=True, slots=True)
class DbGuardrailsConfig:
    """Configure SQL tool guardrails.

    Attributes:
        max_rows: Maximum number of rows returned to the model.
        allow_only_hotel_tables: Whether to enforce a table allowlist.

    """

    max_rows: int
    allow_only_hotel_tables: bool


@dataclass(frozen=True, slots=True)
class DbTimeoutsConfig:
    """Configure database-side timeouts.

    Attributes:
        statement_timeout_ms: Postgres statement_timeout in milliseconds.
            Use 0 to skip setting statement_timeout.

    """

    statement_timeout_ms: int


@dataclass(frozen=True, slots=True)
class DbRetryConfig:
    """Configure retries for transient database connection failures.

    Attributes:
        max_transient_retries: Number of retries after the initial attempt.
        transient_retry_backoff_s: Base backoff (seconds), exponential per attempt.
        retry_writes_on_transient_errors: Whether to retry DML statements. Default
            False to avoid accidental double-writes.

    """

    max_transient_retries: int
    transient_retry_backoff_s: float
    retry_writes_on_transient_errors: bool


@dataclass(frozen=True, slots=True)
class DbConfig:
    """Group database configuration.

    Attributes:
        pool: Client-side pool settings.
        guardrails: SQL validation and result-size guardrails.
        timeouts: Database-side statement timeout settings.
        retry: Transient connection retry policy.

    """

    pool: DbPoolConfig
    guardrails: DbGuardrailsConfig
    timeouts: DbTimeoutsConfig
    retry: DbRetryConfig


@dataclass(frozen=True, slots=True)
class RoomsSqlConfig:
    """Group top-level configuration loaded from TOML.

    Attributes:
        llm: LLM client configuration.
        agent: Prompting configuration.
        prompts: Prompt template configuration.
        db: Database configuration.

    """

    llm: LlmConfig
    agent: AgentConfig
    prompts: PromptsConfig
    db: DbConfig


def load_rooms_sql_config(config_path: Path) -> RoomsSqlConfig:
    """Load configuration from a TOML file.

    Mirrors the style used by `information_agent.py`.

    Args:
        config_path: Path to TOML config.

    Returns:
        RoomsSqlConfig: Parsed config.

    Raises:
        RuntimeError: If missing/unreadable TOML or required keys are missing.

    """
    path = config_path.expanduser().resolve()
    if not path.exists() or not path.is_file():
        msg = f"Config file not found: {path}"
        raise RuntimeError(msg)

    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        msg = f"Failed to read config file: {path}"
        raise RuntimeError(msg) from exc
    except tomllib.TOMLDecodeError as exc:
        msg = f"Invalid TOML in config file: {path}"
        raise RuntimeError(msg) from exc

    try:
        llm = data["llm"]
        agent = data["agent"]
        prompts = data["prompts"]
        db = data["db"]
        pool = db["pool"]
        guardrails = db["guardrails"]
        timeouts = db["timeouts"]
        retry = db["retry"]

        return RoomsSqlConfig(
            llm=LlmConfig(
                model=str(llm["model"]),
                temperature=float(llm["temperature"]),
                reasoning_effort=str(llm["reasoning_effort"]),
                timeout_s=float(llm["timeout_s"]),
                max_retries=int(llm["max_retries"]),
            ),
            agent=AgentConfig(
                dialect=str(agent["dialect"]),
                top_k=int(agent["top_k"]),
            ),
            prompts=PromptsConfig(
                folder=str(prompts["folder"]),
                system_prompt=str(prompts["system_prompt"]),
            ),
            db=DbConfig(
                pool=DbPoolConfig(
                    min_size=int(pool["min_size"]),
                    max_size=int(pool["max_size"]),
                    timeout_s=float(pool["timeout_s"]),
                ),
                guardrails=DbGuardrailsConfig(
                    max_rows=int(guardrails["max_rows"]),
                    allow_only_hotel_tables=bool(guardrails["allow_only_hotel_tables"]),
                ),
                timeouts=DbTimeoutsConfig(
                    statement_timeout_ms=int(timeouts["statement_timeout_ms"]),
                ),
                retry=DbRetryConfig(
                    max_transient_retries=int(retry["max_transient_retries"]),
                    transient_retry_backoff_s=float(retry["transient_retry_backoff_s"]),
                    retry_writes_on_transient_errors=bool(
                        retry["retry_writes_on_transient_errors"],
                    ),
                ),
            ),
        )

    except KeyError as exc:
        msg = f"Missing required config key: {exc}"
        raise RuntimeError(msg) from exc
    except (TypeError, ValueError) as exc:
        msg = "Invalid config value type"
        raise RuntimeError(msg) from exc


# ============================
# Environment and user-facing messages
# ============================


@lru_cache(maxsize=1)
def get_neon_db_url() -> str:
    """Get the Neon connection URL from the environment.

    Returns:
        Neon Postgres connection URL.

    Raises:
        RuntimeError: If NEON_DB_URL is not set.

    """
    url = os.getenv("NEON_DB_URL")
    if not url:
        msg = "NEON_DB_URL is not set"
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


def resolve_prompts_dir(*, prompts_folder: str) -> Path:
    """Resolve the directory containing prompt templates.

    Args:
        prompts_folder: Folder relative to this module containing prompts.

    Returns:
        Resolved path to the prompts directory.

    Raises:
        RuntimeError: If the directory does not exist.

    """
    prompts_dir = (Path(__file__).parent / prompts_folder).resolve()
    if not prompts_dir.exists() or not prompts_dir.is_dir():
        msg = f"Prompts folder not found: {prompts_dir}"
        raise RuntimeError(msg)
    return prompts_dir


def resolve_prompt_path(*, prompts_dir: Path, filename: str) -> Path:
    """Resolve a prompt template file within the prompts directory.

    Args:
        prompts_dir: Directory containing prompt templates.
        filename: Prompt template filename.

    Returns:
        Resolved path to the prompt template.

    Raises:
        RuntimeError: If the template file does not exist.

    """
    candidate = (prompts_dir / filename).resolve()
    if not candidate.exists() or not candidate.is_file():
        msg = f"System prompt template not found: {candidate}"
        raise RuntimeError(msg)
    return candidate


@lru_cache(maxsize=5)
def load_prompt_template(path: Path) -> StringTemplate:
    """Load and cache a prompt template.

    Args:
        path: Path to the template file.

    Returns:
        Parsed Template object.

    Raises:
        RuntimeError: If the file cannot be read.

    """
    try:
        return StringTemplate(path.read_text(encoding="utf-8"))
    except OSError as exc:
        msg = f"Failed to read prompt template at: {path}"
        raise RuntimeError(msg) from exc


def render_system_prompt(  # noqa: PLR0913
    *,
    template_path: Path,
    top_k: int,
    dialect: str,
    enum_values: dict[str, list[str]],
    basic_amenities: list[str],
    additional_amenities: list[str],
    view_types: list[str],
) -> str:
    """Render the system prompt template with runtime substitutions.

    Args:
        template_path: Path to the system prompt template file.
        top_k: Prompt parameter used by the template.
        dialect: SQL dialect label used by the template.
        enum_values: Enum type name -> allowed values.
        basic_amenities: Distinct basic amenity values.
        additional_amenities: Distinct additional amenity values.
        view_types: Distinct view type values.

    Returns:
        Rendered system prompt string.

    """
    template = load_prompt_template(template_path)
    return template.safe_substitute(
        top_k=top_k,
        dialect=dialect,
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
    neon_db_url: str,
) -> tuple[dict[str, list[str]], list[str], list[str], list[str]]:
    """Fetch database metadata used to fill the system prompt.

    The system prompt template is populated with:
      - Enum values for known enum types.
      - Distinct values from array columns in the rooms table.

    Args:
        neon_db_url: Database URL.

    Returns:
        Tuple of (enum_values, basic_amenities, additional_amenities, view_types).

    Raises:
        OperationalError: If metadata queries fail.

    """
    enum_values: dict[str, list[str]] = {}

    try:
        async with (
            await psycopg.AsyncConnection.connect(neon_db_url) as conn,
            conn.cursor() as cur,
        ):
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
# Tool: async SQL execution
# ============================


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


async def _sleep_backoff(base_s: float, attempt: int) -> None:
    """Sleep using exponential backoff.

    Args:
        base_s: Base backoff in seconds.
        attempt: Zero-based attempt index.

    """
    await asyncio.sleep(base_s * (2**attempt))


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


def _tool_result(*, rows: list[dict[str, Any]], truncated: bool) -> dict[str, Any]:
    """Build the tool response payload.

    Args:
        rows: Rows to return.
        truncated: Whether rows were truncated.

    Returns:
        Tool response dictionary.

    """
    return {"truncated": truncated, "rows": rows}


def _tool_error_result(message: str) -> dict[str, Any]:
    """Build a safe error response for the SQL tool.

    Args:
        message: User-facing error message.

    Returns:
        Tool response dict including 'error' and empty 'rows'.

    """
    return {"truncated": False, "rows": [], "error": message}


def build_run_sql_tool(
    *,
    pool: AsyncConnectionPool,
    config: RoomsSqlConfig,
) -> BaseTool:
    """Build the async `run_sql` tool bound to a pool and configuration.

    Args:
        pool: Async database connection pool.
        config: Parsed configuration.

    Returns:
        Async tool callable named "run_sql".

    """

    @tool(parse_docstring=True)
    async def run_sql(query: str) -> dict[str, Any]:
        """Execute a single SQL statement and return rows.

        Args:
            query: One SQL statement (no semicolons). May be SELECT or DML. If DML
                uses RETURNING, returned rows are provided.

        Returns:
            Dict with keys:
              - rows: list[dict[str, Any]]
              - truncated: bool
              - error: str (only present on failure)

        """
        try:
            validate_sql(
                query,
                allow_only_hotel_tables=config.db.guardrails.allow_only_hotel_tables,
            )
        except ValueError as exc:
            msg = str(exc)
            logger.info("run_sql rejected by guardrails: %s", msg)
            return _tool_error_result(msg)

        is_write = _is_write_sql(query)
        attempts = config.db.retry.max_transient_retries + 1

        for attempt in range(attempts):
            try:
                async with (
                    pool.connection(timeout=config.db.pool.timeout_s) as conn,
                    conn.cursor(row_factory=dict_row) as cur,
                ):
                    # Connections in the pool are configured for autocommit so that
                    # single-statement booking/cancel updates commit reliably.
                    #
                    # NOTE: psycopg's type stubs expect a LiteralString for `execute()`.
                    # This cast is only to satisfy static type checkers. Runtime safety
                    # is provided by `validate_sql(...)` above.
                    await cur.execute(cast("LiteralString", query))

                    if cur.description is None:
                        return _tool_result(rows=[], truncated=False)

                    rows_raw = await cur.fetchall()
                    rows = [dict(r) for r in rows_raw]
                    rows, truncated = _truncate_rows(
                        rows,
                        max_rows=config.db.guardrails.max_rows,
                    )
                    return _tool_result(rows=rows, truncated=truncated)

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
                        or config.db.retry.retry_writes_on_transient_errors
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
                        config.db.retry.transient_retry_backoff_s,
                        attempt,
                    )
                    continue

                logger.warning(
                    "run_sql DB/pool operation failed (attempt=%s/%s)",
                    attempt + 1,
                    attempts,
                    exc_info=True,
                )
                return _tool_error_result(_user_facing_db_message())

            except Exception:
                logger.exception("run_sql unexpected failure")
                return _tool_error_result(_user_facing_db_message())

        logger.warning("run_sql failed after retries")
        return _tool_error_result(_user_facing_db_message())

    return run_sql


# ============================
# Factory: initialize once, serve fast
# ============================


@dataclass
class RoomsAgentFactory:
    """Create and manage the DB pool and compiled rooms agent.

    Attributes:
        config: Parsed configuration loaded from TOML.
        neon_db_url: Database URL used to create connections.
        pool: Async connection pool used by the SQL tool.
        agent: Compiled LangGraph agent created after initialization.

    """

    config: RoomsSqlConfig
    neon_db_url: str
    pool: AsyncConnectionPool | None = None
    agent: CompiledStateGraph | None = None

    _init_lock: asyncio.Lock | None = None

    async def ensure_initialized(self) -> None:
        """Ensure the pool and agent are initialized.

        This method is safe to call concurrently; only one initializer will run and
        subsequent callers will wait for completion.

        Raises:
            OperationalError: If initialization fails for an expected operational
                reason.

        """
        if self.agent is not None and self.pool is not None:
            return

        if self._init_lock is None:
            self._init_lock = asyncio.Lock()

        async with self._init_lock:
            if self.agent is not None and self.pool is not None:
                return
            try:
                await self._initialize()
            except OperationalError:
                raise
            except Exception as exc:
                msg = "Rooms agent failed to initialize"
                raise OperationalError(msg) from exc

    async def _initialize(self) -> None:
        """Initialize the pool, render the system prompt, and build the agent.

        Creates the database pool, fetches metadata used to fill the system prompt,
        constructs the LLM client, and builds the compiled agent.

        Notes:
            The pool config sets connections to autocommit mode so that booking/cancel
            writes (single SQL statements) commit reliably when the connection is
            returned to the pool.

        Raises:
            OperationalError: If database metadata cannot be fetched.
            RuntimeError: If required internal state cannot be established.

        """
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

        self.pool = AsyncConnectionPool(
            conninfo=self.neon_db_url,
            min_size=self.config.db.pool.min_size,
            max_size=self.config.db.pool.max_size,
            timeout=self.config.db.pool.timeout_s,
            configure=configure_connection,
            open=False,
        )
        await self.pool.open()

        (
            enum_values,
            basic_amenities,
            additional_amenities,
            view_types,
        ) = await fetch_rooms_metadata(self.neon_db_url)

        prompts_dir = resolve_prompts_dir(prompts_folder=self.config.prompts.folder)
        prompt_path = resolve_prompt_path(
            prompts_dir=prompts_dir,
            filename=self.config.prompts.system_prompt,
        )

        system_prompt = render_system_prompt(
            template_path=prompt_path,
            top_k=self.config.agent.top_k,
            dialect=self.config.agent.dialect,
            enum_values=enum_values,
            basic_amenities=basic_amenities,
            additional_amenities=additional_amenities,
            view_types=view_types,
        )

        llm = ChatOpenAI(
            model=self.config.llm.model,
            temperature=self.config.llm.temperature,
            reasoning={"effort": self.config.llm.reasoning_effort},
            timeout=self.config.llm.timeout_s,
            max_retries=self.config.llm.max_retries,
        )

        if self.pool is None:
            msg = "DB pool not initialized"
            raise RuntimeError(msg)

        run_sql_tool = build_run_sql_tool(pool=self.pool, config=self.config)

        self.agent = create_agent(
            model=llm,
            tools=[run_sql_tool],
            system_prompt=system_prompt,
        )

    def get_agent(self) -> CompiledStateGraph:
        """Return the compiled rooms agent.

        Returns:
            The compiled LangGraph agent.

        Raises:
            RuntimeError: If the factory has not been initialized.

        """
        if self.agent is None:
            msg = (
                "RoomsAgentFactory not initialized. Call await ensure_initialized() "
                "(or initialize_rooms_agent()) first."
            )
            raise RuntimeError(msg)
        return self.agent

    async def aclose(self) -> None:
        """Close the connection pool if it exists.

        This method is idempotent.

        """
        if self.pool is not None:
            await self.pool.close()


# ============================
# Public helpers for FastAPI
# ============================


_FACTORY: dict[str, RoomsAgentFactory] = {}


def get_factory(*, config: RoomsSqlConfig, neon_db_url: str) -> RoomsAgentFactory:
    """Return the singleton RoomsAgentFactory.

    Args:
        config: Parsed configuration loaded from TOML.
        neon_db_url: Database URL.

    Returns:
        A singleton RoomsAgentFactory instance.

    Raises:
        OperationalError: If the factory cannot be constructed due to an expected
            operational failure.

    """
    existing = _FACTORY.get("factory")
    if existing is not None:
        return existing

    try:
        created = RoomsAgentFactory(config=config, neon_db_url=neon_db_url)
    except OperationalError:
        raise
    except Exception as exc:
        msg = "Rooms agent is not configured correctly"
        raise OperationalError(msg) from exc

    _FACTORY["factory"] = created
    return created


async def initialize_rooms_agent(*, config: RoomsSqlConfig, neon_db_url: str) -> None:
    """Initialize the rooms agent at application startup.

    Args:
        config: Parsed configuration loaded from TOML.
        neon_db_url: Database URL.

    Raises:
        OperationalError: If initialization fails.

    """
    try:
        await get_factory(config=config, neon_db_url=neon_db_url).ensure_initialized()
    except OperationalError:
        raise
    except Exception as exc:
        msg = "Rooms agent failed during startup"
        raise OperationalError(msg) from exc


async def shutdown_rooms_agent() -> None:
    """Close DB resources for the rooms agent at application shutdown.

    This function is safe to call even if initialization did not complete.

    """
    factory = _FACTORY.get("factory")
    if factory is None:
        return

    try:
        await factory.aclose()
    except Exception:  # noqa: BLE001
        logger.debug("shutdown_rooms_agent failed to close DB resources", exc_info=True)
    finally:
        _FACTORY.clear()


async def get_rooms_agent(
    *,
    config: RoomsSqlConfig,
    neon_db_url: str,
) -> CompiledStateGraph:
    """Get an initialized rooms agent.

    Args:
        config: Parsed configuration loaded from TOML.
        neon_db_url: Database URL.

    Returns:
        A compiled LangGraph agent that is ready to handle requests.

    Raises:
        OperationalError: If the agent cannot be created or initialized.

    """
    try:
        factory = get_factory(config=config, neon_db_url=neon_db_url)
        await factory.ensure_initialized()
        agent = factory.agent
        if agent is None:
            msg = "Rooms agent failed to initialize"
            raise OperationalError(msg)  # noqa: TRY301
    except OperationalError:
        raise
    except Exception as exc:
        msg = "Rooms agent is temporarily unavailable"
        raise OperationalError(msg) from exc
    else:
        return agent
