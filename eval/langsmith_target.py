"""LangSmith evaluation target for multi-turn Blue Horizon orchestration runs.

This module provides an async target function compatible with LangSmith
evaluate/aevaluate APIs. It runs a full multi-turn case through the shared
OrchestrationManager, lazily provisioning an isolated Postgres schema only when
the router sends a turn to the "rooms" path.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel, ConfigDict, ValidationError

from blue_horizon.agents.orchestration import OrchestrationManager
from blue_horizon.agents.rooms_sql import reset_eval_schema, set_eval_schema
from eval.db_reset import create_case_schema, drop_case_schema
from eval.schema_slots import acquire_schema_slot, release_schema_slot
from load_data.rooms_pgsql import DATA_PATH, get_pgsql_conn_string

if TYPE_CHECKING:
    from contextvars import Token


class RunSqlOutput(BaseModel):
    """Typed payload returned by the rooms ``run_sql`` tool.

    This model captures the subset of fields the evaluation harness cares about
    when summarizing tool activity. The raw tool output may include additional
    keys (e.g., rows), but we avoid storing those in summaries to keep artifacts
    compact and stable across runs.

    Attributes:
        status: Tool status string (e.g., "ok" or "error").
        rowcount: Number of rows returned or affected by the statement.
        truncated: Whether the tool output was truncated by the agent guardrails.
        error: User-facing error message when the tool fails.

    """

    status: str | None = None
    rowcount: int | None = None
    truncated: bool | None = None
    error: str | None = None


class HydratedItemOutput(BaseModel):
    """Typed payload for ``hydrate_items`` outputs used in eval logging.

    The info agent returns hydrated items with rich metadata. The evaluation
    harness only persists the ``source`` and ``text`` fields to summarize
    which contexts were used without logging large metadata blobs.

    Attributes:
        source: Origin of the hydrated item (faq/amenities/services).
        text: Human-readable content used by the agent.

    """

    model_config = ConfigDict(from_attributes=True)

    source: str | None = None
    text: str | None = None

logger = logging.getLogger(__name__)

_ORCHESTRATION_LOCK = asyncio.Lock()
_ORCHESTRATION: OrchestrationManager | None = None
_RESET_POOL_LOCK = asyncio.Lock()
_RESET_POOL: AsyncConnectionPool[Any] | None = None

_MAX_SCHEMA_NAME_LEN = 63
_ROUTE_KEY = "route"


def _sanitize_identifier(raw: str) -> str:
    """Sanitize a string to a safe Postgres identifier fragment.

    Args:
        raw: Raw identifier string.

    Returns:
        Sanitized identifier containing only letters, numbers, and underscores.

    """
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", raw.strip())
    cleaned = cleaned.strip("_")
    return cleaned or "case"


def _get_rooms_data_path() -> Path:
    """Resolve the rooms baseline data path.

    Returns:
        Path to the pickled rooms datasets.

    """
    override = os.getenv("EVAL_ROOMS_DATA_PATH") or os.getenv("ROOMS_DATA_PATH")
    if override:
        return Path(override)
    return DATA_PATH


def _get_schema_slot_config() -> tuple[int, int, float, float]:
    """Load configuration for distributed schema slot limiting.

    Returns:
        Tuple of (max_slots, stale_after_s, wait_timeout_s, poll_interval_s).

    """
    max_slots = int(os.getenv("EVAL_MAX_ACTIVE_SCHEMAS", "0"))
    stale_after_s = int(os.getenv("EVAL_SCHEMA_SLOT_STALE_AFTER_S", "1800"))
    wait_timeout_s = float(os.getenv("EVAL_SCHEMA_SLOT_WAIT_TIMEOUT_S", "300"))
    poll_interval_s = float(os.getenv("EVAL_SCHEMA_SLOT_POLL_INTERVAL_S", "1"))

    max_slots = max(0, max_slots)
    stale_after_s = max(0, stale_after_s)
    wait_timeout_s = max(0.1, wait_timeout_s)
    poll_interval_s = max(0.1, poll_interval_s)

    return max_slots, stale_after_s, wait_timeout_s, poll_interval_s


def _extract_assistant_text(messages: list[BaseMessage]) -> str:
    """Extract the last assistant message content from a list of messages.

    Args:
        messages: Ordered list of LangChain messages.

    Returns:
        The last assistant message content, or an empty string if none found.

    """
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return str(message.content)
    return ""


async def ensure_orchestration_ready() -> OrchestrationManager:
    """Ensure the shared orchestration manager is initialized and ready.

    Returns:
        The shared OrchestrationManager instance.

    Raises:
        RuntimeError: If the orchestration manager does not become ready in time.

    """
    global _ORCHESTRATION  # noqa: PLW0603
    if _ORCHESTRATION is not None and _ORCHESTRATION.is_ready:
        return _ORCHESTRATION

    async with _ORCHESTRATION_LOCK:
        if _ORCHESTRATION is None:
            _ORCHESTRATION = OrchestrationManager()
        if _ORCHESTRATION.is_ready:
            return _ORCHESTRATION
        await _ORCHESTRATION.start()

        timeout_s = float(os.getenv("EVAL_ORCHESTRATION_READY_TIMEOUT_S", "60"))
        start = asyncio.get_running_loop().time()
        while not _ORCHESTRATION.is_ready:
            if asyncio.get_running_loop().time() - start >= timeout_s:
                msg = "OrchestrationManager did not become ready within timeout."
                raise RuntimeError(msg)
            await asyncio.sleep(0.2)

    if _ORCHESTRATION is None:
        msg = "OrchestrationManager was not initialized."
        raise RuntimeError(msg)
    return _ORCHESTRATION


async def ensure_reset_pool() -> AsyncConnectionPool[Any]:
    """Lazily initialize and return the async reset connection pool.

    Returns:
        An open AsyncConnectionPool for schema reset operations.

    """
    global _RESET_POOL  # noqa: PLW0603
    if _RESET_POOL is not None:
        return _RESET_POOL

    async with _RESET_POOL_LOCK:
        if _RESET_POOL is not None:
            return _RESET_POOL

        max_size = int(os.getenv("EVAL_RESET_POOL_MAX_SIZE", "4"))
        min_size = int(os.getenv("EVAL_RESET_POOL_MIN_SIZE", "1"))
        max_size = max(1, max_size)
        min_size = max(1, min(min_size, max_size))

        pool = AsyncConnectionPool(
            conninfo=get_pgsql_conn_string(),
            min_size=min_size,
            max_size=max_size,
            timeout=30,
            open=False,
        )
        await pool.open()
        _RESET_POOL = pool
        return pool


class SchemaManager:
    """Track and manage the lifecycle of a case schema for evaluation runs.

    Attributes:
        case_id: Case identifier used to name the schema.
        run_id: Unique run identifier used in schema naming.
        data_path: Path to baseline rooms data for schema initialization.
        schema: Schema name once created, otherwise ``None``.
        schema_created: Whether the schema has been created for this case.
        schema_token: ContextVar token for resetting the eval schema.
        slot_id: Acquired schema slot identifier, if slot limiting is enabled.

    """

    case_id: str
    run_id: str
    data_path: Path
    schema: str | None
    schema_created: bool
    schema_token: Token[str | None] | None
    slot_id: int | None
    _lock: asyncio.Lock

    def __init__(self, *, case_id: str, run_id: str, data_path: Path) -> None:
        """Initialize the schema manager.

        Args:
            case_id: Case identifier used to name the schema.
            run_id: Unique run identifier used in schema naming.
            data_path: Path to baseline rooms data for schema initialization.

        """
        self.case_id = case_id
        self.run_id = run_id
        self.data_path = data_path
        self.schema = None
        self.schema_created = False
        self.schema_token = None
        self.slot_id = None
        self._lock = asyncio.Lock()

    async def ensure_schema(self) -> None:
        """Create the case schema and set the eval ContextVar if needed."""
        if self.schema_created:
            return
        async with self._lock:
            if self.schema_created:
                return

            schema = self._build_schema_name()
            pool = await ensure_reset_pool()
            max_slots, stale_after_s, wait_timeout_s, poll_interval_s = (
                _get_schema_slot_config()
            )
            if max_slots > 0:
                self.slot_id = await acquire_schema_slot(
                    pool=pool,
                    run_id=self.run_id,
                    case_id=self.case_id,
                    max_slots=max_slots,
                    stale_after_s=stale_after_s,
                    wait_timeout_s=wait_timeout_s,
                    poll_interval_s=poll_interval_s,
                )

            try:
                await create_case_schema(
                    pool=pool,
                    schema=schema,
                    data_path=self.data_path,
                )
            except Exception:
                if self.slot_id is not None:
                    await self._release_slot(pool=pool)
                raise

            self.schema = schema
            self.schema_token = set_eval_schema(schema)
            self.schema_created = True

    def reset_context(self) -> None:
        """Reset the evaluation ContextVar if it was set."""
        if self.schema_token is None:
            return
        reset_eval_schema(self.schema_token)
        self.schema_token = None

    async def drop_schema(self) -> None:
        """Drop the case schema if it was created."""
        if not self.schema_created or self.schema is None:
            return
        pool = await ensure_reset_pool()
        try:
            await drop_case_schema(pool=pool, schema=self.schema)
        finally:
            if self.slot_id is not None:
                await self._release_slot(pool=pool)

    async def release_slot(self) -> None:
        """Release the schema slot if one is held."""
        if self.slot_id is None:
            return
        pool = await ensure_reset_pool()
        await self._release_slot(pool=pool)

    async def _release_slot(self, *, pool: AsyncConnectionPool[Any]) -> None:
        """Release the schema slot using the provided pool."""
        if self.slot_id is None:
            return
        await release_schema_slot(pool=pool, slot_id=self.slot_id)
        self.slot_id = None

    def _build_schema_name(self) -> str:
        """Construct a safe, unique schema name for the case.

        Returns:
            Schema name string limited to Postgres identifier length.

        """
        safe_case = _sanitize_identifier(self.case_id)
        prefix = f"eval_{self.run_id}_"
        max_case_len = max(0, _MAX_SCHEMA_NAME_LEN - len(prefix))
        case_part = safe_case[:max_case_len] if max_case_len else ""
        if not case_part:
            case_part = "case"
        return f"{prefix}{case_part}"


class EvalCaptureCallback(AsyncCallbackHandler):
    """Capture routing decisions and tool artifacts for a single turn.

    Attributes:
        schema_manager: Schema manager used to provision the rooms schema on demand.
        route_pred: Router decision captured for the turn.
        tool_summary: Compact summaries of tools executed in this turn.
        contexts_used: Truncated context snippets captured from hydrate_items output.

    """

    schema_manager: SchemaManager
    route_pred: str | None
    tool_summary: list[dict[str, Any]]
    contexts_used: list[str]

    def __init__(self, *, schema_manager: SchemaManager) -> None:
        """Initialize the callback handler.

        Args:
            schema_manager: Schema manager responsible for lazy schema creation.

        """
        super().__init__()
        self.schema_manager = schema_manager
        self.route_pred = None
        self.tool_summary = []
        self.contexts_used = []

    async def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Capture router outputs and provision schemas before rooms tools run.

        Args:
            outputs: Chain outputs from LangChain/LangGraph.
            run_id: LangChain run ID for the finished chain.
            parent_run_id: Optional parent run ID.
            tags: Optional tags emitted by the chain.
            **kwargs: Additional keyword arguments.

        """
        _ = run_id, parent_run_id, tags, kwargs
        if isinstance(outputs, dict) and _ROUTE_KEY in outputs:
            route_val = outputs.get(_ROUTE_KEY)
            if isinstance(route_val, str):
                self.route_pred = route_val
                if route_val == "rooms" and not self.schema_manager.schema_created:
                    await self.schema_manager.ensure_schema()

    async def on_tool_end(
        self,
        output: Any,  # noqa: ANN401
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Collect compact tool summaries and hydration contexts.

        Args:
            output: Tool output payload.
            run_id: LangChain run ID for the tool.
            parent_run_id: Optional parent run ID.
            tags: Optional tags associated with the tool.
            **kwargs: Additional keyword arguments.

        """
        _ = run_id, parent_run_id, tags, kwargs
        tool_name = _get_tool_name(kwargs)
        if tool_name == "run_sql":
            self._capture_run_sql(output)
        elif tool_name == "hydrate_items":
            self._capture_hydrate_items(output)

    def _capture_run_sql(self, output: RunSqlOutput | Mapping[str, object]) -> None:
        """Capture a compact run_sql summary without row payloads.

        Args:
            output: run_sql tool output.

        """
        if isinstance(output, RunSqlOutput):
            payload = output
        elif isinstance(output, Mapping):
            try:
                payload = RunSqlOutput.model_validate(dict(output))
            except ValidationError:
                return
        else:
            return
        summary = {
            "tool": "run_sql",
            "status": payload.status,
            "rowcount": payload.rowcount,
            "truncated": payload.truncated,
        }
        if payload.error:
            summary["error"] = payload.error
        self.tool_summary.append(summary)

    def _capture_hydrate_items(self, output: list[object]) -> None:
        """Capture hydration summary and contexts used.

        Args:
            output: hydrate_items tool output.

        """
        if not isinstance(output, list):
            return
        summary = {
            "tool": "hydrate_items",
            "count": len(output),
        }
        parsed_items: list[HydratedItemOutput] = []
        for item in output:
            if isinstance(item, HydratedItemOutput):
                parsed_items.append(item)
                continue
            try:
                parsed_items.append(
                    HydratedItemOutput.model_validate(item, from_attributes=True),
                )
            except ValidationError:
                continue

        sources = [
            str(item.source)
            for item in parsed_items
            if item.source is not None
        ]
        if sources:
            summary["sources"] = sources
        self.tool_summary.append(summary)

        for item in parsed_items:
            if item.text is None:
                continue
            self.contexts_used.append(str(item.text))


def _get_tool_name(metadata: dict[str, Any]) -> str | None:
    """Extract the tool name from callback metadata.

    Args:
        metadata: Callback metadata dict.

    Returns:
        Tool name string if available.

    """
    name = metadata.get("name") or metadata.get("tool_name")
    if isinstance(name, str):
        return name
    serialized = metadata.get("serialized")
    if isinstance(serialized, dict):
        serialized_name = serialized.get("name")
        if isinstance(serialized_name, str):
            return serialized_name
    return None


async def run_example(example: dict[str, Any]) -> dict[str, Any]:
    """Run a single LangSmith example through the orchestration pipeline.

    Args:
        example: LangSmith dataset example input containing case_id, turns, and tags.

    Returns:
        Dict containing turn outputs, final schema, and case tags.

    Raises:
        KeyError: If required keys are missing from the example.

    """
    case_id = str(example["case_id"])
    turns = example["turns"]
    case_tags_raw = example.get("tags") or []
    case_tags = list(case_tags_raw)

    orchestration_mgr = await ensure_orchestration_ready()

    thread_id = uuid4().hex
    run_id = uuid4().hex
    schema_manager = SchemaManager(
        case_id=case_id,
        run_id=run_id,
        data_path=_get_rooms_data_path(),
    )

    turn_outputs: list[dict[str, Any]] = []

    try:
        for turn_index, turn in enumerate(turns):
            user_text = str(turn["user"])
            callback = EvalCaptureCallback(schema_manager=schema_manager)
            tags = ["eval", f"case:{case_id}", f"turn:{turn_index}", *case_tags]
            metadata = {
                "case_id": case_id,
                "turn_index": turn_index,
                "db_schema": schema_manager.schema,
            }

            result = await orchestration_mgr.ainvoke(
                thread_id=thread_id,
                user_text=user_text,
                callbacks=[callback],
                tags=tags,
                metadata=metadata,
            )

            messages_raw = result.get("messages", [])
            messages = messages_raw if isinstance(messages_raw, list) else []
            assistant_text = _extract_assistant_text(messages)

            route_pred = callback.route_pred
            if route_pred is None:
                fallback_route = result.get(_ROUTE_KEY)
                if isinstance(fallback_route, str):
                    route_pred = fallback_route

            turn_outputs.append(
                {
                    "assistant_text": assistant_text,
                    "route_pred": route_pred,
                    "tool_summary": callback.tool_summary,
                    "contexts_used": callback.contexts_used,
                },
            )

    finally:
        if schema_manager.schema_created:
            try:
                schema_manager.reset_context()
            except Exception:
                logger.exception("Failed to reset eval schema ContextVar")

            try:
                await schema_manager.drop_schema()
            except Exception:
                logger.exception("Failed to drop eval schema %s", schema_manager.schema)

    return {
        "turn_outputs": turn_outputs,
        "final_db_schema": schema_manager.schema,
        "case_tags": case_tags,
    }
