"""LangSmith evaluation target for multi-turn Blue Horizon orchestration runs.

This module provides an async target function compatible with LangSmith
evaluate/aevaluate APIs.  It runs a full multi-turn case through the shared
OrchestrationManager, capturing per-turn routing decisions, tool summaries,
and RAG context snippets for downstream evaluators.
"""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import re
from collections.abc import Mapping
from typing import Any
from uuid import UUID, uuid4

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from pydantic import BaseModel, ConfigDict, ValidationError

from blue_horizon.agents.information import (
    EvalInfoToolLog,
    get_eval_info_tool_log,
    reset_eval_info_tool_log,
    set_eval_info_tool_log,
)
from blue_horizon.agents.orchestration import OrchestrationManager
from eval._utils import (
    coerce_float as _coerce_float,
)
from eval._utils import (
    coerce_int as _coerce_int,
)
from eval._utils import (
    coerce_strict_bool as _coerce_strict_bool,
)
from eval.config import load_eval_config


class RunSqlOutput(BaseModel):
    """Typed payload returned by the rooms ``run_sql`` tool.

    This model captures the subset of fields the evaluation harness cares about
    when summarizing tool activity. The raw tool output may include additional
    keys (e.g., rows), but we avoid storing those in summaries to keep artifacts
    compact and stable across runs.

    Attributes:
        status: Tool status string (e.g., "ok" or "error").
        rowcount: Number of rows returned or affected by the statement.
        rows: Result rows returned by the tool, when present.
        rows: Result rows returned by the tool, when present.
        truncated: Whether the tool output was truncated by the agent guardrails.
        error: User-facing error message when the tool fails.

    """

    status: str | None = None
    rowcount: int | None = None
    truncated: bool | None = None
    error: str | None = None
    rows: list[dict[str, Any]] | None = None


class HydratedItemOutput(BaseModel):
    """Typed payload for ``hydrate_items`` outputs used in eval logging.

    The info agent returns hydrated items with rich metadata. The evaluation
    harness captures the source, text, and metadata fields for use by
    evaluators and the judge LLM.

    Attributes:
        source: Origin of the hydrated item (faq/amenities/services).
        text: Human-readable content used by the agent.
        metadata: Metadata dict containing price, duration, booking info, etc.

    """

    model_config = ConfigDict(from_attributes=True)

    source: str | None = None
    text: str | None = None
    metadata: dict[str, Any] | None = None

logger = logging.getLogger(__name__)

_ORCHESTRATION_LOCK = asyncio.Lock()
_ORCHESTRATION: OrchestrationManager | None = None

_ROUTE_KEY = "route"  # key from orchestration.py


def _canonicalize_filter_key(key: str) -> str:
    """Normalize filter keys for strict comparisons.

    Args:
        key: Raw filter key.

    Returns:
        Canonicalized key string.

    """
    normalized = key.strip().lower().replace(" ", "_").replace("-", "_")
    normalized = re.sub(r"_+", "_", normalized)
    if normalized == "duration_mintues":
        return "duration_minutes"
    return normalized


def _coerce_strict_filter_value(
    canonical_key: str,
    value: object,
) -> object | None:
    """Coerce filter values based on a strict canonical key.

    Args:
        canonical_key: Canonical filter key.
        value: Raw filter value.

    Returns:
        Coerced value when parseable, otherwise None.

    """
    if canonical_key == "booking_required":
        return _coerce_strict_bool(value)
    if canonical_key.endswith("_price"):
        return _coerce_float(value)
    return _coerce_int(value)


def _swap_range_if_needed(
    norm: dict[str, object],
    min_key: str,
    max_key: str,
) -> None:
    """Swap min/max values if they are reversed.

    Args:
        norm: Normalized filters dict updated in place.
        min_key: Minimum value key.
        max_key: Maximum value key.

    """
    min_val = norm.get(min_key)
    max_val = norm.get(max_key)
    if (
        isinstance(min_val, (int, float))
        and not isinstance(min_val, bool)
        and isinstance(max_val, (int, float))
        and not isinstance(max_val, bool)
        and float(min_val) > float(max_val)
    ):
        norm[min_key], norm[max_key] = norm[max_key], norm[min_key]


def _normalize_info_filters_strict(
    filters: Mapping[str, object] | None,
) -> tuple[dict[str, object] | None, list[str]]:
    """Normalize amenity/service filters using strict canonical keys.

    Args:
        filters: Raw filters dict passed to an info tool, if any.

    Returns:
        Tuple of (normalized filters or None, unknown key list).

    """
    # Evaluators rely on these canonical keys to compare expected filters.
    if not filters:
        return None, []
    if not isinstance(filters, dict):
        return None, ["<non_dict_filters>"]

    canonical_keys = {
        "booking_required",
        "min_price",
        "max_price",
        "min_notice_hours",
        "max_notice_hours",
        "min_duration_minutes",
        "max_duration_minutes",
    }

    norm: dict[str, object] = {}
    unknown_keys: list[str] = []

    for key, value in filters.items():
        canonical = _canonicalize_filter_key(str(key))
        if canonical not in canonical_keys:
            unknown_keys.append(str(key))
            continue

        coerced = _coerce_strict_filter_value(canonical, value)
        if coerced is None:
            unknown_keys.append(f"{key}:<bad_value>")
            continue
        norm[canonical] = coerced

    _swap_range_if_needed(norm, "min_price", "max_price")
    _swap_range_if_needed(norm, "min_notice_hours", "max_notice_hours")
    _swap_range_if_needed(norm, "min_duration_minutes", "max_duration_minutes")

    return (norm or None), unknown_keys


def _extract_filters_from_tool_inputs(inputs: object) -> dict[str, object] | None:
    """Extract a filters dict from tool inputs payloads.

    Args:
        inputs: Tool inputs payload, often a mapping.

    Returns:
        Filters dict if present, otherwise None.

    """
    if not isinstance(inputs, Mapping):
        return None
    raw_filters = inputs.get("filters")
    if isinstance(raw_filters, Mapping):
        return dict(raw_filters)

    excluded_keys = {"query", "k", "top_k"}
    extracted = {
        str(key): value
        for key, value in inputs.items()
        if str(key) not in excluded_keys
    }
    return extracted or None


def _safe_str(obj: object) -> str:
    """Safely convert an object to a string.

    Args:
        obj: Object to convert.

    Returns:
        String representation of the object.

    """
    if isinstance(obj, str):
        return obj
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return repr(obj)


def _redact_secrets(text: str) -> str:
    """Redact obvious secret-like substrings from a text blob.

    Args:
        text: Input text to sanitize.

    Returns:
        Redacted text safe for logging.

    """
    redacted = re.sub(r"sk-[A-Za-z0-9]{10,}", "[REDACTED]", text)
    redacted = re.sub(r"Bearer\\s+\\S+", "Bearer [REDACTED]", redacted)
    redacted = re.sub(
        r"(\\w+://)([^:@\\s]+):([^@\\s]+)@",
        r"\\1[REDACTED]:[REDACTED]@",
        redacted,
    )
    secret_keys = (
        "OPENAI_API_KEY",
        "GOOGLE_API_KEY",
        "LANGSMITH_API_KEY",
    )
    for key in secret_keys:
        if key in redacted:
            redacted = redacted.replace(key, "[REDACTED]")
    return redacted


def _preview(obj: object, max_len: int = 200) -> str:
    """Create a short, redacted preview of a value.

    Args:
        obj: Value to preview.
        max_len: Maximum preview length.

    Returns:
        Sanitized preview string.

    """
    if max_len <= 0:
        return ""
    text = _safe_str(obj)
    text = _redact_secrets(text)
    text = " ".join(text.split())
    if len(text) <= max_len:
        return text
    return f"{text[: max_len - 1]}…"


def _input_keys(obj: object, max_keys: int = 20) -> list[str]:
    """Extract a capped list of input keys from a mapping.

    Args:
        obj: Tool input payload.
        max_keys: Maximum number of keys to return.

    Returns:
        Sorted list of keys when available, otherwise an empty list.

    """
    if not isinstance(obj, Mapping):
        return []
    keys = sorted(str(k) for k in obj)
    return keys[:max_keys]


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


def _extract_tool_payload(output: object) -> object:
    """Extract the tool payload from LangChain tool outputs.

    Args:
        output: Tool output payload, which may be a ToolMessage.

    Returns:
        The raw payload content for downstream parsing.

    """
    if isinstance(output, ToolMessage):
        if output.artifact is not None:
            return output.artifact
        return output.content
    return output


def _parse_tool_content(content: object) -> object:
    """Parse tool message content into Python objects when possible.

    Args:
        content: Tool content payload, often a JSON string.

    Returns:
        Parsed object when JSON is detected, otherwise the original content.

    """
    parsed: object = content
    if isinstance(content, str):
        stripped = content.strip()
        if stripped.startswith(("[", "{")) and stripped.endswith(("]", "}")):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = content
    elif isinstance(content, list):
        for block in content:
            if not isinstance(block, Mapping):
                continue
            if block.get("type") == "json" and "json" in block:
                parsed = block.get("json")
                break
            if block.get("type") == "text" and "text" in block:
                text = block.get("text")
                if isinstance(text, str):
                    parsed = _parse_tool_content(text)
                    break
    return parsed

def _tool_messages_since_last_user(
    messages: list[BaseMessage],
) -> list[ToolMessage]:
    """Collect tool messages that occurred after the last user message.

    Args:
        messages: Ordered list of LangChain messages.

    Returns:
        List of ToolMessage objects for the most recent turn.

    """
    last_user_index = -1
    for idx in range(len(messages) - 1, -1, -1):
        if isinstance(messages[idx], HumanMessage):
            last_user_index = idx
            break
    if last_user_index == -1:
        return []
    return [
        message
        for message in messages[last_user_index + 1 :]
        if isinstance(message, ToolMessage)
    ]


def _extract_hydrate_contexts_from_messages(
    messages: list[BaseMessage],
) -> list[str]:
    """Extract hydrate_items contexts from tool messages.

    Args:
        messages: Ordered list of LangChain messages.

    Returns:
        List of context strings extracted from hydrate_items tool payloads.

    """
    contexts: list[str] = []
    for tool_message in _tool_messages_since_last_user(messages):
        tool_name = getattr(tool_message, "name", None)
        if tool_name != "hydrate_items":
            continue
        payload = _extract_tool_payload(tool_message)
        payload = _parse_tool_content(payload)
        if not isinstance(payload, list):
            continue
        for item in payload:
            try:
                parsed = HydratedItemOutput.model_validate(
                    item,
                    from_attributes=True,
                )
            except ValidationError:
                continue
            if parsed.text is None:
                continue
            contexts.append(str(parsed.text))
    return contexts


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
            _ORCHESTRATION = OrchestrationManager(prune_tool_messages=False)
        if _ORCHESTRATION.is_ready:
            return _ORCHESTRATION
        await _ORCHESTRATION.start()

        timeout_s = load_eval_config().orchestration.ready_timeout_s
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


class EvalCaptureCallback(AsyncCallbackHandler):
    """Capture routing decisions and tool artifacts for a single turn.

    Attributes:
        route_pred: Router decision captured for the turn.
        tool_summary: Compact summaries of tools executed in this turn.
        contexts_used: Truncated context snippets captured from hydrate_items output.

    """

    route_pred: str | None
    tool_summary: list[dict[str, Any]]
    contexts_used: list[str]
    _pending_tool_entries: dict[UUID, dict[str, Any]]

    def __init__(self) -> None:
        """Initialize the callback handler."""
        super().__init__()
        self.route_pred = None
        self.tool_summary = []
        self.contexts_used = []
        self._pending_tool_entries = {}

    async def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Capture router outputs for the current turn.

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

    async def on_tool_start(  # noqa: PLR0913
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Capture filters passed into amenity/service query tools.

        Args:
            serialized: Serialized tool metadata.
            input_str: Raw input string passed to the tool.
            run_id: LangChain run ID for the tool.
            parent_run_id: Optional parent run ID.
            tags: Optional tags associated with the tool.
            metadata: Optional metadata associated with the tool.
            inputs: Parsed tool input payload when available.
            **kwargs: Additional keyword arguments.

        """
        _ = input_str, parent_run_id, tags, metadata, kwargs
        tool_name = None
        if isinstance(serialized, Mapping):
            raw_name = serialized.get("name")
            if isinstance(raw_name, str):
                tool_name = raw_name
        if tool_name is None:
            return

        entry: dict[str, Any] = {
            "tool": tool_name,
            "input_keys": _input_keys(inputs),
            "input_preview": _preview(inputs),
            "status": "started",
        }
        if tool_name in {"query_amenities", "query_services"}:
            raw_filters = _extract_filters_from_tool_inputs(inputs)
            normalized_filters, unknown_keys = _normalize_info_filters_strict(
                raw_filters,
            )
            entry["filters"] = raw_filters
            entry["filters_norm"] = normalized_filters
            if unknown_keys:
                entry["filters_unknown_keys"] = unknown_keys
        if isinstance(inputs, Mapping):
            raw_k = inputs.get("k")
            if raw_k is None:
                raw_k = inputs.get("top_k")
            k_value = _coerce_int(raw_k)
            if k_value is not None:
                entry["k"] = k_value
        self._pending_tool_entries[run_id] = entry

    async def on_tool_end(  # noqa: C901, PLR0912
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
        entry = self._pending_tool_entries.pop(run_id, None)
        tool_name = _get_tool_name(kwargs)
        if tool_name is None and isinstance(entry, dict):
            tool_name = entry.get("tool")

        if tool_name == "run_sql":
            # Extract content from ToolMessage if needed
            actual_output = output
            if isinstance(output, ToolMessage):
                actual_output = output.content
                # Parse string if needed (could be JSON or Python literal)
                if isinstance(actual_output, str):
                    try:
                        actual_output = json.loads(actual_output)
                    except json.JSONDecodeError:
                        # Try Python literal_eval for dict string representations
                        # Preprocess to handle Decimal('...') and datetime.date(...)
                        cleaned = re.sub(
                            r"Decimal\('([^']*)'\)",
                            r'"\1"',
                            actual_output,
                        )
                        cleaned = re.sub(
                            r"datetime\.date\((\d+),\s*(\d+),\s*(\d+)\)",
                            lambda m: (
                                f'"{m.group(1)}-'
                                f'{int(m.group(2)):02d}-'
                                f'{int(m.group(3)):02d}"'
                            ),
                            cleaned,
                        )
                        try:  # noqa: SIM105
                            actual_output = ast.literal_eval(cleaned)
                        except (ValueError, SyntaxError):
                            pass

            # Only capture if we successfully parsed to a dict/mapping
            if isinstance(actual_output, Mapping):
                self._capture_run_sql(actual_output, entry)
            return
        if tool_name == "hydrate_items":
            self._capture_hydrate_items(output, entry)
            return
        if tool_name in {"query_amenities", "query_services", "query_faq"}:
            summary = entry or {"tool": tool_name}
            summary["status"] = "ok"
            if isinstance(output, list):
                summary["output_preview"] = _preview({"num_items": len(output)})
            self.tool_summary.append(summary)
            return
        if tool_name == "reranker":
            summary = entry or {"tool": tool_name}
            summary["status"] = "ok"
            if isinstance(output, list):
                summary["output_preview"] = _preview({"reranked_k": len(output)})
            self.tool_summary.append(summary)
            return
        if entry is not None:
            entry["status"] = "ok"
            self.tool_summary.append(entry)

    async def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Record failures for amenity/service query tools when possible.

        Args:
            error: Error raised by the tool.
            run_id: LangChain run ID for the tool.
            parent_run_id: Optional parent run ID.
            tags: Optional tags associated with the tool.
            **kwargs: Additional keyword arguments.

        """
        _ = error, parent_run_id, tags, kwargs
        entry = self._pending_tool_entries.pop(run_id, None)
        if entry is None:
            return
        entry["status"] = "error"
        entry["error_preview"] = _preview(error)
        self.tool_summary.append(entry)

    def _capture_run_sql(
        self,
        output: RunSqlOutput | Mapping[str, object],
        base_entry: dict[str, Any] | None = None,
    ) -> None:
        """Capture a compact run_sql summary, including a tiny row sample.

        Args:
            output: run_sql tool output.
            base_entry: Optional base entry with input previews.

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
        summary = dict(base_entry or {})
        summary["tool"] = "run_sql"
        summary["status"] = payload.status or summary.get("status") or "ok"
        summary["rowcount"] = payload.rowcount
        summary["truncated"] = payload.truncated
        if isinstance(payload.rows, list) and payload.rows:
            summary["rows"] = self._compact_rows(payload.rows, max_rows=1)
        if payload.error:
            summary["error"] = payload.error
        summary["output_preview"] = _preview(
            {
                "status": payload.status,
                "rowcount": payload.rowcount,
                "truncated": payload.truncated,
            },
        )
        self.tool_summary.append(summary)

        # Add SQL results to contexts_used for judge LLM evaluation
        if isinstance(payload.rows, list) and payload.rows:
            for row in payload.rows:
                if isinstance(row, Mapping):
                    # Format as readable key-value pairs
                    row_str = ", ".join(
                        f"{k}: {v}" for k, v in row.items() if v is not None
                    )
                    if row_str:
                        self.contexts_used.append(f"SQL result: {row_str}")

    @staticmethod
    def _json_safe(value: object) -> object:
        """Convert values to JSON-serializable structures.

        Args:
            value: Value to normalize for JSON serialization.

        Returns:
            A JSON-safe value, recursively normalized.

        """
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Mapping):
            return {str(k): EvalCaptureCallback._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [EvalCaptureCallback._json_safe(item) for item in value]
        return str(value)

    @staticmethod
    def _compact_rows(
        rows: list[dict[str, Any]],
        max_rows: int = 1,
    ) -> list[dict[str, Any]]:
        """Return a JSON-safe, truncated view of tool rows.

        Args:
            rows: Raw rows returned by the tool.
            max_rows: Maximum number of rows to keep.

        Returns:
            A list containing at most ``max_rows`` JSON-safe row dicts.

        """
        if max_rows <= 0:
            return []
        safe_rows: list[dict[str, Any]] = []
        for row in rows[:max_rows]:
            if not isinstance(row, Mapping):
                continue
            safe_rows.append(
                {
                    str(key): EvalCaptureCallback._json_safe(val)
                    for key, val in row.items()
                },
            )
        return safe_rows

    def _capture_hydrate_items(
        self,
        output: list[object],
        base_entry: dict[str, Any] | None = None,
    ) -> None:
        """Capture hydration summary and contexts used.

        Args:
            output: hydrate_items tool output.
            base_entry: Optional base entry with input previews.

        """
        payload = _extract_tool_payload(output)
        payload = _parse_tool_content(payload)
        if not isinstance(payload, list):
            return
        summary = dict(base_entry or {})
        summary["tool"] = "hydrate_items"
        summary["status"] = summary.get("status") or "ok"
        summary["count"] = len(payload)
        parsed_items: list[HydratedItemOutput] = []
        for item in payload:
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
        summary["output_preview"] = _preview({"num_items": len(parsed_items)})
        self.tool_summary.append(summary)

        for item in parsed_items:
            if item.text is None:
                continue
            # Include metadata with the text for evaluator context
            context = str(item.text)
            if hasattr(item, "metadata") and item.metadata:
                # Format metadata as readable key-value pairs
                metadata_str = ", ".join(
                    f"{k}: {v}" for k, v in item.metadata.items() if v is not None
                )
                if metadata_str:
                    context = f"{context}\n[Metadata: {metadata_str}]"
            self.contexts_used.append(context)


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


async def run_example(  # noqa: C901, PLR0912, PLR0915
    example: dict[str, Any],
) -> dict[str, Any]:
    """Run a single LangSmith example through the orchestration pipeline.

    Args:
        example: LangSmith dataset example input containing case_id, turns, and tags.

    Returns:
        Dict containing turn outputs and case tags.

    Raises:
        KeyError: If required keys are missing from the example.

    """
    case_id = str(example["case_id"])
    turns = example["turns"]
    case_tags_raw = example.get("tags") or []
    case_tags = list(case_tags_raw)

    orchestration_mgr = await ensure_orchestration_ready()

    thread_id = uuid4().hex

    turn_outputs: list[dict[str, Any]] = []

    for turn_index, turn in enumerate(turns):
        user_text = str(turn["user"])
        callback = EvalCaptureCallback()
        tags = ["eval", f"case:{case_id}", f"turn:{turn_index}", *case_tags]
        metadata = {
            "case_id": case_id,
            "turn_index": turn_index,
        }

        token = set_eval_info_tool_log(
            EvalInfoToolLog(tool_summary=[], contexts_used=[]),
        )
        try:
            result = await orchestration_mgr.ainvoke(
                thread_id=thread_id,
                user_text=user_text,
                callbacks=[callback],
                tags=tags,
                metadata=metadata,
            )
        finally:
            log = get_eval_info_tool_log()
            reset_eval_info_tool_log(token)

        messages_raw = result.get("messages", [])
        messages = messages_raw if isinstance(messages_raw, list) else []
        assistant_text = _extract_assistant_text(messages)

        route_pred = callback.route_pred
        if route_pred is None:
            fallback_route = result.get(_ROUTE_KEY)
            if isinstance(fallback_route, str):
                route_pred = fallback_route

        tool_summary = callback.tool_summary
        contexts_used = callback.contexts_used
        if not contexts_used:
            contexts_used = _extract_hydrate_contexts_from_messages(messages)
        if log is not None:
            if not tool_summary and log.tool_summary:
                tool_summary = list(log.tool_summary)
                # Normalize filters for query_amenities/query_services from log
                for entry in tool_summary:
                    if (
                        isinstance(entry, dict)
                        and entry.get("tool")
                        in {"query_amenities", "query_services"}
                        and "filters" in entry
                        and "filters_norm" not in entry
                    ):
                        normalized, unknown = _normalize_info_filters_strict(
                            entry.get("filters"),
                        )
                        entry["filters_norm"] = normalized
                        if unknown:
                            entry["filters_unknown_keys"] = unknown
            if not contexts_used and log.contexts_used:
                contexts_used = list(log.contexts_used)
            if log.tool_summary and tool_summary:
                logged_names = {
                    entry.get("tool")
                    for entry in tool_summary
                    if isinstance(entry, dict)
                }
                for entry in log.tool_summary:
                    if (
                        isinstance(entry, dict)
                        and entry.get("tool") not in logged_names
                    ):
                        if (
                            entry.get("tool")
                            in {"query_amenities", "query_services"}
                            and "filters" in entry
                            and "filters_norm" not in entry
                        ):
                            normalized, unknown = _normalize_info_filters_strict(
                                entry.get("filters"),
                            )
                            entry["filters_norm"] = normalized
                            if unknown:
                                entry["filters_unknown_keys"] = unknown
                        tool_summary.append(entry)

        turn_outputs.append(
            {
                "assistant_text": assistant_text,
                "route_pred": route_pred,
                "tool_summary": tool_summary,
                "contexts_used": contexts_used,
            },
        )
        if (
            contexts_used
            and not any(
                entry.get("tool") == "hydrate_items"
                for entry in tool_summary
                if isinstance(entry, dict)
            )
        ):
            tool_summary.append(
                {
                    "tool": "hydrate_items",
                    "status": "ok",
                    "input_keys": ["num_items"],
                    "input_preview": _preview(
                        {"num_items": len(contexts_used)},
                    ),
                },
            )

    return {
        "turn_outputs": turn_outputs,
        "case_tags": case_tags,
    }
