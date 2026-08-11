"""Async callback handler for capturing routing and tool artifacts.

Contains ``RunSqlOutput`` (the typed SQL tool payload model) and
``EvalCaptureCallback`` (the per-turn capture handler).
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from langchain_core.callbacks import AsyncCallbackHandler

if TYPE_CHECKING:
    from uuid import UUID
from langchain_core.messages import ToolMessage
from pydantic import BaseModel, ValidationError

from eval._utils import coerce_int as _coerce_int
from eval._utils import json_safe as _json_safe
from eval.langsmith_target._filter_utils import (
    _INFO_FILTER_KEYS,
    _normalize_info_filters_strict,
)
from eval.langsmith_target._text_utils import (
    _get_tool_name,
    _input_keys,
    _preview,
)

_ROUTE_KEY = "route"  # key from orchestration.py

# Maps a propose_* tool name to the proposal action it creates -- mirrors
# eval.evaluators._booking._PROPOSE_ACTIONS and proposals.ProposalAction.
_PROPOSE_TOOL_NAMES: frozenset[str] = frozenset(
    {"propose_booking", "propose_cancellation", "propose_modification"},
)


def _parse_tool_message_content(output: Any) -> Any:  # noqa: ANN401
    """Unwrap a `ToolMessage` and parse its string content into Python data.

    Shared by the `run_sql` and `propose_*` capture paths, both of which
    receive either a raw dict (direct tool return) or a `ToolMessage` whose
    `content` is a JSON or Python-repr string (needing `Decimal(...)` /
    `datetime.date(...)` preprocessing before `ast.literal_eval`).

    Args:
        output: Raw tool output payload.

    Returns:
        Parsed Python data (typically a dict), or the original value if it
        was not a `ToolMessage`-wrapped string.

    """
    actual_output = output
    if isinstance(output, ToolMessage):
        actual_output = output.content
        if isinstance(actual_output, str):
            try:
                actual_output = json.loads(actual_output)
            except json.JSONDecodeError:
                cleaned = re.sub(
                    r"Decimal\('([^']*)'\)",
                    r'"\1"',
                    actual_output,
                )
                cleaned = re.sub(
                    r"datetime\.date\((\d+),\s*(\d+),\s*(\d+)\)",
                    lambda m: (
                        f'"{m.group(1)}-'
                        f"{int(m.group(2)):02d}-"
                        f'{int(m.group(3)):02d}"'
                    ),
                    cleaned,
                )
                try:  # noqa: SIM105
                    actual_output = ast.literal_eval(cleaned)
                except (ValueError, SyntaxError):
                    pass
    return actual_output


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
            {str(key): _json_safe(val) for key, val in row.items()},
        )
    return safe_rows


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
        truncated: Whether the tool output was truncated by the agent guardrails.
        error: User-facing error message when the tool fails.

    """

    status: str | None = None
    rowcount: int | None = None
    truncated: bool | None = None
    error: str | None = None
    rows: list[dict[str, Any]] | None = None


class ProposeOutput(BaseModel):
    """Typed payload returned by a `propose_*` booking tool.

    Attributes:
        status: Tool status string (`"proposed"` on success, `"error"` on
            refusal -- for example, a requested night is no longer available).
        proposal_id: Identifier of the created proposal, present on success.
        error: User-facing error message, present when the tool refuses.

    """

    status: str | None = None
    proposal_id: str | None = None
    error: str | None = None


class EvalCaptureCallback(AsyncCallbackHandler):
    """Capture routing decisions and tool artifacts for a single turn.

    Attributes:
        route_pred: Router decision captured for the turn.
        tool_summary: Compact summaries of tools executed in this turn.
        contexts_used: Context snippets captured from retrieval output.
        confirm_receipt_text: App-authored receipt text from a post-turn
            auto-confirm, if one committed a proposal this turn. Kept
            separate from `assistant_text` because the model generates its
            response before the auto-confirm runs -- see
            `capture_confirm_result`.

    """

    route_pred: str | None
    tool_summary: list[dict[str, Any]]
    contexts_used: list[str]
    confirm_receipt_text: str | None
    _pending_tool_entries: dict[UUID, dict[str, Any]]
    _parsed_query: dict[str, Any] | None

    def __init__(self) -> None:
        """Initialize the callback handler."""
        super().__init__()
        self.route_pred = None
        self.tool_summary = []
        self.contexts_used = []
        self.confirm_receipt_text = None
        self._pending_tool_entries = {}
        self._parsed_query = None

    async def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Capture router outputs and info-DAG node results for the current turn.

        Args:
            outputs: Chain outputs from LangChain/LangGraph.
            run_id: LangChain run ID for the finished chain.
            parent_run_id: Optional parent run ID.
            tags: Optional tags emitted by the chain.
            **kwargs: Additional keyword arguments.

        """
        _ = run_id, parent_run_id, tags, kwargs
        if not isinstance(outputs, dict):
            return

        # Capture orchestration router decision.
        if _ROUTE_KEY in outputs:
            route_val = outputs.get(_ROUTE_KEY)
            if isinstance(route_val, str):
                self.route_pred = route_val
            return

        # Full-state events (e.g., graph start/end) contain messages alongside
        # other keys and should not be mistaken for individual node outputs.
        if "messages" in outputs:
            return

        self._dispatch_info_dag_node(outputs)

    def _dispatch_info_dag_node(self, outputs: dict[str, Any]) -> None:
        """Route a single info-DAG node output to the appropriate capture handler.

        Args:
            outputs: State-patch dict returned by one info DAG node.

        """
        if "parsed" in outputs:
            self._capture_parse_node(outputs)
        elif "faq_results" in outputs:
            self._capture_faq_node(outputs)
        elif "amenities_results" in outputs or "services_results" in outputs:
            self._capture_catalog_node(outputs)
        elif "top_results" in outputs:
            self._capture_rerank_node(outputs)

    def _capture_parse_node(self, outputs: dict[str, Any]) -> None:
        """Store the parsed query and append a parser tool-summary entry.

        Args:
            outputs: Parse-node output containing ``"parsed"``.

        """
        parsed_raw = outputs["parsed"]
        if hasattr(parsed_raw, "model_dump"):
            parsed_dict: dict[str, Any] = parsed_raw.model_dump()
        elif isinstance(parsed_raw, dict):
            parsed_dict = parsed_raw
        else:
            return
        self._parsed_query = parsed_dict
        self.tool_summary.append(
            {
                "tool": "parser",
                "status": "ok",
                "parsed_query": {k: v for k, v in parsed_dict.items() if v is not None},
            },
        )

    def _capture_faq_node(self, outputs: dict[str, Any]) -> None:
        """Append a query_faq tool-summary entry.

        Args:
            outputs: FAQ-node output containing ``"faq_results"``.

        """
        results = outputs["faq_results"]
        self.tool_summary.append(
            {
                "tool": "query_faq",
                "status": "ok",
                "count": len(results) if isinstance(results, list) else 0,
            },
        )

    def _capture_catalog_node(self, outputs: dict[str, Any]) -> None:
        """Append a query_amenities or query_services tool-summary entry with filters.

        Args:
            outputs: Catalog-node output containing ``"amenities_results"`` or
                ``"services_results"``.

        """
        for state_key, tool_name in (
            ("amenities_results", "query_amenities"),
            ("services_results", "query_services"),
        ):
            if state_key not in outputs:
                continue
            results = outputs[state_key]
            entry: dict[str, Any] = {
                "tool": tool_name,
                "status": "ok",
                "count": len(results) if isinstance(results, list) else 0,
            }
            if self._parsed_query:
                raw_filters = {
                    k: v
                    for k, v in self._parsed_query.items()
                    if k in _INFO_FILTER_KEYS and v is not None
                }
                if raw_filters:
                    norm, unknown = _normalize_info_filters_strict(raw_filters)
                    entry["filters"] = raw_filters
                    entry["filters_norm"] = norm
                    if unknown:
                        entry["filters_unknown_keys"] = unknown
            self.tool_summary.append(entry)
            return

    def _capture_rerank_node(self, outputs: dict[str, Any]) -> None:
        """Append a reranker tool-summary entry and populate contexts_used.

        Args:
            outputs: Rerank-node output containing ``"top_results"``.

        """
        results = outputs["top_results"]
        self.tool_summary.append(
            {
                "tool": "reranker",
                "status": "ok",
                "count": len(results) if isinstance(results, list) else 0,
            },
        )
        for item in results or []:
            context = self._item_to_context(item)
            if context:
                self.contexts_used.append(context)

    @staticmethod
    def _item_to_context(item: Any) -> str | None:  # noqa: ANN401
        """Convert a single retrieval item to a context string.

        Args:
            item: A ``RetrievalItem`` instance or plain dict with ``text`` and
                ``metadata`` fields.

        Returns:
            Context string combining text and metadata, or ``None`` when the
            item has no usable text.

        """
        if hasattr(item, "text"):
            text: object = item.text
            metadata: dict[str, Any] = getattr(item, "metadata", None) or {}
        elif isinstance(item, dict):
            text = item.get("text", "")
            metadata = item.get("metadata") or {}
        else:
            return None
        if not isinstance(text, str) or not text:
            return None
        context = text
        if metadata:
            metadata_str = ", ".join(
                f"{k}: {v}" for k, v in metadata.items() if v is not None
            )
            if metadata_str:
                context = f"{context}\n[Metadata: {metadata_str}]"
        return context

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
        if isinstance(inputs, Mapping):
            raw_k = inputs.get("k")
            if raw_k is None:
                raw_k = inputs.get("top_k")
            k_value = _coerce_int(raw_k)
            if k_value is not None:
                entry["k"] = k_value
            if tool_name == "run_sql":
                raw_sql = inputs.get("query")
                if isinstance(raw_sql, str):
                    entry["sql_query"] = raw_sql
        self._pending_tool_entries[run_id] = entry

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
        entry = self._pending_tool_entries.pop(run_id, None)
        tool_name = _get_tool_name(kwargs)
        if tool_name is None and isinstance(entry, dict):
            tool_name = entry.get("tool")

        if tool_name == "run_sql":
            actual_output = _parse_tool_message_content(output)
            if isinstance(actual_output, Mapping):
                self._capture_run_sql(actual_output, entry)
            return
        if tool_name in _PROPOSE_TOOL_NAMES:
            actual_output = _parse_tool_message_content(output)
            if isinstance(actual_output, Mapping):
                self._capture_propose(tool_name, actual_output, entry)
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
            summary["rows"] = _compact_rows(payload.rows, max_rows=1)
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

    def _capture_propose(
        self,
        tool_name: str,
        output: ProposeOutput | Mapping[str, object],
        base_entry: dict[str, Any] | None = None,
    ) -> None:
        """Capture a compact propose_* summary: status, proposal id, error.

        Args:
            tool_name: One of `propose_booking`, `propose_cancellation`,
                `propose_modification`.
            output: Parsed propose_* tool output.
            base_entry: Optional base entry with input previews.

        """
        if isinstance(output, ProposeOutput):
            payload = output
        elif isinstance(output, Mapping):
            try:
                payload = ProposeOutput.model_validate(dict(output))
            except ValidationError:
                return
        else:
            return
        summary = dict(base_entry or {})
        summary["tool"] = tool_name
        summary["status"] = payload.status or summary.get("status") or "proposed"
        if payload.proposal_id:
            summary["proposal_id"] = payload.proposal_id
        if payload.error:
            summary["error"] = payload.error
        self.tool_summary.append(summary)

    def capture_confirm_result(
        self,
        *,
        action: str,
        already_confirmed: bool,
        result: dict[str, Any],
        receipt_text: str,
    ) -> None:
        """Append a confirm-outcome tool-summary entry for a committed proposal.

        `commit_booking`/`cancel_booking`/`modify_booking` run through
        `ProposalStore.confirm()`, invoked by the harness's auto-confirm
        helper -- never a LangChain-tracked runnable, so `on_tool_end` never
        observes it. The auto-confirm helper calls this directly right after
        `confirm()` returns, so booking evaluators see the same "did a write
        actually happen" picture a human clicking Confirm would have
        produced.

        `receipt_text` is stored separately on `confirm_receipt_text` rather
        than folded into `assistant_text`: in the real app this receipt is a
        distinct, app-authored chat message that appears *after* the
        assistant's own turn (see `_confirm.auto_confirm_pending_proposal`),
        never something the model itself said or could have referenced.

        Args:
            action: Which kind of write was confirmed (`book`, `cancel`, or
                `modify`).
            already_confirmed: True if this replayed a cached result rather
                than performing a new write.
            result: JSON-safe write result, from
                `blue_horizon.agents.booking.receipts.serialize_write_result`.
            receipt_text: App-authored receipt text for this outcome, from
                `blue_horizon.agents.booking.receipts.receipt_message`.

        """
        self.tool_summary.append(
            {
                "tool": "confirm_booking",
                "status": "ok",
                "action": action,
                "already_confirmed": already_confirmed,
                "result": result,
            },
        )
        self.confirm_receipt_text = receipt_text

    def capture_confirm_error(self, *, action: str, error: str) -> None:
        """Append a confirm-failure tool-summary entry.

        Args:
            action: Which kind of write was attempted (`book`, `cancel`, or
                `modify`).
            error: User-facing error message from the failed confirm.

        """
        self.tool_summary.append(
            {
                "tool": "confirm_booking",
                "status": "error",
                "action": action,
                "error": error,
            },
        )

