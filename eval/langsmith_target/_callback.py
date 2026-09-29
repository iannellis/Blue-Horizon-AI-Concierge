"""Async callback handler for capturing routing and tool artifacts.

Contains ``SearchRoomsOutput`` (the typed room search payload model) and
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

# Matches blue_horizon.agents.booking.factory._SEARCH_TOOL_NAME.
_SEARCH_TOOL_NAME = "search_rooms"

# Maps a propose_* tool name to the proposal action it creates -- mirrors
# eval.evaluators._booking._PROPOSE_ACTIONS and proposals.ProposalAction.
_PROPOSE_TOOL_NAMES: frozenset[str] = frozenset(
    {"propose_booking", "propose_cancellation", "propose_modification"},
)


def _parse_tool_message_content(output: Any) -> Any:  # noqa: ANN401
    """Unwrap a `ToolMessage` and parse its string content into Python data.

    Shared by the `search_rooms` and `propose_*` capture paths, both of which
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


class SearchRoomsOutput(BaseModel):
    """Typed payload returned by the booking agent's ``search_rooms`` tool.

    This model captures the subset of fields the evaluation harness cares about
    when summarizing tool activity. Only a one-room sample of ``rooms`` is kept
    in summaries, to keep artifacts compact and stable across runs.

    Attributes:
        status: Tool status string ("ok" or "error").
        matching_count: Number of rooms matching the search, of which at
            most ``[booking.agent].max_search_results`` are returned.
        rooms: The rooms returned, when present.
        error: Error message when the tool fails.
        error_kind: Message-independent failure classification (see
            `resources.SqlErrorKind`), present only on failure. Lets a
            consumer like the stress harness's outcome classifier key off
            structure instead of matching this tool's error text.
        limit_note: Note that the requested ``limit`` was clamped to the
            configured maximum, present only when it was.

    """

    status: str | None = None
    matching_count: int | None = None
    error: str | None = None
    error_kind: str | None = None
    rooms: list[dict[str, Any]] | None = None
    limit_note: str | None = None


def _parse_search_rooms_payload(
    output: SearchRoomsOutput | Mapping[str, object],
) -> SearchRoomsOutput | None:
    """Coerce a search_rooms tool output into a validated `SearchRoomsOutput`.

    Args:
        output: Raw or already-typed search_rooms tool output.

    Returns:
        The validated payload, or `None` if `output` is neither a
        `SearchRoomsOutput` nor a mapping that validates as one.

    """
    if isinstance(output, SearchRoomsOutput):
        return output
    if isinstance(output, Mapping):
        try:
            return SearchRoomsOutput.model_validate(dict(output))
        except ValidationError:
            return None
    return None


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
            self._capture_merge_node(outputs)

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

    def _capture_merge_node(self, outputs: dict[str, Any]) -> None:
        """Append a merge tool-summary entry and populate contexts_used.

        Args:
            outputs: Merge-node output containing ``"top_results"``.

        """
        results = outputs["top_results"]
        self.tool_summary.append(
            {
                "tool": "merge",
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
            if tool_name == _SEARCH_TOOL_NAME:
                entry["search_args"] = _json_safe(dict(inputs))
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

        if tool_name == _SEARCH_TOOL_NAME:
            actual_output = _parse_tool_message_content(output)
            if isinstance(actual_output, Mapping):
                self._capture_search_rooms(actual_output, entry)
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

    def _capture_search_rooms(
        self,
        output: SearchRoomsOutput | Mapping[str, object],
        base_entry: dict[str, Any] | None = None,
    ) -> None:
        """Capture a compact search_rooms summary, including a one-room sample.

        Args:
            output: search_rooms tool output.
            base_entry: Optional base entry with input previews.

        """
        payload = _parse_search_rooms_payload(output)
        if payload is None:
            return
        summary = dict(base_entry or {})
        summary["tool"] = _SEARCH_TOOL_NAME
        summary["status"] = payload.status or summary.get("status") or "ok"
        summary["matching_count"] = payload.matching_count
        if isinstance(payload.rooms, list) and payload.rooms:
            summary["rows"] = _compact_rows(payload.rooms, max_rows=1)
        if payload.error:
            summary["error"] = payload.error
        if payload.error_kind:
            summary["error_kind"] = payload.error_kind
        if payload.limit_note:
            summary["limit_note"] = payload.limit_note
        summary["output_preview"] = _preview(
            {
                "status": payload.status,
                "matching_count": payload.matching_count,
                "returned": len(payload.rooms or []),
            },
        )
        self.tool_summary.append(summary)

        # Add search results to contexts_used for judge LLM evaluation
        if payload.matching_count is not None and payload.status == "ok":
            self.contexts_used.append(
                f"Room search: {payload.matching_count} matching rooms",
            )
        for room in payload.rooms or []:
            if isinstance(room, Mapping):
                # Format as readable key-value pairs
                room_str = ", ".join(
                    f"{k}: {v}" for k, v in room.items() if v is not None
                )
                if room_str:
                    self.contexts_used.append(f"Room search result: {room_str}")

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

