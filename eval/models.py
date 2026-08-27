"""Typed wire shapes for data the eval harness reads back from LangSmith.

``TurnOutput``/``ToolSummaryEntry`` mirror what
``eval.langsmith_target._callback.EvalCaptureCallback`` writes into a run's
``outputs["turn_outputs"]``; ``ExampleTurn`` mirors one entry of a dataset
case's ``inputs["turns"]``. Evaluators parse once, at the LangSmith
Run/Example boundary, via ``_validate_list`` below (reused by
``eval.evaluators._common``'s extraction functions), and read typed
attributes from then on instead of re-checking ``isinstance`` at every call
site.

Every model uses ``extra="allow"``: both producers evolve independently of
this file, and an unrecognised key should survive a round trip rather than
be silently dropped.
"""

from __future__ import annotations

from typing import Any

from pydantic import AliasChoices, BaseModel, Field, ValidationError, field_validator


def _validate_list[T: BaseModel](model: type[T], raw: object) -> list[T]:
    """Validate a raw value into a list of model instances, best-effort.

    Args:
        model: Pydantic model class to validate each list entry against.
        raw: Raw value from a LangSmith run/example, expected to be a list
            of dicts but not guaranteed to be.

    Returns:
        A list of validated model instances. Non-list input is treated as
        empty, and any entry that fails validation is dropped rather than
        raising -- run/example payloads come from external, occasionally
        malformed data, and one bad turn should not blank out an otherwise
        usable run.

    """
    if not isinstance(raw, list):
        return []
    items: list[T] = []
    for entry in raw:
        try:
            items.append(model.model_validate(entry))
        except (ValidationError, TypeError):
            continue
    return items


class ToolSummaryEntry(BaseModel):
    """One compact tool-call summary captured by ``EvalCaptureCallback``.

    Only the fields evaluators actually read are declared here. Every other
    key the callback captures (``input_keys``, ``k``, ``sql_query``,
    ``count``, ``filters``, ``rows``, ``proposal_id``, ``action``,
    ``result``, ...) survives parsing via ``extra="allow"`` and remains
    reachable with ``getattr(entry, key, None)``, but has no typed
    attribute here.

    Attributes:
        tool: Tool name, e.g. "run_sql", "propose_booking", "confirm_booking".
        status: Tool status string ("ok", "error", "started", "proposed", ...).
        error: User-facing error message, present when the tool failed.
        rowcount: Rows returned or affected, present on run_sql entries.
        already_confirmed: Whether a confirm_booking entry replayed a
            cached result rather than performing a new write.
        filters_norm: Canonicalized amenity/service filters, when the info
            agent applied any.
        filters_unknown_keys: Filter keys the harness could not
            canonicalize, if any.
        input_preview: Truncated preview of the tool's input, captured on
            tool start.
        output_preview: Truncated preview of the tool's output.
        error_preview: Truncated preview of an on_tool_error failure.
        parsed_query: The parser node's structured query output, when this
            entry is a "parser" tool-summary entry.

    """

    model_config = {"extra": "allow"}

    tool: str | None = None
    status: str | None = None
    error: str | None = None
    rowcount: int | None = None
    already_confirmed: bool | None = None
    filters_norm: dict[str, Any] | None = None
    filters_unknown_keys: list[str] = Field(default_factory=list)
    input_preview: str | None = None
    output_preview: str | None = None
    error_preview: str | None = None
    parsed_query: dict[str, Any] | None = None


class TurnOutput(BaseModel):
    """One turn's captured outputs, as produced by ``EvalCaptureCallback``.

    Attributes:
        route_pred: Router decision captured for the turn.
        assistant_text: The assistant's response text for the turn.
        tool_summary: Compact summaries of tools executed in this turn.
            Entries that fail to parse are dropped individually rather than
            invalidating the whole turn.
        contexts_used: Context snippets captured from retrieval output.
        confirm_receipt_text: App-authored receipt text from a post-turn
            auto-confirm, when one committed a proposal this turn.

    """

    model_config = {"extra": "allow"}

    route_pred: str | None = None
    assistant_text: str | None = None
    tool_summary: list[ToolSummaryEntry] = Field(default_factory=list)
    contexts_used: list[str] = Field(default_factory=list)
    confirm_receipt_text: str | None = None

    @field_validator("tool_summary", mode="before")
    @classmethod
    def _coerce_tool_summary(cls, value: object) -> list[ToolSummaryEntry]:
        """Best-effort parse ``tool_summary``, dropping malformed entries.

        Args:
            value: Raw ``tool_summary`` value from the run output.

        Returns:
            Parsed tool-summary entries; non-list input yields an empty list.

        """
        return _validate_list(ToolSummaryEntry, value)

    @field_validator("contexts_used", mode="before")
    @classmethod
    def _coerce_contexts_used(cls, value: object) -> list[str]:
        """Filter ``contexts_used`` down to non-empty strings.

        The producer (``EvalCaptureCallback._item_to_context``) only ever
        appends non-empty strings, so this only guards against malformed
        external data, not a real production shape.

        Args:
            value: Raw ``contexts_used`` value from the run output.

        Returns:
            The subset of ``value`` that is non-empty strings, or an empty
            list when ``value`` is not a list at all.

        """
        if not isinstance(value, list):
            return []
        return [item for item in value if isinstance(item, str) and item]


class ExampleTurn(BaseModel):
    """One turn of a dataset case: the guest message plus optional labels.

    Attributes:
        user: The guest's message text for this turn.
        reference: Reference answer text for RAG/reference-subset scoring.
            Accepts the legacy aliases ``expected_answer``/``ground_truth``
            for datasets that used those names instead; whichever of the
            three keys is present first (in that order) wins.
        expected_route: Expected router decision ("info" or "booking").
        expected_filters: Expected amenity/service filters for this turn,
            when the case labels one. ``None`` means the case does not
            label filter expectations for this turn.
        expected_success: Whether a booking write attempted in this turn is
            expected to succeed.
        expect_injection: Whether this turn is labeled as an injection
            attempt, as a bool or the string "true".

    """

    model_config = {"extra": "allow", "populate_by_name": True}

    user: str | None = None
    reference: str | None = Field(
        default=None,
        validation_alias=AliasChoices("reference", "expected_answer", "ground_truth"),
    )
    expected_route: str | None = None
    expected_filters: dict[str, Any] | None = None
    expected_success: bool | None = None
    expect_injection: bool | str | None = None
