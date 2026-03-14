"""Shared constants and utilities for Blue Horizon evaluators."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from eval._utils import truncate

if TYPE_CHECKING:
    from langsmith.schemas import Example, Run

# Required tool names for info agent turns
_INFO_REQUIRED_TOOLS = (
    "query_faq",
    "query_amenities",
    "query_services",
    "reranker",
)

# Tool names used by rooms agent SQL generation
_SQL_TOOL_NAMES = ("run_sql",)


def _iter_turn_outputs(run: Run) -> list[dict[str, Any]]:
    """Extract the turn_outputs list from a LangSmith run.

    Args:
        run: LangSmith run object.

    Returns:
        List of turn output dicts (empty if missing or invalid).

    """
    outputs = run.outputs or {}
    turn_outputs = outputs.get("turn_outputs") or []
    if isinstance(turn_outputs, list):
        return [t for t in turn_outputs if isinstance(t, dict)]
    return []


def _rag_extract_turn_inputs(
    run: Run,
    example: Example,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[object]]:
    """Extract turn data from run outputs and example inputs.

    Args:
        run: LangSmith run object containing turn outputs.
        example: LangSmith example object containing dataset turns.

    Returns:
        Tuple of (turn_outputs, example_turns, reference_answers).

    """
    outputs = run.outputs or {}
    turn_outputs_raw = outputs.get("turn_outputs") or []
    turn_outputs = [t for t in turn_outputs_raw if isinstance(t, dict)]

    inputs = example.inputs or {}
    example_turns_raw = inputs.get("turns") or []
    example_turns = [t for t in example_turns_raw if isinstance(t, dict)]

    reference_answers = inputs.get("reference_answers") or []
    if not isinstance(reference_answers, list):
        reference_answers = []

    return turn_outputs, example_turns, reference_answers


def _rag_extract_reference(
    example_turn: dict[str, object],
    reference_answers: list[object],
    index: int,
    max_chars: int,
) -> str | None:
    """Extract a reference answer for a turn if available.

    Args:
        example_turn: Example turn dict containing potential reference fields.
        reference_answers: Optional list of reference answers from example inputs.
        index: Turn index used to lookup reference list entries.
        max_chars: Maximum length for the reference string.

    Returns:
        Truncated reference string if available, otherwise None.

    """
    reference = example_turn.get("reference")
    if reference is None:
        reference = example_turn.get("expected_answer")
    if reference is None:
        reference = example_turn.get("ground_truth")
    if reference is None and index < len(reference_answers):
        reference = reference_answers[index]
    if reference is None:
        return None
    text = truncate(reference, max_chars)
    return text if text else None


def _get_example_turns(example: Example) -> list[dict[str, Any]]:
    """Extract the turns list from a LangSmith example input.

    Args:
        example: LangSmith example object.

    Returns:
        List of example turn dicts (empty if missing or invalid).

    """
    inputs = example.inputs or {}
    turns = inputs.get("turns") or []
    if isinstance(turns, list):
        return [t for t in turns if isinstance(t, dict)]
    return []
