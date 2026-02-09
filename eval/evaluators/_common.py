"""Shared constants and utilities for Blue Horizon evaluators."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langsmith.schemas import Example, Run

# Required tool names for info agent turns
_INFO_REQUIRED_TOOLS = (
    "query_faq",
    "query_amenities",
    "query_services",
    "reranker",
    "hydrate_items",
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
