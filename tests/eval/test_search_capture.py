"""Tests that `EvalCaptureCallback` records `search_rooms` results faithfully.

The callback falls through to ``status="ok"`` for any tool it does not
recognise. Before `search_rooms` had its own capture path, a renamed tool
would have been recorded as succeeding even when it failed, silently
zeroing `booking_tool_errors` and the stress harness's outage detection.
These run a real LangChain tool, so the result reaches the callback the same
way it does in the agent.
"""

# ruff: noqa: S101
from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from eval.langsmith_target._callback import EvalCaptureCallback


class _Args(BaseModel):
    """Stand-in argument model for a search tool.

    Attributes:
        min_floor: Any filter, so the tool takes one argument.
        limit: A defaulted argument, as on the real tool, to show that
            defaults the model did not send are not captured.

    """

    min_floor: int | None = None
    limit: int = 4


def _run_search_tool(result: dict[str, Any]) -> EvalCaptureCallback:
    """Invoke a `search_rooms` tool returning `result` and capture it.

    Args:
        result: What the tool returns.

    Returns:
        The callback after the tool call.

    """

    async def _search(**_: Any) -> dict[str, Any]:  # noqa: ANN401
        """Return the canned result.

        Args:
            **_: Ignored search arguments.

        Returns:
            `result`.

        """
        return result

    tool = StructuredTool.from_function(
        coroutine=_search,
        name="search_rooms",
        description="Search rooms.",
        args_schema=_Args,
    )
    callback = EvalCaptureCallback()
    asyncio.run(
        tool.ainvoke(
            {
                "type": "tool_call",
                "id": "call-1",
                "name": "search_rooms",
                "args": {"min_floor": 20},
            },
            config={"callbacks": [callback]},
        ),
    )
    return callback


class TestSearchRoomsCapture:
    """A `search_rooms` result keeps its status, error kind, and count."""

    def test_error_result_is_captured_as_error(self) -> None:
        """An outage is recorded as an error with its error kind."""
        callback = _run_search_tool(
            {
                "status": "error",
                "matching_count": 0,
                "rooms": [],
                "error": "DATABASE_UNAVAILABLE: ...",
                "error_kind": "unavailable",
            },
        )
        [entry] = callback.tool_summary
        assert entry["tool"] == "search_rooms"
        assert entry["status"] == "error"
        assert entry["error_kind"] == "unavailable"
        assert entry["search_args"] == {"min_floor": 20}

    def test_ok_result_records_count_and_contexts(self) -> None:
        """A success records matching_count and one context line per room."""
        callback = _run_search_tool(
            {
                "status": "ok",
                "matching_count": 12,
                "rooms": [
                    {"room_number": 2001, "floor": 20},
                    {"room_number": 2002, "floor": 20},
                ],
            },
        )
        [entry] = callback.tool_summary
        assert entry["status"] == "ok"
        assert entry["matching_count"] == 12  # noqa: PLR2004
        assert entry["rows"] == [{"room_number": 2001, "floor": 20}]
        assert "Room search: 12 matching rooms" in callback.contexts_used
        assert any("2002" in context for context in callback.contexts_used)

    def test_search_args_hold_only_model_supplied_keys(self) -> None:
        """A defaulted argument the model did not send is not captured."""
        callback = _run_search_tool({"status": "ok", "matching_count": 0})
        [entry] = callback.tool_summary
        assert entry["search_args"] == {"min_floor": 20}
        assert "limit_note" not in entry

    def test_limit_note_is_captured(self) -> None:
        """A clamp note from the tool is copied into the summary."""
        note = "Asked for 50 rooms; returned the first 15."
        callback = _run_search_tool(
            {"status": "ok", "matching_count": 40, "rooms": [], "limit_note": note},
        )
        [entry] = callback.tool_summary
        assert entry["limit_note"] == note
