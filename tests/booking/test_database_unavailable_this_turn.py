"""Tests for `database_unavailable_this_turn`.

The booking dispatch node uses it to decide whether a booking turn that
returned normally must still be reported as failed. It reads `run_sql`
results structurally, so these tests build the `ToolMessage` shape LangChain
produces for a dict-returning tool: the dict encoded as JSON.
"""
# ruff: noqa: S101

from __future__ import annotations

import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from blue_horizon.agents.booking.factory import database_unavailable_this_turn


def _run_sql_result(error_kind: str | None) -> ToolMessage:
    """Build a `run_sql` tool message as the agent's tool node records it.

    Args:
        error_kind: The result's `error_kind`, or None for a successful query.

    Returns:
        ToolMessage: The tool result, its content encoded as JSON.

    """
    payload: dict[str, object] = {"status": "ok", "rows": [], "rowcount": 0}
    if error_kind is not None:
        payload = {"status": "error", "error": "failed", "error_kind": error_kind}
    return ToolMessage(
        content=json.dumps(payload), name="run_sql", tool_call_id="call-1",
    )


class TestDatabaseUnavailableThisTurn:
    """Only an `unavailable` `run_sql` result since the last guest message counts."""

    def test_unavailable_result_in_this_turn(self) -> None:
        """An unreachable database this turn is detected."""
        messages = [
            HumanMessage(content="Any suites free?"),
            _run_sql_result("unavailable"),
            AIMessage(content="Please try again shortly."),
        ]
        assert database_unavailable_this_turn(messages)

    def test_sql_error_is_not_an_outage(self) -> None:
        """A query error the model can rewrite is not reported as an outage."""
        messages = [
            HumanMessage(content="Any suites free?"),
            _run_sql_result("sql"),
            _run_sql_result(None),
            AIMessage(content="Suite 901 is free."),
        ]
        assert not database_unavailable_this_turn(messages)

    def test_outage_in_an_earlier_turn_is_ignored(self) -> None:
        """An outage before the latest guest message belongs to an earlier turn."""
        messages = [
            HumanMessage(content="Any suites free?"),
            _run_sql_result("unavailable"),
            AIMessage(content="Unavailable."),
            HumanMessage(content="How about now?"),
            _run_sql_result(None),
            AIMessage(content="Suite 901 is free."),
        ]
        assert not database_unavailable_this_turn(messages)

    def test_other_tools_and_non_json_content_are_ignored(self) -> None:
        """Only `run_sql` results with a decodable JSON object are read."""
        messages = [
            HumanMessage(content="Show my bookings."),
            ToolMessage(
                content=json.dumps({"error_kind": "unavailable"}),
                name="list_my_bookings",
                tool_call_id="call-1",
            ),
            ToolMessage(content="not json", name="run_sql", tool_call_id="call-2"),
        ]
        assert not database_unavailable_this_turn(messages)
