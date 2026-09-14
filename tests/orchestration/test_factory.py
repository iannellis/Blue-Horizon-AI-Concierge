"""Tests for the orchestration graph's failed-turn handling.

Drives the real compiled graph from `build_orchestration_agent` against a fake
`OrchestrationResources`: a stubbed router and sub-agent, a real `MemorySaver`,
and short timeouts. No model provider, Redis, or Postgres is involved.
"""

# ruff: noqa: S101
from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from psycopg_pool import PoolTimeout

from blue_horizon.agents.booking.write_ops import BookingUnavailableError
from blue_horizon.agents.orchestration.factory import (
    _group_turns,
    build_orchestration_agent,
)
from blue_horizon.agents.orchestration.models import RouteDecision

if TYPE_CHECKING:
    import pytest
    from langchain_core.runnables import RunnableConfig
    from langgraph.graph.state import CompiledStateGraph

    from blue_horizon.agents.orchestration.models import RouteStep
    from blue_horizon.agents.orchestration.resources import OrchestrationResources

_FACTORY_LOGGER = "blue_horizon.agents.orchestration.factory"
_SHORT_TIMEOUT_S = 0.05
_HANG_S = 5.0


class _FakeRouter:
    """Router stub that routes to a fixed step, raises, or hangs past its timeout."""

    def __init__(self) -> None:
        """Start as a router that routes every message to `info`."""
        self.step: RouteStep = "info"
        self.fail = False
        self.hang = False

    async def ainvoke(self, _messages: object) -> RouteDecision:
        """Return a routing decision, or misbehave as configured.

        Args:
            _messages: Ignored router input.

        Returns:
            RouteDecision: The configured `step`.

        Raises:
            ConnectionError: When `fail` is set, standing in for an
                unreachable model provider.

        """
        if self.hang:
            await asyncio.sleep(_HANG_S)
        if self.fail:
            msg = "network unreachable"
            raise ConnectionError(msg)
        return RouteDecision(step=self.step)


class _FakeSubAgent:
    """Sub-agent stub that replies, replies with nothing, raises, or hangs."""

    def __init__(self) -> None:
        """Start as a sub-agent that returns one fixed reply."""
        self.reply: str | None = "Here you go."
        self.fail = False
        self.fail_with_cause: OSError | None = None
        self.hang = False
        self.tool_results: list[BaseMessage] = []
        self.raise_exc: Exception | None = None

    async def ainvoke(self, _state: object, **_kwargs: object) -> dict[str, Any]:
        """Return a state patch, or misbehave as configured.

        Args:
            _state: Ignored conversation state.
            **_kwargs: Ignored keyword arguments, such as `config`.

        Returns:
            dict[str, Any]: A patch with any `tool_results` followed by the
            reply, or with no messages when `reply` is `None`.

        Raises:
            RuntimeError: When `fail` is set, or raised from
                `fail_with_cause` when that is set.
            Exception: `raise_exc`, raised as is, when that is set.

        """
        if self.hang:
            await asyncio.sleep(_HANG_S)
        if self.raise_exc is not None:
            raise self.raise_exc
        if self.fail_with_cause is not None:
            msg = "Connection error."
            raise RuntimeError(msg) from self.fail_with_cause
        if self.fail:
            msg = "sub-agent broke"
            raise RuntimeError(msg)
        if self.reply is None:
            return {"messages": []}
        return {"messages": [*self.tool_results, AIMessage(content=self.reply)]}


def _build_graph(router: _FakeRouter, sub_agent: _FakeSubAgent) -> CompiledStateGraph:
    """Compile the real orchestration graph around fake resources.

    Args:
        router: Router stub.
        sub_agent: Stub used for both the info and booking sub-agents.

    Returns:
        CompiledStateGraph: The graph under test, with its own checkpointer.

    """
    config = SimpleNamespace(
        messages=SimpleNamespace(error="Could not complete.", refusal="No."),
        orchestration=SimpleNamespace(
            router_timeout_s=_SHORT_TIMEOUT_S,
            info_timeout_s=_SHORT_TIMEOUT_S,
            booking_timeout_s=_SHORT_TIMEOUT_S,
        ),
    )
    resources = SimpleNamespace(
        config=config,
        router=router,
        get_system_prompt=lambda: "Route the message.",
        get_info_agent=lambda: sub_agent,
        get_booking_agent=lambda: sub_agent,
        checkpointer=MemorySaver(),
    )
    return build_orchestration_agent(
        resources=cast("OrchestrationResources", resources),
    )


def _run_turn(graph: CompiledStateGraph, text: str) -> dict[str, Any]:
    """Run one turn on a fixed thread and return the final state.

    Args:
        graph: The graph under test.
        text: The guest's message.

    Returns:
        dict[str, Any]: Final graph state after the turn.

    """
    config: RunnableConfig = {"configurable": {"thread_id": "t1"}}
    return cast(
        "dict[str, Any]",
        asyncio.run(
            graph.ainvoke({"messages": [HumanMessage(content=text)]}, config=config),
        ),
    )


def _contents(state: dict[str, Any]) -> list[object]:
    """Return the content of each message in a state's history.

    Args:
        state: Final graph state.

    Returns:
        list[object]: Message contents, in order.

    """
    messages = cast("list[BaseMessage]", state["messages"])
    return [msg.content for msg in messages]


def _run_sql_result(error_kind: str) -> ToolMessage:
    """Build a failed `run_sql` tool result as the booking agent records it.

    Args:
        error_kind: The result's `error_kind`.

    Returns:
        ToolMessage: The tool result, its content encoded as JSON.

    """
    payload = {"status": "error", "error": "failed", "error_kind": error_kind}
    return ToolMessage(
        content=json.dumps(payload), name="run_sql", tool_call_id="call-1",
    )


def _booking_unavailable() -> BookingUnavailableError:
    """Build the error a booking tool raises when its pool cannot connect.

    Returns:
        BookingUnavailableError: Chained from a `PoolTimeout`, as `write_ops`
        raises it.

    """
    msg = "Could not reach the database to complete this request."
    error = BookingUnavailableError(msg)
    error.__cause__ = PoolTimeout("couldn't get a connection after 10.00 sec")
    return error


class TestFailureLogging:
    """A network failure logs one line; any other failure keeps its traceback."""

    def test_router_network_failure_logs_one_line(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """An unreachable router logs a warning with no traceback."""
        router = _FakeRouter()
        router.fail = True
        graph = _build_graph(router, _FakeSubAgent())
        with caplog.at_level(logging.WARNING, logger=_FACTORY_LOGGER):
            _run_turn(graph, "Hello")
        records = _failure_records(caplog)
        assert len(records) == 1
        assert records[0].levelno == logging.WARNING
        assert records[0].exc_info is None
        assert "network unreachable" in records[0].getMessage()

    def test_wrapped_network_failure_logs_one_line(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A client error raised from an `OSError` names the root cause."""
        sub_agent = _FakeSubAgent()
        sub_agent.fail_with_cause = OSError("getaddrinfo failed")
        graph = _build_graph(_FakeRouter(), sub_agent)
        with caplog.at_level(logging.WARNING, logger=_FACTORY_LOGGER):
            _run_turn(graph, "Hello")
        records = _failure_records(caplog)
        assert len(records) == 1
        assert records[0].exc_info is None
        assert "getaddrinfo failed" in records[0].getMessage()

    def test_booking_database_outage_logs_one_line(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A booking tool that could not reach the database logs one warning."""
        sub_agent = _FakeSubAgent()
        sub_agent.raise_exc = _booking_unavailable()
        graph = _build_graph(_FakeRouter(), sub_agent)
        with caplog.at_level(logging.WARNING, logger=_FACTORY_LOGGER):
            _run_turn(graph, "Show my bookings.")
        records = _failure_records(caplog)
        assert len(records) == 1
        assert records[0].levelno == logging.WARNING
        assert records[0].exc_info is None
        assert "couldn't get a connection" in records[0].getMessage()

    def test_other_failure_keeps_traceback(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A sub-agent defect is logged at ERROR with its traceback."""
        sub_agent = _FakeSubAgent()
        sub_agent.fail = True
        graph = _build_graph(_FakeRouter(), sub_agent)
        with caplog.at_level(logging.WARNING, logger=_FACTORY_LOGGER):
            _run_turn(graph, "Hello")
        records = _failure_records(caplog)
        assert len(records) == 1
        assert records[0].levelno == logging.ERROR
        assert records[0].exc_info is not None


def _failure_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    """Return the captured records that report a node failure.

    Args:
        caplog: Pytest log capture fixture.

    Returns:
        list[logging.LogRecord]: Records whose message contains "failed".

    """
    return [r for r in caplog.records if " failed" in r.getMessage()]


class TestFailedTurn:
    """A failed turn records `turn_error` and leaves no trace in history."""

    def test_completed_turn_has_no_turn_error(self) -> None:
        """A normal turn keeps its question and reply, with no error."""
        graph = _build_graph(_FakeRouter(), _FakeSubAgent())
        state = _run_turn(graph, "What time is breakfast?")
        assert state.get("turn_error") is None
        assert _contents(state) == ["What time is breakfast?", "Here you go."]

    def test_router_exception_records_internal(self) -> None:
        """An unreachable router records `internal` and drops the turn."""
        router = _FakeRouter()
        router.fail = True
        graph = _build_graph(router, _FakeSubAgent())
        state = _run_turn(graph, "Tell me about your concert hall.")
        assert state["turn_error"] == "internal"
        assert _contents(state) == []

    def test_router_timeout_records_timeout(self) -> None:
        """A router past its timeout records `timeout`."""
        router = _FakeRouter()
        router.hang = True
        graph = _build_graph(router, _FakeSubAgent())
        state = _run_turn(graph, "Hello")
        assert state["turn_error"] == "timeout"
        assert _contents(state) == []

    def test_sub_agent_timeout_records_timeout(self) -> None:
        """A sub-agent past its timeout records `timeout`."""
        sub_agent = _FakeSubAgent()
        sub_agent.hang = True
        graph = _build_graph(_FakeRouter(), sub_agent)
        state = _run_turn(graph, "Hello")
        assert state["turn_error"] == "timeout"
        assert _contents(state) == []

    def test_sub_agent_exception_records_internal(self) -> None:
        """A sub-agent that raises records `internal`."""
        sub_agent = _FakeSubAgent()
        sub_agent.fail = True
        graph = _build_graph(_FakeRouter(), sub_agent)
        state = _run_turn(graph, "Hello")
        assert state["turn_error"] == "internal"
        assert _contents(state) == []

    def test_empty_turn_records_internal(self) -> None:
        """A turn that produced no reply is a failure, not an empty answer."""
        sub_agent = _FakeSubAgent()
        sub_agent.reply = None
        graph = _build_graph(_FakeRouter(), sub_agent)
        state = _run_turn(graph, "Hello")
        assert state["turn_error"] == "internal"
        assert _contents(state) == []

    def test_booking_database_outage_records_unavailable(self) -> None:
        """A booking reply written after a database outage is discarded."""
        router = _FakeRouter()
        router.step = "booking"
        sub_agent = _FakeSubAgent()
        sub_agent.tool_results = [_run_sql_result("unavailable")]
        graph = _build_graph(router, sub_agent)
        state = _run_turn(graph, "Any suites free?")
        assert state["turn_error"] == "unavailable"
        assert _contents(state) == []

    def test_booking_sql_error_completes_normally(self) -> None:
        """A query error the model recovered from is not an outage."""
        router = _FakeRouter()
        router.step = "booking"
        sub_agent = _FakeSubAgent()
        sub_agent.tool_results = [_run_sql_result("sql")]
        graph = _build_graph(router, sub_agent)
        state = _run_turn(graph, "Any suites free?")
        assert state.get("turn_error") is None
        assert _contents(state)[-1] == "Here you go."

    def test_booking_tool_outage_records_unavailable(self) -> None:
        """A booking tool raising `BookingUnavailableError` records `unavailable`."""
        router = _FakeRouter()
        router.step = "booking"
        sub_agent = _FakeSubAgent()
        sub_agent.raise_exc = _booking_unavailable()
        graph = _build_graph(router, sub_agent)
        state = _run_turn(graph, "Show my bookings.")
        assert state["turn_error"] == "unavailable"
        assert _contents(state) == []

    def test_info_turn_is_not_checked_for_outages(self) -> None:
        """Only the booking dispatch node discards a reply on an outage."""
        sub_agent = _FakeSubAgent()
        sub_agent.tool_results = [_run_sql_result("unavailable")]
        graph = _build_graph(_FakeRouter(), sub_agent)
        state = _run_turn(graph, "Any suites free?")
        assert state.get("turn_error") is None

    def test_failure_keeps_earlier_completed_turns(self) -> None:
        """Only the failed turn is dropped; earlier history survives."""
        sub_agent = _FakeSubAgent()
        graph = _build_graph(_FakeRouter(), sub_agent)
        _run_turn(graph, "First question")
        sub_agent.fail = True
        state = _run_turn(graph, "Second question")
        assert state["turn_error"] == "internal"
        assert _contents(state) == ["First question", "Here you go."]

    def test_next_turn_clears_an_earlier_failure(self) -> None:
        """A failure held in the checkpoint is not reported on the next turn."""
        router = _FakeRouter()
        router.fail = True
        graph = _build_graph(router, _FakeSubAgent())
        _run_turn(graph, "Tell me about your concert hall.")
        router.fail = False
        state = _run_turn(graph, "Tell me about your concert hall.")
        assert state.get("turn_error") is None
        assert _contents(state) == [
            "Tell me about your concert hall.",
            "Here you go.",
        ]


class TestGroupTurns:
    """`_group_turns` pairs each HumanMessage with the AIMessages after it."""

    def test_groups_replies_and_drops_leading_ai(self) -> None:
        """AI messages before any human message belong to no turn."""
        stray = AIMessage(content="stray")
        first = HumanMessage(content="q1")
        first_reply = AIMessage(content="a1")
        second = HumanMessage(content="q2")
        turns = _group_turns([stray, first, first_reply, second])
        assert turns == [(first, [first_reply]), (second, [])]
