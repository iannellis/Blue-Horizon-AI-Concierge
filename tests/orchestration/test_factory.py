"""Tests for the orchestration graph's failed-turn handling.

Drives the real compiled graph from `build_orchestration_agent` against a fake
`OrchestrationResources`: a stubbed router and sub-agent, a real `MemorySaver`,
and short timeouts. No model provider, Redis, or Postgres is involved.
"""

# ruff: noqa: S101
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from blue_horizon.agents.orchestration.factory import (
    _group_turns,
    build_orchestration_agent,
)
from blue_horizon.agents.orchestration.models import RouteDecision

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig
    from langgraph.graph.state import CompiledStateGraph

    from blue_horizon.agents.orchestration.resources import OrchestrationResources

_SHORT_TIMEOUT_S = 0.05
_HANG_S = 5.0


class _FakeRouter:
    """Router stub that routes to `info`, raises, or hangs past its timeout."""

    def __init__(self) -> None:
        """Start as a router that routes every message to `info`."""
        self.fail = False
        self.hang = False

    async def ainvoke(self, _messages: object) -> RouteDecision:
        """Return a routing decision, or misbehave as configured.

        Args:
            _messages: Ignored router input.

        Returns:
            RouteDecision: Always the `info` step.

        Raises:
            ConnectionError: When `fail` is set, standing in for an
                unreachable model provider.

        """
        if self.hang:
            await asyncio.sleep(_HANG_S)
        if self.fail:
            msg = "network unreachable"
            raise ConnectionError(msg)
        return RouteDecision(step="info")


class _FakeSubAgent:
    """Sub-agent stub that replies, replies with nothing, raises, or hangs."""

    def __init__(self) -> None:
        """Start as a sub-agent that returns one fixed reply."""
        self.reply: str | None = "Here you go."
        self.fail = False
        self.hang = False

    async def ainvoke(self, _state: object, **_kwargs: object) -> dict[str, Any]:
        """Return a state patch, or misbehave as configured.

        Args:
            _state: Ignored conversation state.
            **_kwargs: Ignored keyword arguments, such as `config`.

        Returns:
            dict[str, Any]: A patch with the reply, or with no messages when
            `reply` is `None`.

        Raises:
            RuntimeError: When `fail` is set.

        """
        if self.hang:
            await asyncio.sleep(_HANG_S)
        if self.fail:
            msg = "sub-agent broke"
            raise RuntimeError(msg)
        if self.reply is None:
            return {"messages": []}
        return {"messages": [AIMessage(content=self.reply)]}


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
