"""Tests for OrchestrationManager.

Unit tests cover only the pure-logic branches that do not require a running
LangGraph agent: the not-ready guard in ``ainvoke_stream`` and the stage
deduplication logic driven by ``_NODE_TO_STAGE``.
"""

from __future__ import annotations

# ruff: noqa: S101
import asyncio
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage

from blue_horizon.agents.orchestration.manager import OrchestrationManager

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_manager(
    *,
    agent: object = None,
    unavailable: str = "Unavailable.",
) -> OrchestrationManager:
    """Create an OrchestrationManager with minimal mocked internals.

    Bypasses ``__init__`` so that no real resources are created.

    Args:
        agent: Value to assign to the ``_agent`` slot.  ``None`` simulates
            the not-ready state; a MagicMock simulates a compiled graph.
        unavailable: Text returned by ``get_unavailable_message()``.

    Returns:
        Configured OrchestrationManager instance.

    """
    manager = OrchestrationManager.__new__(OrchestrationManager)
    manager._agent = agent  # type: ignore[assignment]  # noqa: SLF001
    manager._llm_semaphore = asyncio.Semaphore(1)  # noqa: SLF001
    mock_resources = MagicMock()
    mock_resources.get_config.return_value.messages.unavailable = unavailable
    manager._resources = mock_resources  # noqa: SLF001
    return manager


async def _collect(gen: AsyncGenerator[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drain an async generator into a list.

    Args:
        gen: Async generator to exhaust.

    Returns:
        All yielded items collected into a list.

    """
    return [item async for item in gen]


async def _async_events(
    events: list[dict[str, Any]],
) -> AsyncGenerator[dict[str, Any]]:
    """Yield each event dict in order as an async generator.

    Args:
        events: Sequence of event dicts to yield.

    Yields:
        Each event dict from the input list.

    """
    for e in events:
        yield e


def _chain_start(node: str) -> dict[str, Any]:
    """Build a minimal ``on_chain_start`` event for a named graph node.

    Args:
        node: The ``langgraph_node`` metadata value.

    Returns:
        Event dict matching the shape filtered by ``ainvoke_stream``.

    """
    return {"event": "on_chain_start", "metadata": {"langgraph_node": node}}


def _mock_agent(
    events: list[dict[str, Any]],
    *,
    ai_response: str = "Reply.",
) -> MagicMock:
    """Build a mock compiled graph for ``ainvoke_stream`` tests.

    Args:
        events: Raw LangGraph events to yield from ``astream_events``.
        ai_response: Plain-string content for the final AI message returned
            by ``aget_state``.

    Returns:
        MagicMock configured to simulate a ``CompiledStateGraph``.

    """
    mock = MagicMock()
    mock.astream_events.return_value = _async_events(events)
    snapshot = MagicMock()
    snapshot.values = {"messages": [AIMessage(content=ai_response)]}
    mock.aget_state = AsyncMock(return_value=snapshot)
    return mock


# ---------------------------------------------------------------------------
# ainvoke_stream — not-ready guard
# ---------------------------------------------------------------------------


class TestAinvokeStreamNotReady:
    """ainvoke_stream yields a single done event when the agent is not ready."""

    def test_yields_single_done_event(self) -> None:
        """Only one event is emitted: a done event."""
        manager = _make_manager(agent=None)
        events = asyncio.run(
            _collect(manager.ainvoke_stream(thread_id="t1", user_text="hi")),
        )
        assert len(events) == 1
        assert events[0]["type"] == "done"

    def test_done_event_contains_unavailable_message(self) -> None:
        """The done event's response carries the configured unavailable message."""
        manager = _make_manager(agent=None, unavailable="Sorry, not ready.")
        events = asyncio.run(
            _collect(manager.ainvoke_stream(thread_id="t1", user_text="hi")),
        )
        assert events[0]["response"] == "Sorry, not ready."

    def test_no_stage_events_emitted(self) -> None:
        """No stage events are emitted before the done event."""
        manager = _make_manager(agent=None)
        events = asyncio.run(
            _collect(manager.ainvoke_stream(thread_id="t1", user_text="hi")),
        )
        assert all(e["type"] != "stage" for e in events)


# ---------------------------------------------------------------------------
# ainvoke_stream — stage event emission and deduplication
# ---------------------------------------------------------------------------


class TestAinvokeStreamStages:
    """ainvoke_stream emits deduplicated stage events then a done event."""

    def test_known_node_emits_stage_event(self) -> None:
        """An on_chain_start for a known node emits one stage event."""
        manager = _make_manager(agent=_mock_agent([_chain_start("router")]))
        events = asyncio.run(
            _collect(manager.ainvoke_stream(thread_id="t1", user_text="hi")),
        )
        stage_events = [e for e in events if e["type"] == "stage"]
        assert len(stage_events) == 1
        assert stage_events[0]["label"] == "Routing your request\u2026"

    def test_unknown_node_emits_no_stage_event(self) -> None:
        """An on_chain_start for an unrecognised node is silently ignored."""
        manager = _make_manager(agent=_mock_agent([_chain_start("finalize")]))
        events = asyncio.run(
            _collect(manager.ainvoke_stream(thread_id="t1", user_text="hi")),
        )
        stage_events = [e for e in events if e["type"] == "stage"]
        assert len(stage_events) == 0

    def test_shared_stage_key_nodes_emit_single_event(self) -> None:
        """query_faq/amenities/services share a stage key and emit only once."""
        manager = _make_manager(
            agent=_mock_agent([
                _chain_start("query_faq"),
                _chain_start("query_amenities"),
                _chain_start("query_services"),
            ]),
        )
        events = asyncio.run(
            _collect(manager.ainvoke_stream(thread_id="t1", user_text="hi")),
        )
        stage_events = [e for e in events if e["type"] == "stage"]
        assert len(stage_events) == 1
        assert stage_events[0]["label"] == "Searching hotel information\u2026"

    def test_info_and_query_nodes_share_stage_key(self) -> None:
        """The outer 'info' dispatch node and query_* nodes share a stage key."""
        manager = _make_manager(
            agent=_mock_agent([
                _chain_start("info"),
                _chain_start("query_faq"),
            ]),
        )
        events = asyncio.run(
            _collect(manager.ainvoke_stream(thread_id="t1", user_text="hi")),
        )
        stage_events = [e for e in events if e["type"] == "stage"]
        assert len(stage_events) == 1

    def test_non_chain_start_events_produce_no_stage(self) -> None:
        """Events with types other than on_chain_start are ignored."""
        meta = {"langgraph_node": "router"}
        manager = _make_manager(
            agent=_mock_agent([
                {"event": "on_chat_model_stream", "metadata": meta},
                {"event": "on_chain_end", "metadata": meta},
            ]),
        )
        events = asyncio.run(
            _collect(manager.ainvoke_stream(thread_id="t1", user_text="hi")),
        )
        stage_events = [e for e in events if e["type"] == "stage"]
        assert len(stage_events) == 0

    def test_done_event_contains_final_ai_response(self) -> None:
        """The done event carries the last AI message from the graph state."""
        manager = _make_manager(agent=_mock_agent([], ai_response="Final answer."))
        events = asyncio.run(
            _collect(manager.ainvoke_stream(thread_id="t1", user_text="hi")),
        )
        done_events = [e for e in events if e["type"] == "done"]
        assert len(done_events) == 1
        assert done_events[0]["response"] == "Final answer."

    def test_done_event_is_last(self) -> None:
        """The done event is always the final event in the stream."""
        manager = _make_manager(
            agent=_mock_agent([_chain_start("router"), _chain_start("rooms")]),
        )
        events = asyncio.run(
            _collect(manager.ainvoke_stream(thread_id="t1", user_text="hi")),
        )
        assert events[-1]["type"] == "done"

    def test_full_info_path_stage_sequence(self) -> None:
        """All info-path nodes produce deduplicated stages in order."""
        manager = _make_manager(
            agent=_mock_agent([
                _chain_start("router"),
                _chain_start("info"),
                _chain_start("parse"),
                _chain_start("query_faq"),
                _chain_start("query_amenities"),
                _chain_start("query_services"),
                _chain_start("rerank"),
                _chain_start("respond"),
            ]),
        )
        events = asyncio.run(
            _collect(manager.ainvoke_stream(thread_id="t1", user_text="hi")),
        )
        stage_labels = [e["label"] for e in events if e["type"] == "stage"]
        # routing, search (once), parse, rerank, respond = 5 unique stage keys
        assert len(stage_labels) == 5  # noqa: PLR2004
        assert stage_labels[0] == "Routing your request\u2026"
        assert stage_labels[-1] == "Generating response\u2026"
