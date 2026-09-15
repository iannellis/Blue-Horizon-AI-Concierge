"""Tests for OrchestrationManager.

Unit tests cover only the pure-logic branches that do not require a running
LangGraph agent: the not-ready guard in ``ainvoke``/``ainvoke_stream``, the
stage deduplication logic driven by ``_NODE_TO_STAGE``, and -- driving
``_init_loop`` for real, against a stubbed ``startup_check`` -- the
readiness classification the failure-handling rework depends on.
"""

from __future__ import annotations

# ruff: noqa: S101
import asyncio
import logging
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage

from blue_horizon.agents.exceptions import ConfigurationError, OperationalError
from blue_horizon.agents.orchestration.manager import OrchestrationManager, Readiness

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable

# Long enough that a passing test never hits it under normal machine load,
# short enough that a hung test (the loop wedged, or a readiness transition
# that never happens) fails fast instead of joining the suite's stalled
# tests.
_POLL_TIMEOUT_S = 2.0
_POLL_INTERVAL_S = 0.005


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_manager(
    *,
    agent: object = None,
    unavailable: str = "Unavailable.",
) -> OrchestrationManager:
    """Create an OrchestrationManager with minimal mocked internals.

    Bypasses ``__init__`` so that no real resources are created. Suitable
    for tests that call ``ainvoke``/``ainvoke_stream`` directly; it never
    starts ``_init_loop``, so it is not suitable for readiness-transition
    tests -- see ``_make_uninitialized_manager`` for those.

    Args:
        agent: Value to assign to the ``_agent`` slot.  ``None`` simulates
            the not-ready state; a MagicMock simulates a compiled graph.
        unavailable: Text returned by ``get_readiness_message()`` while
            ``STARTING``.

    Returns:
        Configured OrchestrationManager instance.

    """
    manager = OrchestrationManager.__new__(OrchestrationManager)
    manager._agent = agent  # type: ignore[assignment]  # noqa: SLF001
    manager._readiness = (  # noqa: SLF001
        Readiness.READY if agent is not None else Readiness.STARTING
    )
    manager._llm_semaphore = asyncio.Semaphore(1)  # noqa: SLF001
    manager._thread_customers = {}  # noqa: SLF001
    mock_resources = MagicMock()
    mock_resources.config.messages.unavailable = unavailable
    mock_resources.config.messages.error = "Could not complete."
    mock_resources.config.messages.database_unavailable = "Database unreachable."
    # No proposal pending by default, so stage tests don't see a spurious
    # "proposal" event mixed into their stage/done assertions.
    booking_resources = mock_resources.booking_resources
    booking_resources.proposals.get_pending_for_thread.return_value = None
    manager._resources = mock_resources  # noqa: SLF001
    return manager


def _make_uninitialized_manager(
    *,
    startup_check_side_effect: object,
    init_retry_base_s: float = 0.001,
    init_retry_max_s: float = 0.001,
) -> OrchestrationManager:
    """Build a manager that drives the real ``_init_loop``.

    Bypasses ``__init__`` (no real resources), but -- unlike
    ``_make_manager`` -- sets up every field ``start()``/``_init_loop``
    touch: ``_lock``, ``_stop_event``, and a stubbed async
    ``startup_check()``. Backoff defaults to near-zero so a multi-attempt
    test does not sit through real exponential delays.

    Args:
        startup_check_side_effect: Passed straight to
            ``AsyncMock(side_effect=...)`` for
            ``resources.startup_check``: an exception (or exception class)
            to raise every call, or a list consumed one entry per call
            (each entry either an exception to raise or ``None`` to
            succeed).
        init_retry_base_s: Backoff base passed through the mocked config.
        init_retry_max_s: Backoff cap passed through the mocked config.

    Returns:
        An OrchestrationManager ready for ``await manager.start()``.

    """
    manager = OrchestrationManager.__new__(OrchestrationManager)
    manager._agent = None  # noqa: SLF001
    manager._readiness = Readiness.STARTING  # noqa: SLF001
    manager._llm_semaphore = asyncio.Semaphore(1)  # noqa: SLF001
    manager._thread_customers = {}  # noqa: SLF001
    manager._init_task = None  # noqa: SLF001
    manager._lock = asyncio.Lock()  # noqa: SLF001
    manager._stop_event = asyncio.Event()  # noqa: SLF001

    mock_resources = MagicMock()
    mock_resources.startup_check = AsyncMock(side_effect=startup_check_side_effect)
    mock_resources.aclose = AsyncMock()
    mock_resources.config.orchestration.init_retry_base_s = init_retry_base_s
    mock_resources.config.orchestration.init_retry_max_s = init_retry_max_s
    manager._resources = mock_resources  # noqa: SLF001
    return manager


async def _wait_until(condition: Callable[[], bool]) -> None:
    """Poll ``condition()`` until it is truthy, or fail after a timeout.

    Args:
        condition: Zero-argument callable checked every `_POLL_INTERVAL_S`.

    Raises:
        AssertionError: If `condition()` is never truthy within
            `_POLL_TIMEOUT_S`.

    """
    elapsed = 0.0
    while elapsed < _POLL_TIMEOUT_S:
        if condition():
            return
        await asyncio.sleep(_POLL_INTERVAL_S)
        elapsed += _POLL_INTERVAL_S
    pytest.fail(f"condition not met within {_POLL_TIMEOUT_S}s")


async def _collect(gen: AsyncGenerator[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drain an async generator into a list.

    Args:
        gen: Async generator to exhaust.

    Returns:
        All yielded items collected into a list.

    """
    return [item async for item in gen]


async def _run_stream(manager: OrchestrationManager) -> list[dict[str, Any]]:
    """Drive ``ainvoke_stream`` with fixed thread/customer test identity.

    Args:
        manager: Manager under test.

    Returns:
        All yielded stream events collected into a list.

    """
    gen = manager.ainvoke_stream(thread_id="t1", user_text="hi", customer_id=1)
    return await _collect(gen)


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
    turn_error: str | None = None,
) -> MagicMock:
    """Build a mock compiled graph for ``ainvoke_stream`` tests.

    Appends a synthetic ``on_chain_end`` event for the ``finalize`` node,
    since ``ainvoke_stream`` reads its final response from that event's own
    output rather than a separate ``aget_state()`` re-read (a post-hoc read
    can return a different concurrent request's reply on the same
    ``thread_id`` -- see ``OrchestrationManager.ainvoke_stream``'s docstring).

    Args:
        events: Raw LangGraph events to yield from ``astream_events``, before
            the synthetic finalize event.
        ai_response: Plain-string content for the final AI message carried
            by the synthetic finalize event's output.
        turn_error: When set, the finalize output instead reports a failed
            turn with this code and no messages, as ``finalize_node`` does.

    Returns:
        MagicMock configured to simulate a ``CompiledStateGraph``.

    """
    mock = MagicMock()
    output: dict[str, Any] = {"messages": [AIMessage(content=ai_response)]}
    if turn_error is not None:
        output = {"messages": [], "turn_error": turn_error}
    finalize_event = {
        "event": "on_chain_end",
        "metadata": {"langgraph_node": "finalize"},
        "data": {"output": output},
    }
    mock.astream_events.return_value = _async_events([*events, finalize_event])
    return mock


# ---------------------------------------------------------------------------
# ainvoke / ainvoke_stream — not-ready guard
#
# A not-ready manager used to paint over the gap with a fake `done` event
# carrying the "unavailable" message -- delivered as HTTP 200, so it landed
# in the transcript as something the concierge said. That path is removed:
# both methods now require an agent and raise if there is not one, and
# `api.app.chat`'s readiness gate is what actually keeps a not-ready
# manager from reaching either method in production (see api/app.py).
# ---------------------------------------------------------------------------


class TestAinvokeRequiresReady:
    """`ainvoke` raises rather than fabricating a reply when not ready."""

    def test_raises_when_agent_is_none(self) -> None:
        """A not-ready manager raises instead of returning a fake reply."""
        manager = _make_manager(agent=None)
        with pytest.raises(RuntimeError, match="Orchestration agent"):
            asyncio.run(
                manager.ainvoke(thread_id="t1", user_text="hi", customer_id=1),
            )


class TestAinvokeStreamRequiresReady:
    """`ainvoke_stream` raises rather than yielding a fake done event."""

    def test_raises_when_agent_is_none(self) -> None:
        """A not-ready manager raises instead of yielding any event."""
        manager = _make_manager(agent=None)
        with pytest.raises(RuntimeError, match="Orchestration agent"):
            asyncio.run(_run_stream(manager))

    def test_yields_no_events_before_raising(self) -> None:
        """Draining the generator by hand confirms it is empty, not just short."""
        manager = _make_manager(agent=None)
        gen = manager.ainvoke_stream(thread_id="t1", user_text="hi", customer_id=1)

        async def _first_item() -> dict[str, Any]:
            return await gen.__anext__()

        with pytest.raises(RuntimeError):
            asyncio.run(_first_item())


# ---------------------------------------------------------------------------
# ainvoke_stream — stage event emission and deduplication
# ---------------------------------------------------------------------------


class TestAinvokeStreamStages:
    """ainvoke_stream emits deduplicated stage events then a done event."""

    def test_known_node_emits_stage_event(self) -> None:
        """An on_chain_start for a known node emits one stage event."""
        manager = _make_manager(agent=_mock_agent([_chain_start("router")]))
        events = asyncio.run(_run_stream(manager))
        stage_events = [e for e in events if e["type"] == "stage"]
        assert len(stage_events) == 1
        assert stage_events[0]["label"] == "Routing your request\u2026"

    def test_unknown_node_emits_no_stage_event(self) -> None:
        """An on_chain_start for an unrecognised node is silently ignored."""
        manager = _make_manager(agent=_mock_agent([_chain_start("finalize")]))
        events = asyncio.run(_run_stream(manager))
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
        events = asyncio.run(_run_stream(manager))
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
        events = asyncio.run(_run_stream(manager))
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
        events = asyncio.run(_run_stream(manager))
        stage_events = [e for e in events if e["type"] == "stage"]
        assert len(stage_events) == 0

    def test_done_event_contains_final_ai_response(self) -> None:
        """The done event carries the last AI message from the graph state."""
        manager = _make_manager(agent=_mock_agent([], ai_response="Final answer."))
        events = asyncio.run(_run_stream(manager))
        done_events = [e for e in events if e["type"] == "done"]
        assert len(done_events) == 1
        assert done_events[0]["response"] == "Final answer."

    def test_done_event_is_last(self) -> None:
        """The done event is always the final event in the stream."""
        manager = _make_manager(
            agent=_mock_agent([_chain_start("router"), _chain_start("booking")]),
        )
        events = asyncio.run(_run_stream(manager))
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
                _chain_start("merge"),
                _chain_start("respond"),
            ]),
        )
        events = asyncio.run(_run_stream(manager))
        stage_labels = [e["label"] for e in events if e["type"] == "stage"]
        # routing, search (once), parse, merge, respond = 5 unique stage keys
        assert len(stage_labels) == 5  # noqa: PLR2004
        assert stage_labels[0] == "Routing your request\u2026"
        assert stage_labels[-1] == "Generating response\u2026"


# ---------------------------------------------------------------------------
# _init_loop — readiness classification
#
# Drives the real _init_loop against a stubbed startup_check(), which is
# also, incidentally, the first coverage this loop has had at all: nothing
# previously touched _init_loop, startup_check's retry, or stop_never, so
# the unbounded-retry behavior the archived-branch design in the
# failure-handling plan depends on was unpinned before this.
# ---------------------------------------------------------------------------


class TestInitLoopReadinessClassification:
    """`_init_loop` sets `readiness` from the exception `startup_check` raises."""

    def test_configuration_error_sets_failed(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A permanent misconfiguration is reported as FAILED, not STARTING."""
        monkeypatch.setattr(
            "blue_horizon.agents.orchestration.manager.build_orchestration_agent",
            MagicMock(),
        )
        manager = _make_uninitialized_manager(
            startup_check_side_effect=ConfigurationError("bad role"),
        )

        async def _run() -> None:
            await manager.start()
            try:
                await _wait_until(lambda: manager.readiness is Readiness.FAILED)
                assert manager.is_ready is False
            finally:
                await manager.stop()

        asyncio.run(_run())

    def test_operational_error_sets_starting(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A transient dependency outage is reported as STARTING, not FAILED."""
        monkeypatch.setattr(
            "blue_horizon.agents.orchestration.manager.build_orchestration_agent",
            MagicMock(),
        )
        manager = _make_uninitialized_manager(
            startup_check_side_effect=OperationalError("db unreachable"),
        )

        async def _run() -> None:
            await manager.start()
            try:
                # Already STARTING at construction; give the loop a few
                # failed attempts to prove it is *staying* STARTING, not
                # merely starting there before its first attempt runs.
                await asyncio.sleep(0.05)
                assert manager.readiness is Readiness.STARTING
                assert manager.is_ready is False
            finally:
                await manager.stop()

        asyncio.run(_run())

    def test_unclassified_error_defaults_to_starting(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An exception type this module does not recognise still degrades safely.

        A misclassification should degrade to bad copy (STARTING), never to
        a guest being told a possibly-transient failure will never recover.
        """
        monkeypatch.setattr(
            "blue_horizon.agents.orchestration.manager.build_orchestration_agent",
            MagicMock(),
        )
        manager = _make_uninitialized_manager(
            startup_check_side_effect=ValueError("unexpected bug"),
        )

        async def _run() -> None:
            await manager.start()
            try:
                await asyncio.sleep(0.05)
                assert manager.readiness is Readiness.STARTING
            finally:
                await manager.stop()

        asyncio.run(_run())

    def test_failed_state_still_retries_and_can_recover(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """FAILED does not stop the loop: a later success still reaches READY.

        The succeeding attempt is held until the test has observed FAILED.
        With near-zero backoff the loop can otherwise pass through FAILED
        and reach READY between two polls, and READY never reverts, so the
        wait for FAILED would time out.
        """
        monkeypatch.setattr(
            "blue_horizon.agents.orchestration.manager.build_orchestration_agent",
            MagicMock(),
        )
        failures = iter([ConfigurationError("bad role") for _ in range(2)])
        release_success = asyncio.Event()

        async def _startup_check() -> None:
            """Raise each queued failure, then succeed once released.

            Raises:
                ConfigurationError: For each of the first two calls.

            """
            failure = next(failures, None)
            if failure is not None:
                raise failure
            await release_success.wait()

        manager = _make_uninitialized_manager(startup_check_side_effect=_startup_check)

        async def _run() -> None:
            await manager.start()
            try:
                await _wait_until(lambda: manager.readiness is Readiness.FAILED)
                release_success.set()
                await _wait_until(lambda: manager.is_ready)
                assert manager.readiness is Readiness.READY
            finally:
                # Never leave the loop blocked in startup_check on teardown.
                release_success.set()
                await manager.stop()

        asyncio.run(_run())


class TestInitLoopLogging:
    """A dependency that stays down logs its traceback once, not every retry."""

    def test_repeated_failure_logs_traceback_only_once(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Identical consecutive failures carry `exc_info` on the first log only."""
        monkeypatch.setattr(
            "blue_horizon.agents.orchestration.manager.build_orchestration_agent",
            MagicMock(),
        )
        failures = 3
        manager = _make_uninitialized_manager(
            startup_check_side_effect=[
                OperationalError("db unreachable") for _ in range(failures)
            ]
            + [None],
        )
        caplog.set_level(
            logging.WARNING, logger="blue_horizon.agents.orchestration.manager",
        )

        async def _run() -> None:
            await manager.start()
            try:
                await _wait_until(lambda: manager.is_ready)
            finally:
                await manager.stop()

        asyncio.run(_run())

        records = [r for r in caplog.records if "Initialization failed" in r.message]
        assert len(records) == failures
        assert records[0].exc_info is not None
        assert all(r.exc_info is None for r in records[1:])

    def test_changed_failure_logs_traceback_again(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A different failure is logged in full even right after another."""
        monkeypatch.setattr(
            "blue_horizon.agents.orchestration.manager.build_orchestration_agent",
            MagicMock(),
        )
        manager = _make_uninitialized_manager(
            startup_check_side_effect=[
                OperationalError("db unreachable"),
                OperationalError("redis unreachable"),
                None,
            ],
        )
        caplog.set_level(
            logging.WARNING, logger="blue_horizon.agents.orchestration.manager",
        )

        async def _run() -> None:
            await manager.start()
            try:
                await _wait_until(lambda: manager.is_ready)
            finally:
                await manager.stop()

        asyncio.run(_run())

        records = [r for r in caplog.records if "Initialization failed" in r.message]
        assert [r.exc_info is not None for r in records] == [True, True]


class TestInitLoopSimulatedSlowStartup:
    """Substitute for an archived Neon branch, which cannot be produced on demand.

    The readiness state machine does not care *why* startup is slow, so
    driving `startup_check` to fail or block repeatedly exercises the same
    behavior a slow unarchive would.
    """

    def test_stays_starting_through_repeated_failures_then_recovers(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Readiness stays STARTING across many failed attempts, then recovers."""
        monkeypatch.setattr(
            "blue_horizon.agents.orchestration.manager.build_orchestration_agent",
            MagicMock(),
        )
        attempts = 20
        manager = _make_uninitialized_manager(
            startup_check_side_effect=[
                OperationalError("still resuming") for _ in range(attempts)
            ]
            + [None],
        )

        async def _run() -> None:
            await manager.start()
            try:
                # Poll across several retry cycles without ever observing
                # FAILED or READY too early -- the loop must keep retrying
                # on its own, with no caller intervention, per stop_never.
                for _ in range(attempts * 3):
                    assert manager.readiness in (Readiness.STARTING, Readiness.READY)
                    if manager.is_ready:
                        break
                    await asyncio.sleep(_POLL_INTERVAL_S)
                assert manager.is_ready is True
                startup_check = cast(
                    "AsyncMock", manager._resources.startup_check,  # noqa: SLF001
                )
                assert startup_check.await_count >= attempts + 1
            finally:
                await manager.stop()

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# Failed turns
#
# A turn that failed inside the graph arrives as a finalize output carrying
# `turn_error` and no reply. It must end the stream with an `error` event,
# never a `done` event a client would render as the concierge's answer.
# ---------------------------------------------------------------------------


class TestFailedTurn:
    """A failed turn is reported as an error, and its proposal cannot survive."""

    def test_stream_emits_error_event_instead_of_done(self) -> None:
        """The stream ends with an error event carrying the code and copy."""
        manager = _make_manager(
            agent=_mock_agent([_chain_start("router")], turn_error="timeout"),
        )
        events = asyncio.run(_run_stream(manager))
        assert [e["type"] for e in events] == ["stage", "error"]
        assert events[-1] == {
            "type": "error",
            "code": "timeout",
            "message": "Could not complete.",
        }

    def test_unavailable_turn_uses_database_copy(self) -> None:
        """A database outage carries its own copy, not the generic error."""
        manager = _make_manager(agent=_mock_agent([], turn_error="unavailable"))
        events = asyncio.run(_run_stream(manager))
        assert events == [
            {
                "type": "error",
                "code": "unavailable",
                "message": "Database unreachable.",
            },
        ]

    def test_stream_invalidates_proposal_and_emits_none(self) -> None:
        """A proposal left by a failed turn is invalidated, never surfaced."""
        manager = _make_manager(agent=_mock_agent([], turn_error="internal"))
        proposals = cast("MagicMock", manager.get_booking_resources().proposals)
        proposals.get_pending_for_thread.return_value = MagicMock()
        events = asyncio.run(_run_stream(manager))
        assert all(e["type"] != "proposal" for e in events)
        # Once as the turn starts, and once more for the failure.
        assert proposals.invalidate_thread.call_count == 2  # noqa: PLR2004

    def test_ainvoke_invalidates_proposal_of_failed_turn(self) -> None:
        """The non-streaming path invalidates a failed turn's proposal too."""
        agent = MagicMock()
        agent.ainvoke = AsyncMock(
            return_value={"messages": [], "turn_error": "internal"},
        )
        manager = _make_manager(agent=agent)
        proposals = cast("MagicMock", manager.get_booking_resources().proposals)
        result = asyncio.run(
            manager.ainvoke(thread_id="t1", user_text="hi", customer_id=1),
        )
        assert result["turn_error"] == "internal"
        assert proposals.invalidate_thread.call_count == 2  # noqa: PLR2004
