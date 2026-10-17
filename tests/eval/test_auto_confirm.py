"""Tests for the eval/stress harnesses' auto-confirm helper."""
# ruff: noqa: S101

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from blue_horizon.agents.booking import proposals as proposals_module
from blue_horizon.agents.booking import write_ops
from eval.langsmith_target._confirm import auto_confirm_pending_proposal

_THREAD_ID = "thread-1"
_REFUSAL = "Room 726 is not available for every night requested."


def _orchestration(confirm_error: Exception) -> MagicMock:
    """Build an orchestration double whose pending proposal fails to confirm.

    Args:
        confirm_error: Exception `ProposalStore.confirm` raises.

    Returns:
        MagicMock: Orchestration double with an async
        `append_assistant_message`.

    """
    proposals = MagicMock()
    proposals.get_pending_for_thread.return_value = SimpleNamespace(
        proposal_id="p1", action="book",
    )
    proposals.confirm = AsyncMock(side_effect=confirm_error)
    orchestration = MagicMock()
    orchestration.get_booking_resources.return_value = SimpleNamespace(
        proposals=proposals, get_write_pool=MagicMock(),
    )
    orchestration.append_assistant_message = AsyncMock()
    return orchestration


def _run(orchestration: MagicMock, callback: MagicMock) -> None:
    """Run `auto_confirm_pending_proposal` once for the test thread.

    Args:
        orchestration: Orchestration double.
        callback: Capture-callback double.

    """
    asyncio.run(
        auto_confirm_pending_proposal(
            orchestration,
            thread_id=_THREAD_ID,
            customer_id=1,
            callback=callback,
        ),
    )


class TestAutoConfirmRefusal:
    """A refused confirm is captured and recorded in the thread's history."""

    def test_refused_write_appends_refusal_note(self) -> None:
        """A `BookingWriteError` leaves an app-authored note on the thread."""
        orchestration = _orchestration(write_ops.BookingWriteError(_REFUSAL))
        callback = MagicMock()

        _run(orchestration, callback)

        callback.capture_confirm_error.assert_called_once_with(
            action="book", error=_REFUSAL,
        )
        orchestration.append_assistant_message.assert_awaited_once()
        note = orchestration.append_assistant_message.await_args.kwargs
        assert note["thread_id"] == _THREAD_ID
        assert note["text"].startswith("Not confirmed:")
        assert _REFUSAL in note["text"]

    def test_proposal_error_appends_nothing(self) -> None:
        """A store-level refusal (unknown or foreign proposal) writes no note."""
        orchestration = _orchestration(
            proposals_module.ProposalNotFoundError("expired"),
        )
        callback = MagicMock()

        _run(orchestration, callback)

        callback.capture_confirm_error.assert_called_once()
        orchestration.append_assistant_message.assert_not_awaited()
