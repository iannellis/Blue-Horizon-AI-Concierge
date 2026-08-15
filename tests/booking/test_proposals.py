"""Tests for `ProposalStore`, the human-in-the-loop proposal mechanism.

These tests are DB-independent: `ProposalStore.create()`, `dismiss()`, and
the pending-lookup/TTL/supersession machinery never touch the database, and
`confirm()`'s dispatch to `write_ops` is exercised here with the
`write_ops.commit_booking`/`cancel_booking`/`modify_booking` functions
monkeypatched out, since their actual correctness against a live database is
covered separately by `tests/booking/test_write_ops.py` (`db_integration`).
What is under test here is the store's own contract: single confirm-use,
ownership, supersession, invalidation-on-new-turn, TTL expiry, and the
pricing-mismatch assertion.
"""
# ruff: noqa: S101

from __future__ import annotations

import asyncio
import time
from decimal import Decimal
from typing import TYPE_CHECKING, Any, cast

import pytest

from blue_horizon.agents.booking import proposals as proposals_module
from blue_horizon.agents.booking.proposals import (
    ProposalNotFoundError,
    ProposalOwnershipError,
    ProposalStore,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from psycopg_pool import AsyncConnectionPool

    from blue_horizon.agents.booking import write_ops

_THREAD_ID = "thread-1"
_OTHER_THREAD_ID = "thread-2"
_CUSTOMER_ID = 7
_OTHER_CUSTOMER_ID = 9
_DEFAULT_TTL_S = 1800.0

# confirm() forwards write_pool to a write_ops function monkeypatched out in
# every test below, so its value is never actually inspected; typed as None
# to keep call sites honest that no real pool is used.
_UNUSED_WRITE_POOL = cast("AsyncConnectionPool[Any]", None)


class _FakeCommitResult:
    """Stand-in for a `write_ops` result carrying only what `confirm()` reads.

    Attributes:
        total_amount: Charged/new total, read for `book`/`modify` proposals.
        refunded_amount: Refund amount, read for `cancel` proposals.

    """

    def __init__(
        self,
        *,
        total_amount: Decimal | None = None,
        refunded_amount: Decimal | None = None,
    ) -> None:
        """Store whichever total field the calling test cares about.

        Args:
            total_amount: Value returned by `getattr(result, "total_amount")`.
            refunded_amount: Value returned by `getattr(result, "refunded_amount")`.

        """
        if total_amount is not None:
            self.total_amount = total_amount
        if refunded_amount is not None:
            self.refunded_amount = refunded_amount


def _make_store(*, ttl_s: float = _DEFAULT_TTL_S) -> ProposalStore:
    """Build an empty `ProposalStore`.

    Args:
        ttl_s: TTL passed through to the store.

    Returns:
        A fresh, empty `ProposalStore`.

    """
    return ProposalStore(ttl_s=ttl_s)


def _create_book_proposal(
    store: ProposalStore,
    *,
    thread_id: str = _THREAD_ID,
    customer_id: int = _CUSTOMER_ID,
    total: str = "100.00",
) -> proposals_module.Proposal:
    """Create a `book` proposal with a minimal, valid summary/details pair.

    Args:
        store: Store to create the proposal in.
        thread_id: Conversation thread.
        customer_id: Guest the proposal belongs to.
        total: Fixed-2-decimal total string stored in `summary["total"]`.

    Returns:
        The newly created proposal.

    """
    return store.create(
        thread_id=thread_id,
        customer_id=customer_id,
        action="book",
        summary={"rooms": [], "total": total},
        details=[],
    )


class TestCreateAndLookup:
    """`create()` and `get_pending_for_thread()`."""

    def test_create_returns_proposal_with_expected_fields(self) -> None:
        """A created proposal round-trips its own fields."""
        store = _make_store()
        proposal = _create_book_proposal(store)
        assert proposal.thread_id == _THREAD_ID
        assert proposal.customer_id == _CUSTOMER_ID
        assert proposal.action == "book"
        assert proposal.summary["total"] == "100.00"
        assert proposal.proposal_id

    def test_get_pending_for_thread_returns_none_when_absent(self) -> None:
        """A thread with no proposal returns `None`, not an error."""
        store = _make_store()
        assert store.get_pending_for_thread(_THREAD_ID) is None

    def test_get_pending_for_thread_returns_created_proposal(self) -> None:
        """A just-created proposal is immediately visible as pending."""
        store = _make_store()
        proposal = _create_book_proposal(store)
        assert store.get_pending_for_thread(_THREAD_ID) == proposal

    def test_second_proposal_supersedes_first_on_same_thread(self) -> None:
        """Creating a second proposal on a thread retires the first."""
        store = _make_store()
        first = _create_book_proposal(store, total="100.00")
        second = _create_book_proposal(store, total="200.00")

        assert store.get_pending_for_thread(_THREAD_ID) == second
        with pytest.raises(ProposalNotFoundError):
            store.dismiss(proposal_id=first.proposal_id, customer_id=_CUSTOMER_ID)

    def test_proposals_on_different_threads_do_not_interfere(self) -> None:
        """Two threads each keep their own pending proposal independently."""
        store = _make_store()
        first = _create_book_proposal(store, thread_id=_THREAD_ID)
        second = _create_book_proposal(store, thread_id=_OTHER_THREAD_ID)

        assert store.get_pending_for_thread(_THREAD_ID) == first
        assert store.get_pending_for_thread(_OTHER_THREAD_ID) == second


class TestInvalidateThread:
    """`invalidate_thread()` -- called on every new user turn."""

    def test_invalidate_thread_clears_pending_proposal(self) -> None:
        """A pending proposal disappears once its thread is invalidated."""
        store = _make_store()
        proposal = _create_book_proposal(store)

        store.invalidate_thread(_THREAD_ID)

        assert store.get_pending_for_thread(_THREAD_ID) is None
        with pytest.raises(ProposalNotFoundError):
            store.dismiss(proposal_id=proposal.proposal_id, customer_id=_CUSTOMER_ID)

    def test_invalidate_thread_with_no_pending_proposal_is_a_no_op(self) -> None:
        """Invalidating an already-empty thread raises nothing."""
        store = _make_store()
        store.invalidate_thread(_THREAD_ID)
        assert store.get_pending_for_thread(_THREAD_ID) is None


class TestDismiss:
    """`dismiss()` -- the Cancel-button path."""

    def test_dismiss_pending_proposal_succeeds(self) -> None:
        """Dismissing a pending proposal returns it and clears it."""
        store = _make_store()
        proposal = _create_book_proposal(store)

        dismissed = store.dismiss(
            proposal_id=proposal.proposal_id, customer_id=_CUSTOMER_ID,
        )

        assert dismissed == proposal
        assert store.get_pending_for_thread(_THREAD_ID) is None

    def test_dismiss_unknown_proposal_raises_not_found(self) -> None:
        """An unrecognized proposal id raises `ProposalNotFoundError`."""
        store = _make_store()
        with pytest.raises(ProposalNotFoundError):
            store.dismiss(proposal_id="does-not-exist", customer_id=_CUSTOMER_ID)

    def test_dismiss_wrong_customer_raises_ownership_error(self) -> None:
        """Dismissing another guest's proposal raises `ProposalOwnershipError`."""
        store = _make_store()
        proposal = _create_book_proposal(store, customer_id=_CUSTOMER_ID)
        with pytest.raises(ProposalOwnershipError):
            store.dismiss(
                proposal_id=proposal.proposal_id, customer_id=_OTHER_CUSTOMER_ID,
            )

    def test_dismiss_twice_raises_not_found_on_second_call(self) -> None:
        """Dismiss is not idempotent: a repeat dismiss finds nothing pending."""
        store = _make_store()
        proposal = _create_book_proposal(store)
        store.dismiss(proposal_id=proposal.proposal_id, customer_id=_CUSTOMER_ID)
        with pytest.raises(ProposalNotFoundError):
            store.dismiss(proposal_id=proposal.proposal_id, customer_id=_CUSTOMER_ID)


class TestTtlExpiry:
    """TTL-bounded purge of never-confirmed proposals."""

    def test_expired_proposal_is_not_returned_as_pending(self) -> None:
        """A proposal older than the TTL stops being pending."""
        store = _make_store(ttl_s=0.05)
        _create_book_proposal(store)
        time.sleep(0.1)
        assert store.get_pending_for_thread(_THREAD_ID) is None

    def test_expired_proposal_cannot_be_dismissed(self) -> None:
        """A proposal older than the TTL raises `ProposalNotFoundError` on dismiss."""
        store = _make_store(ttl_s=0.05)
        proposal = _create_book_proposal(store)
        time.sleep(0.1)
        with pytest.raises(ProposalNotFoundError):
            store.dismiss(proposal_id=proposal.proposal_id, customer_id=_CUSTOMER_ID)


class TestConfirm:
    """`confirm()` -- single-use commit dispatch.

    No pytest-asyncio plugin is installed in this project, so each test wraps
    its coroutine in `asyncio.run(...)`, matching the existing convention in
    `tests/booking/test_db_utils.py` and `tests/orchestration/test_manager.py`.
    """

    def test_confirm_book_dispatches_to_commit_booking(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A `book` proposal's confirm calls `write_ops.commit_booking`."""
        store = _make_store()
        proposal = _create_book_proposal(store, total="100.00")
        fake_result = _FakeCommitResult(total_amount=Decimal("100.00"))
        calls: list[dict[str, Any]] = []

        async def fake_commit_booking(
            write_pool: AsyncConnectionPool[Any],
            *,
            customer_id: int,
            rooms: Sequence[write_ops.RoomRequest],
        ) -> _FakeCommitResult:
            """Record the call and return a fixed result.

            Args:
                write_pool: Unused; `confirm()` forwards its own `write_pool`
                    argument (`None` in these tests) without inspecting it.
                customer_id: Forwarded by `confirm()`, recorded for assertion.
                rooms: Forwarded by `confirm()`, recorded for assertion.

            Returns:
                The fixed `fake_result`.

            """
            _ = write_pool
            calls.append({"customer_id": customer_id, "rooms": rooms})
            return fake_result

        monkeypatch.setattr(
            proposals_module.write_ops, "commit_booking", fake_commit_booking,
        )

        outcome = asyncio.run(
            store.confirm(
                proposal_id=proposal.proposal_id,
                customer_id=_CUSTOMER_ID,
                write_pool=_UNUSED_WRITE_POOL,
            ),
        )

        assert outcome.result is fake_result
        assert outcome.already_confirmed is False
        assert calls == [{"customer_id": _CUSTOMER_ID, "rooms": []}]

    def test_confirm_cancel_dispatches_to_cancel_booking(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A `cancel` proposal's confirm calls `write_ops.cancel_booking`."""
        store = _make_store()
        proposal = store.create(
            thread_id=_THREAD_ID,
            customer_id=_CUSTOMER_ID,
            action="cancel",
            summary={"rooms": [], "total": "50.00"},
            details=(123, None),
        )
        fake_result = _FakeCommitResult(refunded_amount=Decimal("50.00"))
        calls: list[dict[str, Any]] = []

        async def fake_cancel_booking(
            write_pool: AsyncConnectionPool[Any],
            *,
            customer_id: int,
            booking_id: int,
            rooms: Sequence[write_ops.CancelRoomInstruction] | None,
        ) -> _FakeCommitResult:
            """Record the call and return a fixed result.

            Args:
                write_pool: Unused; `confirm()` forwards its own `write_pool`
                    argument (`None` in these tests) without inspecting it.
                customer_id: Forwarded by `confirm()`, recorded for assertion.
                booking_id: Forwarded by `confirm()`, recorded for assertion.
                rooms: Forwarded by `confirm()`, recorded for assertion.

            Returns:
                The fixed `fake_result`.

            """
            _ = write_pool
            calls.append(
                {"customer_id": customer_id, "booking_id": booking_id, "rooms": rooms},
            )
            return fake_result

        monkeypatch.setattr(
            proposals_module.write_ops, "cancel_booking", fake_cancel_booking,
        )

        outcome = asyncio.run(
            store.confirm(
                proposal_id=proposal.proposal_id,
                customer_id=_CUSTOMER_ID,
                write_pool=_UNUSED_WRITE_POOL,
            ),
        )

        assert outcome.result is fake_result
        assert calls == [
            {"customer_id": _CUSTOMER_ID, "booking_id": 123, "rooms": None},
        ]

    def test_confirm_modify_dispatches_to_modify_booking(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A `modify` proposal's confirm calls `write_ops.modify_booking`."""
        store = _make_store()
        # A bare string stands in for a ModifyRoomInstruction: confirm()'s
        # dispatch only forwards `details` opaquely to the monkeypatched
        # fake_modify_booking below, never inspecting its structure.
        changes = cast("list[write_ops.ModifyRoomInstruction]", ["fake-instruction"])
        proposal = store.create(
            thread_id=_THREAD_ID,
            customer_id=_CUSTOMER_ID,
            action="modify",
            summary={"changes": [], "total": "75.00"},
            details=(456, changes),
        )
        fake_result = _FakeCommitResult(total_amount=Decimal("75.00"))
        calls: list[dict[str, Any]] = []

        async def fake_modify_booking(
            write_pool: AsyncConnectionPool[Any],
            *,
            customer_id: int,
            booking_id: int,
            changes: Sequence[write_ops.ModifyRoomInstruction],
        ) -> _FakeCommitResult:
            """Record the call and return a fixed result.

            Args:
                write_pool: Unused; `confirm()` forwards its own `write_pool`
                    argument (`None` in these tests) without inspecting it.
                customer_id: Forwarded by `confirm()`, recorded for assertion.
                booking_id: Forwarded by `confirm()`, recorded for assertion.
                changes: Forwarded by `confirm()`, recorded for assertion.

            Returns:
                The fixed `fake_result`.

            """
            _ = write_pool
            calls.append(
                {
                    "customer_id": customer_id,
                    "booking_id": booking_id,
                    "changes": changes,
                },
            )
            return fake_result

        monkeypatch.setattr(
            proposals_module.write_ops, "modify_booking", fake_modify_booking,
        )

        outcome = asyncio.run(
            store.confirm(
                proposal_id=proposal.proposal_id,
                customer_id=_CUSTOMER_ID,
                write_pool=_UNUSED_WRITE_POOL,
            ),
        )

        assert outcome.result is fake_result
        assert calls == [
            {"customer_id": _CUSTOMER_ID, "booking_id": 456, "changes": changes},
        ]

    def test_second_confirm_replays_cached_result_without_recommitting(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A double confirm of the same proposal is a no-op, not a double-write.

        This is the guarantee that stops a double-clicked Confirm button from
        double-booking.
        """
        store = _make_store()
        proposal = _create_book_proposal(store, total="100.00")
        fake_result = _FakeCommitResult(total_amount=Decimal("100.00"))
        call_count = 0

        async def fake_commit_booking(
            write_pool: AsyncConnectionPool[Any],
            *,
            customer_id: int,
            rooms: Sequence[write_ops.RoomRequest],
        ) -> _FakeCommitResult:
            """Count the call and return a fixed result.

            Args:
                write_pool: Unused; `confirm()` forwards its own `write_pool`
                    argument (`None` in these tests) without inspecting it.
                customer_id: Unused; only the call count matters here.
                rooms: Unused; only the call count matters here.

            Returns:
                The fixed `fake_result`.

            """
            _ = write_pool, customer_id, rooms
            nonlocal call_count
            call_count += 1
            return fake_result

        monkeypatch.setattr(
            proposals_module.write_ops, "commit_booking", fake_commit_booking,
        )

        async def _confirm_twice() -> tuple[Any, Any]:
            first = await store.confirm(
                proposal_id=proposal.proposal_id,
                customer_id=_CUSTOMER_ID,
                write_pool=_UNUSED_WRITE_POOL,
            )
            second = await store.confirm(
                proposal_id=proposal.proposal_id,
                customer_id=_CUSTOMER_ID,
                write_pool=_UNUSED_WRITE_POOL,
            )
            return first, second

        first, second = asyncio.run(_confirm_twice())

        assert first.already_confirmed is False
        assert second.already_confirmed is True
        assert second.result is first.result
        assert call_count == 1

    def test_confirm_unknown_proposal_raises_not_found(self) -> None:
        """Confirming an unrecognized proposal id raises `ProposalNotFoundError`."""
        store = _make_store()
        with pytest.raises(ProposalNotFoundError):
            asyncio.run(
                store.confirm(
                    proposal_id="does-not-exist",
                    customer_id=_CUSTOMER_ID,
                    write_pool=_UNUSED_WRITE_POOL,
                ),
            )

    def test_confirm_wrong_customer_raises_ownership_error(self) -> None:
        """Confirming another guest's pending proposal raises an ownership error."""
        store = _make_store()
        proposal = _create_book_proposal(store, customer_id=_CUSTOMER_ID)
        with pytest.raises(ProposalOwnershipError):
            asyncio.run(
                store.confirm(
                    proposal_id=proposal.proposal_id,
                    customer_id=_OTHER_CUSTOMER_ID,
                    write_pool=_UNUSED_WRITE_POOL,
                ),
            )

    def test_confirm_wrong_customer_on_already_confirmed_proposal_raises(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Ownership is re-checked on the cached-result replay path too."""
        store = _make_store()
        proposal = _create_book_proposal(
            store, customer_id=_CUSTOMER_ID, total="100.00",
        )

        async def fake_commit_booking(
            write_pool: AsyncConnectionPool[Any],
            *,
            customer_id: int,
            rooms: Sequence[write_ops.RoomRequest],
        ) -> _FakeCommitResult:
            """Return a fixed successful result, ignoring all arguments.

            Args:
                write_pool: Unused; `confirm()` forwards its own `write_pool`
                    argument (`None` in these tests) without inspecting it.
                customer_id: Unused; the test only cares about the result.
                rooms: Unused; the test only cares about the result.

            Returns:
                A fixed result matching the proposal's total.

            """
            _ = write_pool, customer_id, rooms
            return _FakeCommitResult(total_amount=Decimal("100.00"))

        monkeypatch.setattr(
            proposals_module.write_ops, "commit_booking", fake_commit_booking,
        )

        async def _confirm_then_confirm_as_other() -> None:
            await store.confirm(
                proposal_id=proposal.proposal_id,
                customer_id=_CUSTOMER_ID,
                write_pool=_UNUSED_WRITE_POOL,
            )
            await store.confirm(
                proposal_id=proposal.proposal_id,
                customer_id=_OTHER_CUSTOMER_ID,
                write_pool=_UNUSED_WRITE_POOL,
            )

        with pytest.raises(ProposalOwnershipError):
            asyncio.run(_confirm_then_confirm_as_other())

    def test_confirm_pricing_mismatch_raises_assertion_error(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A commit total that disagrees with the dialog's total is a fatal bug.

        Prices are fixed and never rewritten, so this can only fire on a bug
        in this codebase -- exactly the case where a guest would otherwise be
        charged a number the dialog never showed.
        """
        store = _make_store()
        proposal = _create_book_proposal(store, total="100.00")

        async def fake_commit_booking(
            write_pool: AsyncConnectionPool[Any],
            *,
            customer_id: int,
            rooms: Sequence[write_ops.RoomRequest],
        ) -> _FakeCommitResult:
            """Return a result whose total deliberately disagrees with the proposal.

            Args:
                write_pool: Unused; `confirm()` forwards its own `write_pool`
                    argument (`None` in these tests) without inspecting it.
                customer_id: Unused; the test only cares about the mismatch.
                rooms: Unused; the test only cares about the mismatch.

            Returns:
                A result whose total does not match the proposal's total.

            """
            _ = write_pool, customer_id, rooms
            return _FakeCommitResult(total_amount=Decimal("999.00"))

        monkeypatch.setattr(
            proposals_module.write_ops, "commit_booking", fake_commit_booking,
        )

        with pytest.raises(AssertionError):
            asyncio.run(
                store.confirm(
                    proposal_id=proposal.proposal_id,
                    customer_id=_CUSTOMER_ID,
                    write_pool=_UNUSED_WRITE_POOL,
                ),
            )
