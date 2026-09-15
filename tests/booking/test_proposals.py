"""Tests for `ProposalStore`, the human-in-the-loop proposal mechanism.

These tests are DB-independent: `ProposalStore.create()`, `dismiss()`, and
the pending-lookup/TTL/supersession machinery never touch the database, and
`confirm()`'s dispatch to `write_ops` is exercised here with the
`write_ops.commit_booking`/`cancel_booking`/`modify_booking`/
`find_booking_for_rooms` functions monkeypatched out, since their actual
correctness against a live database is covered separately by
`tests/booking/test_write_ops.py` (`db_integration`). What is under test here
is the store's own contract: single confirm-use, ownership, supersession,
invalidation-on-new-turn, TTL expiry, the pricing-mismatch refusal, and
the reconciliation read that settles a lost commit ack (step 18).
"""
# ruff: noqa: S101

from __future__ import annotations

import asyncio
import logging
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
_PROPOSALS_LOGGER = "blue_horizon.agents.booking.proposals"

# confirm() forwards write_pool to a write_ops function monkeypatched out in
# every test below, so its value is never actually inspected; typed as None
# to keep call sites honest that no real pool is used.
_UNUSED_WRITE_POOL = cast("AsyncConnectionPool[Any]", None)


class _FakeCommitResult:
    """Stand-in for a `write_ops` result carrying only what `confirm()` reads.

    Attributes:
        booking_id: Booking the result refers to, read by the audit log line.
        total_amount: Charged or new total, as a real result carries.
        refunded_amount: Refund amount, as a real cancel result carries.

    """

    def __init__(
        self,
        *,
        total_amount: Decimal | None = None,
        refunded_amount: Decimal | None = None,
        booking_id: int = 1,
    ) -> None:
        """Store whichever total field the calling test cares about.

        Args:
            total_amount: Value returned by `getattr(result, "total_amount")`.
            refunded_amount: Value returned by `getattr(result, "refunded_amount")`.
            booking_id: Value logged when the proposal is confirmed.

        """
        self.booking_id = booking_id
        if total_amount is not None:
            self.total_amount = total_amount
        if refunded_amount is not None:
            self.refunded_amount = refunded_amount


async def _fake_find_booking_for_rooms_returns_none(
    write_pool: AsyncConnectionPool[Any],
    *,
    customer_id: int,
    rooms: Sequence[write_ops.RoomRequest],
) -> _FakeCommitResult | None:
    """Stand in for a reconciliation read that finds no matching booking.

    The default fake for most `BookingUnavailableError` tests below: it
    simulates the common case where nothing was actually committed, so
    `confirm()` should behave exactly as it did before step 18 introduced
    the reconciliation read.

    Args:
        write_pool: Unused; `confirm()` forwards its own `write_pool`
            argument (`None` in these tests) without inspecting it.
        customer_id: Unused; this fake always reports nothing found.
        rooms: Unused; this fake always reports nothing found.

    Returns:
        None, unconditionally.

    """
    _ = write_pool, customer_id, rooms
    return None


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
            expected_total: Decimal | None = None,
        ) -> _FakeCommitResult:
            """Record the call and return a fixed result.

            Args:
                write_pool: Unused; `confirm()` forwards its own `write_pool`
                    argument (`None` in these tests) without inspecting it.
                customer_id: Forwarded by `confirm()`, recorded for assertion.
                rooms: Forwarded by `confirm()`, recorded for assertion.
                expected_total: Forwarded by `confirm()`, recorded for assertion.

            Returns:
                The fixed `fake_result`.

            """
            _ = write_pool
            calls.append(
                {
                    "customer_id": customer_id,
                    "rooms": rooms,
                    "expected_total": expected_total,
                },
            )
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
        assert calls == [
            {
                "customer_id": _CUSTOMER_ID,
                "rooms": [],
                "expected_total": Decimal("100.00"),
            },
        ]

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
            expected_total: Decimal | None = None,
        ) -> _FakeCommitResult:
            """Record the call and return a fixed result.

            Args:
                write_pool: Unused; `confirm()` forwards its own `write_pool`
                    argument (`None` in these tests) without inspecting it.
                customer_id: Forwarded by `confirm()`, recorded for assertion.
                booking_id: Forwarded by `confirm()`, recorded for assertion.
                rooms: Forwarded by `confirm()`, recorded for assertion.
                expected_total: Forwarded by `confirm()`, recorded for assertion.

            Returns:
                The fixed `fake_result`.

            """
            _ = write_pool
            calls.append(
                {
                    "customer_id": customer_id,
                    "booking_id": booking_id,
                    "rooms": rooms,
                    "expected_total": expected_total,
                },
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
            {
                "customer_id": _CUSTOMER_ID,
                "booking_id": 123,
                "rooms": None,
                "expected_total": Decimal("50.00"),
            },
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
            expected_total: Decimal | None = None,
        ) -> _FakeCommitResult:
            """Record the call and return a fixed result.

            Args:
                write_pool: Unused; `confirm()` forwards its own `write_pool`
                    argument (`None` in these tests) without inspecting it.
                customer_id: Forwarded by `confirm()`, recorded for assertion.
                booking_id: Forwarded by `confirm()`, recorded for assertion.
                changes: Forwarded by `confirm()`, recorded for assertion.
                expected_total: Forwarded by `confirm()`, recorded for assertion.

            Returns:
                The fixed `fake_result`.

            """
            _ = write_pool
            calls.append(
                {
                    "customer_id": customer_id,
                    "booking_id": booking_id,
                    "changes": changes,
                    "expected_total": expected_total,
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
            {
                "customer_id": _CUSTOMER_ID,
                "booking_id": 456,
                "changes": changes,
                "expected_total": Decimal("75.00"),
            },
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
            expected_total: Decimal | None = None,
        ) -> _FakeCommitResult:
            """Count the call and return a fixed result.

            Args:
                write_pool: Unused; `confirm()` forwards its own `write_pool`
                    argument (`None` in these tests) without inspecting it.
                customer_id: Unused; only the call count matters here.
                rooms: Unused; only the call count matters here.
                expected_total: Unused; the total the real write checks before COMMIT.

            Returns:
                The fixed `fake_result`.

            """
            _ = write_pool, customer_id, rooms, expected_total
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

    def test_booking_write_error_retires_proposal_and_second_confirm_404s(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A deterministic refusal retires the proposal; a retry then 404s.

        `BookingWriteError` means the write was evaluated and refused (for
        example, the nights were taken in the meantime), so nothing about a
        second attempt could change the outcome.
        """
        store = _make_store()
        proposal = _create_book_proposal(store, total="100.00")

        async def fake_commit_booking(
            write_pool: AsyncConnectionPool[Any],
            *,
            customer_id: int,
            rooms: Sequence[write_ops.RoomRequest],
            expected_total: Decimal | None = None,
        ) -> _FakeCommitResult:
            """Refuse unconditionally, simulating nights taken in the meantime."""
            _ = write_pool, customer_id, rooms, expected_total
            msg = "Room 101 is not available for every night requested."
            raise proposals_module.write_ops.BookingWriteError(msg)

        monkeypatch.setattr(
            proposals_module.write_ops, "commit_booking", fake_commit_booking,
        )

        with pytest.raises(proposals_module.write_ops.BookingWriteError):
            asyncio.run(
                store.confirm(
                    proposal_id=proposal.proposal_id,
                    customer_id=_CUSTOMER_ID,
                    write_pool=_UNUSED_WRITE_POOL,
                ),
            )

        assert store.get_pending_for_thread(_THREAD_ID) is None
        with pytest.raises(ProposalNotFoundError):
            asyncio.run(
                store.confirm(
                    proposal_id=proposal.proposal_id,
                    customer_id=_CUSTOMER_ID,
                    write_pool=_UNUSED_WRITE_POOL,
                ),
            )

    def test_booking_unavailable_error_leaves_proposal_pending_for_retry(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A database-unreachable failure leaves the proposal pending.

        Unlike `BookingWriteError`, `BookingUnavailableError` means the
        write was never evaluated, so the guest must be able to press
        Confirm again on the same proposal rather than seeing "That request
        has expired." The reconciliation read (step 18) finds nothing here,
        since nothing was actually committed, so it changes nothing about
        this outcome.
        """
        store = _make_store()
        proposal = _create_book_proposal(store, total="100.00")

        async def fake_commit_booking(
            write_pool: AsyncConnectionPool[Any],
            *,
            customer_id: int,
            rooms: Sequence[write_ops.RoomRequest],
            expected_total: Decimal | None = None,
        ) -> _FakeCommitResult:
            """Simulate a dead pool on the first call only."""
            _ = write_pool, customer_id, rooms, expected_total
            msg = "Could not reach the database to complete this request."
            raise proposals_module.write_ops.BookingUnavailableError(msg)

        monkeypatch.setattr(
            proposals_module.write_ops, "commit_booking", fake_commit_booking,
        )
        monkeypatch.setattr(
            proposals_module.write_ops,
            "find_booking_for_rooms",
            _fake_find_booking_for_rooms_returns_none,
        )

        with pytest.raises(proposals_module.write_ops.BookingUnavailableError):
            asyncio.run(
                store.confirm(
                    proposal_id=proposal.proposal_id,
                    customer_id=_CUSTOMER_ID,
                    write_pool=_UNUSED_WRITE_POOL,
                ),
            )

        assert store.get_pending_for_thread(_THREAD_ID) is proposal

    def test_reconciliation_finds_the_commit_and_reports_success(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The lost-ack case: the commit landed, only the ack was lost.

        `commit_booking` raises `BookingUnavailableError` (the connection
        died after COMMIT but before the ack returned), but
        `find_booking_for_rooms` then finds the guest's own booking. That
        must report success and cache it, not leave the guest thinking the
        room was lost when they actually have it -- the exact contradiction
        step 18 exists to close.
        """
        store = _make_store()
        proposal = _create_book_proposal(store, total="100.00")
        reconciled_result = _FakeCommitResult(total_amount=Decimal("100.00"))

        async def fake_commit_booking(
            write_pool: AsyncConnectionPool[Any],
            *,
            customer_id: int,
            rooms: Sequence[write_ops.RoomRequest],
            expected_total: Decimal | None = None,
        ) -> _FakeCommitResult:
            """Simulate the ack being lost after a commit that landed."""
            _ = write_pool, customer_id, rooms, expected_total
            msg = "Could not reach the database to complete this request."
            raise proposals_module.write_ops.BookingUnavailableError(msg)

        find_calls: list[dict[str, Any]] = []

        async def fake_find_booking_for_rooms(
            write_pool: AsyncConnectionPool[Any],
            *,
            customer_id: int,
            rooms: Sequence[write_ops.RoomRequest],
        ) -> _FakeCommitResult:
            """Report the lost commit as found, recording the call."""
            _ = write_pool
            find_calls.append({"customer_id": customer_id, "rooms": rooms})
            return reconciled_result

        monkeypatch.setattr(
            proposals_module.write_ops, "commit_booking", fake_commit_booking,
        )
        monkeypatch.setattr(
            proposals_module.write_ops,
            "find_booking_for_rooms",
            fake_find_booking_for_rooms,
        )

        outcome = asyncio.run(
            store.confirm(
                proposal_id=proposal.proposal_id,
                customer_id=_CUSTOMER_ID,
                write_pool=_UNUSED_WRITE_POOL,
            ),
        )

        assert outcome.already_confirmed is False
        assert outcome.result is reconciled_result
        assert find_calls == [{"customer_id": _CUSTOMER_ID, "rooms": []}]
        # Retired and cached exactly as a normal successful commit would be,
        # so a second confirm replays it instead of re-attempting the write.
        assert store.get_pending_for_thread(_THREAD_ID) is None
        replay = asyncio.run(
            store.confirm(
                proposal_id=proposal.proposal_id,
                customer_id=_CUSTOMER_ID,
                write_pool=_UNUSED_WRITE_POOL,
            ),
        )
        assert replay.already_confirmed is True
        assert replay.result is reconciled_result

    def test_reconciliation_read_failure_leaves_proposal_pending(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A reconciliation read that cannot reach the database is not a guess.

        The read itself failing must not be read as "the commit did not
        land": that would turn one failure (the commit's) into a second,
        different-shaped one (a false negative). The proposal stays pending
        either way, exactly as if no reconciliation had been attempted.
        """
        store = _make_store()
        proposal = _create_book_proposal(store, total="100.00")

        async def fake_commit_booking(
            write_pool: AsyncConnectionPool[Any],
            *,
            customer_id: int,
            rooms: Sequence[write_ops.RoomRequest],
            expected_total: Decimal | None = None,
        ) -> _FakeCommitResult:
            """Simulate a dead pool."""
            _ = write_pool, customer_id, rooms, expected_total
            msg = "Could not reach the database to complete this request."
            raise proposals_module.write_ops.BookingUnavailableError(msg)

        async def fake_find_booking_for_rooms(
            write_pool: AsyncConnectionPool[Any],
            *,
            customer_id: int,
            rooms: Sequence[write_ops.RoomRequest],
        ) -> _FakeCommitResult | None:
            """Simulate the reconciliation read itself being unreachable."""
            _ = write_pool, customer_id, rooms
            msg = "Could not reach the database to complete this request."
            raise proposals_module.write_ops.BookingUnavailableError(msg)

        monkeypatch.setattr(
            proposals_module.write_ops, "commit_booking", fake_commit_booking,
        )
        monkeypatch.setattr(
            proposals_module.write_ops,
            "find_booking_for_rooms",
            fake_find_booking_for_rooms,
        )

        with pytest.raises(proposals_module.write_ops.BookingUnavailableError):
            asyncio.run(
                store.confirm(
                    proposal_id=proposal.proposal_id,
                    customer_id=_CUSTOMER_ID,
                    write_pool=_UNUSED_WRITE_POOL,
                ),
            )

        assert store.get_pending_for_thread(_THREAD_ID) is proposal

    def test_reconciliation_is_skipped_for_cancel_proposals(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """`cancel`/`modify` proposals never call `find_booking_for_rooms`.

        `booking_rooms_no_overlap`'s uniqueness, the property that makes the
        reconciliation read exact rather than heuristic, is specific to a
        fresh `book`. `cancel` and `modify` already describe their own
        outcome through `BookingWriteError` (already cancelled, unknown
        `booking_room_id`), so step 18 scopes the read to `action == "book"`.
        """
        store = _make_store()
        proposal = store.create(
            thread_id=_THREAD_ID,
            customer_id=_CUSTOMER_ID,
            action="cancel",
            summary={"rooms": [], "total": "50.00"},
            details=(123, None),
        )

        async def fake_cancel_booking(
            write_pool: AsyncConnectionPool[Any],
            *,
            customer_id: int,
            booking_id: int,
            rooms: Sequence[write_ops.CancelRoomInstruction] | None,
            expected_total: Decimal | None = None,
        ) -> _FakeCommitResult:
            """Simulate a dead pool."""
            _ = write_pool, customer_id, booking_id, rooms, expected_total
            msg = "Could not reach the database to complete this request."
            raise proposals_module.write_ops.BookingUnavailableError(msg)

        find_calls: list[object] = []

        async def fake_find_booking_for_rooms(
            write_pool: AsyncConnectionPool[Any],
            *,
            customer_id: int,
            rooms: Sequence[write_ops.RoomRequest],
        ) -> _FakeCommitResult | None:
            """Record whether it was called at all; must never be for cancel."""
            _ = write_pool, customer_id, rooms
            find_calls.append(None)
            return None

        monkeypatch.setattr(
            proposals_module.write_ops, "cancel_booking", fake_cancel_booking,
        )
        monkeypatch.setattr(
            proposals_module.write_ops,
            "find_booking_for_rooms",
            fake_find_booking_for_rooms,
        )

        with pytest.raises(proposals_module.write_ops.BookingUnavailableError):
            asyncio.run(
                store.confirm(
                    proposal_id=proposal.proposal_id,
                    customer_id=_CUSTOMER_ID,
                    write_pool=_UNUSED_WRITE_POOL,
                ),
            )

        assert find_calls == []
        assert store.get_pending_for_thread(_THREAD_ID) is proposal

    def test_retry_after_unavailable_error_succeeds_and_caches(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A retry that lands after an unavailable failure commits and caches.

        Re-attempting `_commit_proposal` on the retained-pending proposal is
        what makes invariant 6 (double-booking is structurally impossible)
        the thing keeping this retry safe: the second call re-prices under
        `FOR UPDATE` exactly as a first attempt would, it is never skipped.
        """
        store = _make_store()
        proposal = _create_book_proposal(store, total="100.00")
        fake_result = _FakeCommitResult(total_amount=Decimal("100.00"))
        call_count = 0

        async def flaky_commit_booking(
            write_pool: AsyncConnectionPool[Any],
            *,
            customer_id: int,
            rooms: Sequence[write_ops.RoomRequest],
            expected_total: Decimal | None = None,
        ) -> _FakeCommitResult:
            """Fail with BookingUnavailableError once, then succeed."""
            _ = write_pool, customer_id, rooms, expected_total
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                msg = "Could not reach the database to complete this request."
                raise proposals_module.write_ops.BookingUnavailableError(msg)
            return fake_result

        monkeypatch.setattr(
            proposals_module.write_ops, "commit_booking", flaky_commit_booking,
        )
        monkeypatch.setattr(
            proposals_module.write_ops,
            "find_booking_for_rooms",
            _fake_find_booking_for_rooms_returns_none,
        )

        with pytest.raises(proposals_module.write_ops.BookingUnavailableError):
            asyncio.run(
                store.confirm(
                    proposal_id=proposal.proposal_id,
                    customer_id=_CUSTOMER_ID,
                    write_pool=_UNUSED_WRITE_POOL,
                ),
            )

        outcome = asyncio.run(
            store.confirm(
                proposal_id=proposal.proposal_id,
                customer_id=_CUSTOMER_ID,
                write_pool=_UNUSED_WRITE_POOL,
            ),
        )

        assert call_count == 2  # noqa: PLR2004
        assert outcome.already_confirmed is False
        assert outcome.result is fake_result
        assert store.get_pending_for_thread(_THREAD_ID) is None

    def test_retained_pending_proposal_dies_on_invalidate_thread(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A new turn on the thread still kills a retry-pending proposal.

        Retention across `BookingUnavailableError` is bounded by the
        guest's next utterance, exactly like any other pending proposal:
        `manager.ainvoke*` calls `invalidate_thread` at the start of every
        turn.
        """
        store = _make_store()
        proposal = _create_book_proposal(store, total="100.00")

        async def fake_commit_booking(
            write_pool: AsyncConnectionPool[Any],
            *,
            customer_id: int,
            rooms: Sequence[write_ops.RoomRequest],
            expected_total: Decimal | None = None,
        ) -> _FakeCommitResult:
            """Simulate a dead pool."""
            _ = write_pool, customer_id, rooms, expected_total
            msg = "Could not reach the database to complete this request."
            raise proposals_module.write_ops.BookingUnavailableError(msg)

        monkeypatch.setattr(
            proposals_module.write_ops, "commit_booking", fake_commit_booking,
        )
        monkeypatch.setattr(
            proposals_module.write_ops,
            "find_booking_for_rooms",
            _fake_find_booking_for_rooms_returns_none,
        )

        with pytest.raises(proposals_module.write_ops.BookingUnavailableError):
            asyncio.run(
                store.confirm(
                    proposal_id=proposal.proposal_id,
                    customer_id=_CUSTOMER_ID,
                    write_pool=_UNUSED_WRITE_POOL,
                ),
            )

        store.invalidate_thread(_THREAD_ID)

        assert store.get_pending_for_thread(_THREAD_ID) is None
        with pytest.raises(ProposalNotFoundError):
            asyncio.run(
                store.confirm(
                    proposal_id=proposal.proposal_id,
                    customer_id=_CUSTOMER_ID,
                    write_pool=_UNUSED_WRITE_POOL,
                ),
            )

    def test_retained_pending_proposal_still_expires_on_ttl(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A retry-pending proposal is not exempt from the TTL purge.

        Only a *confirmed* proposal (present in `_results`) is exempt from
        `_purge_expired` -- see its docstring. A proposal left pending after
        `BookingUnavailableError` was never confirmed, so it still expires.
        """
        store = _make_store(ttl_s=0.05)
        proposal = _create_book_proposal(store, total="100.00")

        async def fake_commit_booking(
            write_pool: AsyncConnectionPool[Any],
            *,
            customer_id: int,
            rooms: Sequence[write_ops.RoomRequest],
            expected_total: Decimal | None = None,
        ) -> _FakeCommitResult:
            """Simulate a dead pool."""
            _ = write_pool, customer_id, rooms, expected_total
            msg = "Could not reach the database to complete this request."
            raise proposals_module.write_ops.BookingUnavailableError(msg)

        monkeypatch.setattr(
            proposals_module.write_ops, "commit_booking", fake_commit_booking,
        )
        monkeypatch.setattr(
            proposals_module.write_ops,
            "find_booking_for_rooms",
            _fake_find_booking_for_rooms_returns_none,
        )

        with pytest.raises(proposals_module.write_ops.BookingUnavailableError):
            asyncio.run(
                store.confirm(
                    proposal_id=proposal.proposal_id,
                    customer_id=_CUSTOMER_ID,
                    write_pool=_UNUSED_WRITE_POOL,
                ),
            )

        time.sleep(0.1)

        assert store.get_pending_for_thread(_THREAD_ID) is None

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
            expected_total: Decimal | None = None,
        ) -> _FakeCommitResult:
            """Return a fixed successful result, ignoring all arguments.

            Args:
                write_pool: Unused; `confirm()` forwards its own `write_pool`
                    argument (`None` in these tests) without inspecting it.
                customer_id: Unused; the test only cares about the result.
                rooms: Unused; the test only cares about the result.
                expected_total: Unused; the total the real write checks before COMMIT.

            Returns:
                A fixed result matching the proposal's total.

            """
            _ = write_pool, customer_id, rooms, expected_total
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

    def test_pricing_mismatch_retires_proposal_and_logs_an_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A write refused for a pricing mismatch retires the proposal loudly.

        The write raises `PricingMismatchError` before it commits, so nothing
        was charged. Prices are fixed, so only a bug can cause it: the
        proposal is retired like any other refusal, but the line is an error
        rather than the info line an ordinary refusal gets.
        """
        store = _make_store()
        proposal = _create_book_proposal(store, total="100.00")

        async def refuse_on_mismatch(
            write_pool: AsyncConnectionPool[Any],
            *,
            customer_id: int,
            rooms: Sequence[write_ops.RoomRequest],
            expected_total: Decimal | None = None,
        ) -> _FakeCommitResult:
            """Refuse as the real write does when its total disagrees.

            Args:
                write_pool: Unused; `confirm()` forwards its own `write_pool`
                    argument (`None` in these tests) without inspecting it.
                customer_id: Unused; the test only cares about the mismatch.
                rooms: Unused; the test only cares about the mismatch.
                expected_total: Unused; the test only cares about the mismatch.

            Raises:
                PricingMismatchError: Unconditionally.

            """
            _ = write_pool, customer_id, rooms, expected_total
            raise proposals_module.write_ops.PricingMismatchError(
                expected_total=Decimal("100.00"), computed_total=Decimal("999.00"),
            )

        monkeypatch.setattr(
            proposals_module.write_ops, "commit_booking", refuse_on_mismatch,
        )

        with (
            caplog.at_level(logging.INFO, logger=_PROPOSALS_LOGGER),
            pytest.raises(proposals_module.write_ops.PricingMismatchError),
        ):
            asyncio.run(
                store.confirm(
                    proposal_id=proposal.proposal_id,
                    customer_id=_CUSTOMER_ID,
                    write_pool=_UNUSED_WRITE_POOL,
                ),
            )

        assert store.get_pending_for_thread(_THREAD_ID) is None
        records = [r for r in caplog.records if "pricing mismatch" in r.getMessage()]
        assert len(records) == 1
        assert records[0].levelno == logging.ERROR
        assert "computed_total=999.00" in records[0].getMessage()


class TestAuditLogging:
    """Each proposal lifecycle event logs one self-contained line."""

    def test_create_logs_the_proposal(self, caplog: pytest.LogCaptureFixture) -> None:
        """A created proposal is logged with its id, action, thread, and guest."""
        store = _make_store()
        with caplog.at_level(logging.INFO, logger=_PROPOSALS_LOGGER):
            proposal = _create_book_proposal(store)
        message = _only_message(caplog, "Proposal created")
        assert f"proposal_id={proposal.proposal_id}" in message
        assert f"thread_id={_THREAD_ID}" in message
        assert f"customer_id={_CUSTOMER_ID}" in message

    def test_confirm_logs_the_booking_it_wrote(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A confirmed proposal's line names the booking the write produced."""
        store = _make_store()
        proposal = _create_book_proposal(store, total="100.00")
        fake_result = _FakeCommitResult(total_amount=Decimal("100.00"), booking_id=42)

        async def fake_commit_booking(
            write_pool: AsyncConnectionPool[Any],
            *,
            customer_id: int,
            rooms: Sequence[write_ops.RoomRequest],
            expected_total: Decimal | None = None,
        ) -> _FakeCommitResult:
            """Return a fixed successful result, ignoring all arguments."""
            _ = write_pool, customer_id, rooms, expected_total
            return fake_result

        monkeypatch.setattr(
            proposals_module.write_ops, "commit_booking", fake_commit_booking,
        )

        with caplog.at_level(logging.INFO, logger=_PROPOSALS_LOGGER):
            asyncio.run(
                store.confirm(
                    proposal_id=proposal.proposal_id,
                    customer_id=_CUSTOMER_ID,
                    write_pool=_UNUSED_WRITE_POOL,
                ),
            )

        message = _only_message(caplog, "Proposal confirmed")
        assert f"proposal_id={proposal.proposal_id}" in message
        assert "booking_id=42" in message
        assert "duration_ms=" in message
        (record,) = [
            r for r in caplog.records if r.getMessage().startswith("Proposal confirmed")
        ]
        # Attributes, so a shipped line can be grouped and aggregated.
        assert record.__dict__["action"] == "book"
        assert isinstance(record.__dict__["duration_ms"], int)

    def test_wrong_guest_is_logged_as_a_warning(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Another guest's attempt on a proposal is a warning naming both guests."""
        store = _make_store()
        proposal = _create_book_proposal(store, customer_id=_CUSTOMER_ID)
        with (
            caplog.at_level(logging.INFO, logger=_PROPOSALS_LOGGER),
            pytest.raises(ProposalOwnershipError),
        ):
            store.dismiss(
                proposal_id=proposal.proposal_id, customer_id=_OTHER_CUSTOMER_ID,
            )
        records = [r for r in caplog.records if "different guest" in r.getMessage()]
        assert len(records) == 1
        assert records[0].levelno == logging.WARNING
        assert f"customer_id={_OTHER_CUSTOMER_ID} attempted" in records[0].getMessage()


def _only_message(caplog: pytest.LogCaptureFixture, prefix: str) -> str:
    """Return the one captured message starting with `prefix`.

    Args:
        caplog: Pytest log capture fixture.
        prefix: Start of the message to find.

    Returns:
        str: The formatted message.

    """
    messages = [
        r.getMessage() for r in caplog.records if r.getMessage().startswith(prefix)
    ]
    assert len(messages) == 1
    return messages[0]
