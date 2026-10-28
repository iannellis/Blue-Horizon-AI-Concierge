"""Tests for the per-guest room-night limit's counting rule.

DB-independent: `_require_room_night_allowance` runs one aggregate query and
decides from its row, so a fake cursor that returns a fixed row exercises
the decision. The query itself, the advisory lock, and the writes that call
both are covered against a real database in `tests/booking/test_write_ops.py`.
"""
# ruff: noqa: S101

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, cast

import pytest

from blue_horizon.agents.booking import write_ops

if TYPE_CHECKING:
    import psycopg

_CUSTOMER_ID = 7
_BOOKING_ID = 42
_LIMIT = write_ops.RoomNightLimit(max_room_nights=60, window_hours=24)


class _FakeCursor:
    """Cursor stand-in that records queries and returns one fixed row.

    Attributes:
        row: Row returned by `fetchone`.
        executed: Parameters of every query run, in order.

    """

    def __init__(self, *, other_nights: int, booking_in_window: bool) -> None:
        """Store the row the window query will return.

        Args:
            other_nights: Room-nights on the guest's other bookings in the window.
            booking_in_window: Whether the booking being written was made in
                the window.

        """
        self.row = {
            "other_nights": other_nights,
            "booking_in_window": booking_in_window,
        }
        self.executed: list[dict[str, Any]] = []

    async def execute(self, query: str, params: dict[str, Any]) -> None:
        """Record a query's parameters.

        Args:
            query: SQL text; unused.
            params: Bound parameters, recorded.

        """
        _ = query
        self.executed.append(params)

    async def fetchone(self) -> dict[str, Any]:
        """Return the fixed row.

        Returns:
            The row given at construction.

        """
        return self.row


def _check(
    cur: _FakeCursor,
    *,
    booking_id: int | None,
    nights_before: int,
    nights_after: int,
) -> None:
    """Run the allowance check against a fake cursor.

    Args:
        cur: Fake cursor holding the window query's row.
        booking_id: Booking being written, or ``None`` for a new one.
        nights_before: Room-nights the booking held before the write.
        nights_after: Room-nights the booking holds after the write.

    """
    asyncio.run(
        write_ops._require_room_night_allowance(  # noqa: SLF001
            cast("psycopg.AsyncCursor[dict[str, Any]]", cur),
            customer_id=_CUSTOMER_ID,
            limit=_LIMIT,
            booking_id=booking_id,
            nights_before=nights_before,
            nights_after=nights_after,
        ),
    )


class TestNewBooking:
    """A new booking adds all its room-nights to the guest's window total."""

    def test_reaching_the_limit_exactly_passes(self) -> None:
        """The limit is inclusive."""
        cur = _FakeCursor(other_nights=50, booking_in_window=False)
        _check(cur, booking_id=None, nights_before=0, nights_after=10)
        assert cur.executed[0]["window_hours"] == _LIMIT.window_hours

    def test_one_past_the_limit_is_refused_with_the_total(self) -> None:
        """The refusal names the limit and the total the guest would reach."""
        cur = _FakeCursor(other_nights=50, booking_in_window=False)
        with pytest.raises(write_ops.BookingLimitError) as caught:
            _check(cur, booking_id=None, nights_before=0, nights_after=11)
        assert caught.value.total_room_nights == 61  # noqa: PLR2004
        assert caught.value.max_room_nights == _LIMIT.max_room_nights
        assert "61" in str(caught.value)


class TestModification:
    """A modification is checked only when it adds room-nights."""

    def test_change_adding_no_nights_skips_the_query(self) -> None:
        """A guest over the limit can still shorten or swap a stay."""
        cur = _FakeCursor(other_nights=500, booking_in_window=True)
        _check(cur, booking_id=_BOOKING_ID, nights_before=10, nights_after=10)
        _check(cur, booking_id=_BOOKING_ID, nights_before=10, nights_after=4)
        assert cur.executed == []

    def test_booking_made_in_the_window_counts_in_full(self) -> None:
        """Growing a recent booking counts its whole new size."""
        cur = _FakeCursor(other_nights=40, booking_in_window=True)
        with pytest.raises(write_ops.BookingLimitError) as caught:
            _check(cur, booking_id=_BOOKING_ID, nights_before=18, nights_after=21)
        assert caught.value.total_room_nights == 61  # noqa: PLR2004

    def test_booking_made_before_the_window_counts_only_the_growth(self) -> None:
        """A seeded guest's older booking adds only the nights the change adds."""
        cur = _FakeCursor(other_nights=40, booking_in_window=False)
        _check(cur, booking_id=_BOOKING_ID, nights_before=100, nights_after=120)
        with pytest.raises(write_ops.BookingLimitError) as caught:
            _check(cur, booking_id=_BOOKING_ID, nights_before=100, nights_after=121)
        assert caught.value.total_room_nights == 61  # noqa: PLR2004
