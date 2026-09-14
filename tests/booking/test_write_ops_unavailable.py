"""Tests for `write_ops`'s pool-acquisition failure handling.

No real database connection is required: `pool.connection()` is mocked to
raise directly, exercising `reraise_operational_as_unavailable` without a
live Postgres. See `test_write_ops.py` for the `db_integration`-marked tests
that exercise these same functions against a real database.
"""

# ruff: noqa: S101

import asyncio
import datetime as dt
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import psycopg
import pytest
from psycopg_pool import PoolTimeout

from blue_horizon.agents.booking import write_ops

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _pool_that_fails_to_connect(exc: BaseException) -> MagicMock:
    """Build a fake connection pool whose `.connection()` raises on entry.

    Simulates a failure at the pool-acquisition boundary itself (an
    exhausted pool, or a Neon compute that has not resumed yet): nothing
    inside the `async with pool.connection() as conn:` body ever runs.

    Args:
        exc: The exception `.connection().__aenter__()` should raise.

    Returns:
        A mock pool suitable for passing to a `write_ops` write function.

    """
    connection_cm = MagicMock()
    connection_cm.__aenter__ = AsyncMock(side_effect=exc)
    connection_cm.__aexit__ = AsyncMock(return_value=False)
    pool = MagicMock()
    pool.connection = MagicMock(return_value=connection_cm)
    return pool


def _pool_with_no_available_nights() -> MagicMock:
    """Build a fake pool whose cursor reports zero available nights.

    Connects successfully, opens a transaction and cursor successfully, but
    `fetchall()` always returns no rows, so `_price_one_room` raises
    `write_ops.BookingWriteError` before any write is attempted. Used to
    confirm `reraise_operational_as_unavailable` leaves that exception
    alone rather than converting it.

    Returns:
        A mock pool suitable for passing to a `write_ops` write function.

    """
    cursor = MagicMock()
    cursor.__aenter__ = AsyncMock(return_value=cursor)
    cursor.__aexit__ = AsyncMock(return_value=False)
    cursor.execute = AsyncMock()
    cursor.fetchall = AsyncMock(return_value=[])
    cursor.fetchone = AsyncMock(return_value=None)

    transaction_cm = MagicMock()
    transaction_cm.__aenter__ = AsyncMock(return_value=None)
    transaction_cm.__aexit__ = AsyncMock(return_value=False)

    conn = MagicMock()
    conn.__aenter__ = AsyncMock(return_value=conn)
    conn.__aexit__ = AsyncMock(return_value=False)
    conn.transaction = MagicMock(return_value=transaction_cm)
    conn.cursor = MagicMock(return_value=cursor)

    connection_cm = MagicMock()
    connection_cm.__aenter__ = AsyncMock(return_value=conn)
    connection_cm.__aexit__ = AsyncMock(return_value=False)
    pool = MagicMock()
    pool.connection = MagicMock(return_value=connection_cm)
    return pool


_ROOM_REQUEST = write_ops.RoomRequest(
    room_id=1,
    room_number=101,
    check_in=dt.date(2024, 1, 1),
    check_out=dt.date(2024, 1, 3),
)

# One case per write_ops write function, each hitting the pool-acquisition
# boundary before doing anything function-specific.
_WRITE_CALLS: dict[str, tuple[Any, dict[str, Any]]] = {
    "commit_booking": (
        write_ops.commit_booking,
        {"customer_id": 1, "rooms": [_ROOM_REQUEST]},
    ),
    "cancel_booking": (
        write_ops.cancel_booking,
        {"customer_id": 1, "booking_id": 1},
    ),
    "modify_booking": (
        write_ops.modify_booking,
        {"customer_id": 1, "booking_id": 1, "changes": []},
    ),
}


# ---------------------------------------------------------------------------
# Operational failures become BookingUnavailableError
# ---------------------------------------------------------------------------


class TestPoolAcquisitionFailureIsUnavailable:
    """Every write function reports a dead pool as `BookingUnavailableError`."""

    @pytest.mark.parametrize("write_name", list(_WRITE_CALLS))
    @pytest.mark.parametrize(
        "exc",
        [
            psycopg.OperationalError("connection timeout expired"),
            psycopg.InterfaceError("connection already closed"),
            PoolTimeout("couldn't get a connection in time"),
            TimeoutError("timed out"),
        ],
    )
    def test_operational_failure_raises_unavailable(
        self, write_name: str, exc: BaseException,
    ) -> None:
        """A connect-time failure of any recognised kind is reported uniformly."""
        write_fn, kwargs = _WRITE_CALLS[write_name]
        pool = _pool_that_fails_to_connect(exc)
        with pytest.raises(write_ops.BookingUnavailableError):
            asyncio.run(write_fn(pool, **kwargs))

    def test_unavailable_error_chains_the_original_exception(self) -> None:
        """The original driver exception is preserved as the cause, for logs."""
        original = psycopg.OperationalError("connection timeout expired")
        pool = _pool_that_fails_to_connect(original)
        with pytest.raises(write_ops.BookingUnavailableError) as exc_info:
            asyncio.run(
                write_ops.commit_booking(
                    pool, customer_id=1, rooms=[_ROOM_REQUEST],
                ),
            )
        assert exc_info.value.__cause__ is original


# ---------------------------------------------------------------------------
# BookingWriteError is never converted
# ---------------------------------------------------------------------------


class TestBookingWriteErrorNotSwallowed:
    """`reraise_operational_as_unavailable` leaves a real refusal alone."""

    def test_commit_booking_write_error_propagates_unchanged(self) -> None:
        """An ordinary refusal (no nights available) is not reported as unavailable."""
        pool = _pool_with_no_available_nights()
        with pytest.raises(write_ops.BookingWriteError) as exc_info:
            asyncio.run(
                write_ops.commit_booking(
                    pool, customer_id=1, rooms=[_ROOM_REQUEST],
                ),
            )
        assert "not available" in str(exc_info.value)

    def test_cancel_booking_write_error_propagates_unchanged(self) -> None:
        """A booking that does not exist raises `BookingWriteError`, not unavailable."""
        pool = _pool_with_no_available_nights()
        with pytest.raises(write_ops.BookingWriteError) as exc_info:
            asyncio.run(
                write_ops.cancel_booking(pool, customer_id=1, booking_id=1),
            )
        assert "does not exist" in str(exc_info.value)


# ---------------------------------------------------------------------------
# The reservations-panel reads share the same handling
# ---------------------------------------------------------------------------


class TestListReadsPoolAcquisitionFailure:
    """`list_bookings` and `list_customers` report a dead pool as unavailable.

    Without this, a network outage reached `/v1/bookings` and
    `/v1/customers` as an uncaught `PoolTimeout` and a 500.
    """

    def test_list_bookings_raises_unavailable(self) -> None:
        """A checkout timeout in `list_bookings` becomes `BookingUnavailableError`."""
        pool = _pool_that_fails_to_connect(PoolTimeout("couldn't get a connection"))
        with pytest.raises(write_ops.BookingUnavailableError):
            asyncio.run(write_ops.list_bookings(pool, customer_id=1))

    def test_list_customers_raises_unavailable(self) -> None:
        """A checkout timeout in `list_customers` becomes `BookingUnavailableError`."""
        pool = _pool_that_fails_to_connect(PoolTimeout("couldn't get a connection"))
        with pytest.raises(write_ops.BookingUnavailableError):
            asyncio.run(write_ops.list_customers(pool, seeded_customer_count=15))


# ---------------------------------------------------------------------------
# find_booking_for_rooms shares the same pool-acquisition failure handling
# ---------------------------------------------------------------------------


class TestFindBookingForRoomsPoolAcquisitionFailure:
    """The reconciliation read reports a dead pool the same way the writes do.

    So the confirm-path caller (`ProposalStore.confirm`) can tell "checked
    and found nothing" (a clean `None`) apart from "could not check" (this
    exception) -- collapsing the two would let a failed reconciliation read
    report a room as lost when nothing was actually established.
    """

    @pytest.mark.parametrize(
        "exc",
        [
            psycopg.OperationalError("connection timeout expired"),
            psycopg.InterfaceError("connection already closed"),
            PoolTimeout("couldn't get a connection in time"),
            TimeoutError("timed out"),
        ],
    )
    def test_operational_failure_raises_unavailable(self, exc: BaseException) -> None:
        """A connect-time failure of any recognised kind is reported uniformly."""
        pool = _pool_that_fails_to_connect(exc)
        with pytest.raises(write_ops.BookingUnavailableError):
            asyncio.run(
                write_ops.find_booking_for_rooms(
                    pool, customer_id=1, rooms=[_ROOM_REQUEST],
                ),
            )

    def test_empty_rooms_returns_none_without_touching_the_pool(self) -> None:
        """No rooms to reconcile is a trivial `None`, never a database round trip."""
        pool = _pool_that_fails_to_connect(
            psycopg.OperationalError("connection timeout expired"),
        )
        result = asyncio.run(
            write_ops.find_booking_for_rooms(pool, customer_id=1, rooms=[]),
        )
        assert result is None
