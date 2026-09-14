"""Tests for the `propose_*` tools' database queries under a dead connection pool.

`create_agent` does not catch tool exceptions, so whatever these queries raise
escapes the booking agent, and the orchestration graph records `unavailable`
only for a `BookingUnavailableError`. Before these queries were wrapped, an
outage during a proposal escaped as a raw `PoolTimeout` and was reported as an
internal failure. No real database connection is required: `pool.connection()`
is mocked to raise directly.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest
from psycopg_pool import PoolTimeout

from blue_horizon.agents.booking import write_ops
from blue_horizon.agents.booking.factory import _refund_preview, _room_id_for_number

_ROOM_REQUEST = write_ops.RoomRequest(
    room_id=1,
    room_number=101,
    check_in=dt.date(2024, 1, 1),
    check_out=dt.date(2024, 1, 3),
)


def _pool_that_fails_to_connect() -> MagicMock:
    """Build a fake connection pool whose `.connection()` times out on entry.

    Returns:
        A mock pool whose checkout raises `PoolTimeout`.

    """
    connection_cm = MagicMock()
    connection_cm.__aenter__ = AsyncMock(
        side_effect=PoolTimeout("couldn't get a connection after 10.00 sec"),
    )
    connection_cm.__aexit__ = AsyncMock(return_value=False)
    pool = MagicMock()
    pool.connection = MagicMock(return_value=connection_cm)
    return pool


class TestPricingPreviewsRaiseUnavailable:
    """The `write_ops` pricing previews report a dead pool as unavailable."""

    def test_price_rooms(self) -> None:
        """`propose_booking`'s pricing read raises `BookingUnavailableError`."""
        with pytest.raises(write_ops.BookingUnavailableError):
            asyncio.run(
                write_ops.price_rooms(_pool_that_fails_to_connect(), [_ROOM_REQUEST]),
            )

    def test_price_modification_room(self) -> None:
        """`propose_modification`'s pricing read raises `BookingUnavailableError`."""
        with pytest.raises(write_ops.BookingUnavailableError):
            asyncio.run(
                write_ops.price_modification_room(
                    _pool_that_fails_to_connect(),
                    existing_room_id=1,
                    existing_check_in=dt.date(2024, 1, 1),
                    existing_check_out=dt.date(2024, 1, 3),
                    request=_ROOM_REQUEST,
                ),
            )


class TestFactoryQueriesRaiseUnavailable:
    """The booking factory's own queries report a dead pool as unavailable."""

    def test_room_id_for_number(self) -> None:
        """Resolving a room number raises `BookingUnavailableError`."""
        with pytest.raises(write_ops.BookingUnavailableError):
            asyncio.run(_room_id_for_number(_pool_that_fails_to_connect(), 101))

    def test_refund_preview(self) -> None:
        """A partial cancellation's refund preview raises `BookingUnavailableError`."""
        row = write_ops.RoomStaySummary(
            booking_room_id=1,
            room_id=1,
            room_number=101,
            check_in=dt.date(2024, 1, 1),
            check_out=dt.date(2024, 1, 4),
            total_amount=Decimal("300.00"),
        )
        # A trim, not a full cancellation, so the preview queries the database.
        instruction = write_ops.CancelRoomInstruction(
            booking_room_id=1, new_check_out=dt.date(2024, 1, 2),
        )
        with pytest.raises(write_ops.BookingUnavailableError):
            asyncio.run(
                _refund_preview(_pool_that_fails_to_connect(), row, instruction),
            )
