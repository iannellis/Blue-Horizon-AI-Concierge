"""Tests for `blue_horizon.agents.booking.write_ops.resolve_trim`.

`resolve_trim` is pure date arithmetic shared by `_apply_cancel_instruction`
(which locks and writes) and `factory._refund_preview` (which only reads,
for the confirmation dialog), so it needs no database and runs in the
default (non-`db_integration`) suite. The trim/cancel behaviour this exists
to serve is also covered end-to-end through `cancel_booking` in
`tests/booking/test_write_ops.py`, but that coverage is `db_integration`-only
and does not run by default -- these tests close that gap.
"""
# ruff: noqa: S101

from __future__ import annotations

import datetime as dt

import pytest

from blue_horizon.agents.booking.write_ops import (
    BookingWriteError,
    CancelRoomInstruction,
    TrimRanges,
    resolve_trim,
)

_CHECK_IN = dt.date(2026, 9, 1)
_CHECK_OUT = dt.date(2026, 9, 10)


class TestResolveTrim:
    """resolve_trim() turns a trim/cancel instruction into release ranges."""

    def test_no_edges_set_cancels_whole_stay(self) -> None:
        """Neither edge set means the whole stay is released."""
        instruction = CancelRoomInstruction(booking_room_id=1)
        assert (
            resolve_trim(instruction, check_in=_CHECK_IN, check_out=_CHECK_OUT)
            is None
        )

    def test_front_trim(self) -> None:
        """A later check-in releases the front of the stay."""
        new_check_in = dt.date(2026, 9, 4)
        instruction = CancelRoomInstruction(
            booking_room_id=1, new_check_in=new_check_in,
        )
        assert resolve_trim(
            instruction, check_in=_CHECK_IN, check_out=_CHECK_OUT,
        ) == TrimRanges(
            released_start=_CHECK_IN,
            released_end=new_check_in,
            remaining_in=new_check_in,
            remaining_out=_CHECK_OUT,
        )

    def test_back_trim(self) -> None:
        """An earlier check-out releases the back of the stay."""
        new_check_out = dt.date(2026, 9, 7)
        instruction = CancelRoomInstruction(
            booking_room_id=1, new_check_out=new_check_out,
        )
        assert resolve_trim(
            instruction, check_in=_CHECK_IN, check_out=_CHECK_OUT,
        ) == TrimRanges(
            released_start=new_check_out,
            released_end=_CHECK_OUT,
            remaining_in=_CHECK_IN,
            remaining_out=new_check_out,
        )

    def test_both_edges_changed_is_rejected(self) -> None:
        """Changing both edges at once would leave a mid-stay hole."""
        instruction = CancelRoomInstruction(
            booking_room_id=1,
            new_check_in=dt.date(2026, 9, 4),
            new_check_out=dt.date(2026, 9, 7),
        )
        with pytest.raises(BookingWriteError, match="both ends"):
            resolve_trim(instruction, check_in=_CHECK_IN, check_out=_CHECK_OUT)

    def test_front_edge_outside_stay_is_rejected(self) -> None:
        """A new check-in on or before the current one is not a trim."""
        instruction = CancelRoomInstruction(
            booking_room_id=1, new_check_in=dt.date(2026, 8, 25),
        )
        with pytest.raises(BookingWriteError, match="Trim start"):
            resolve_trim(instruction, check_in=_CHECK_IN, check_out=_CHECK_OUT)

    def test_back_edge_outside_stay_is_rejected(self) -> None:
        """A new check-out on or after the current one is not a trim."""
        instruction = CancelRoomInstruction(
            booking_room_id=1, new_check_out=dt.date(2026, 9, 15),
        )
        with pytest.raises(BookingWriteError, match="Trim end"):
            resolve_trim(instruction, check_in=_CHECK_IN, check_out=_CHECK_OUT)

    def test_check_in_echoing_current_value_is_dropped(self) -> None:
        """A restated, unchanged check-in is treated as unset, not a trim."""
        new_check_out = dt.date(2026, 9, 7)
        instruction = CancelRoomInstruction(
            booking_room_id=1,
            new_check_in=_CHECK_IN,
            new_check_out=new_check_out,
        )
        assert resolve_trim(
            instruction, check_in=_CHECK_IN, check_out=_CHECK_OUT,
        ) == TrimRanges(
            released_start=new_check_out,
            released_end=_CHECK_OUT,
            remaining_in=_CHECK_IN,
            remaining_out=new_check_out,
        )

    def test_check_out_echoing_current_value_is_dropped(self) -> None:
        """A restated, unchanged check-out is treated as unset, not a trim."""
        new_check_in = dt.date(2026, 9, 4)
        instruction = CancelRoomInstruction(
            booking_room_id=1,
            new_check_in=new_check_in,
            new_check_out=_CHECK_OUT,
        )
        assert resolve_trim(
            instruction, check_in=_CHECK_IN, check_out=_CHECK_OUT,
        ) == TrimRanges(
            released_start=_CHECK_IN,
            released_end=new_check_in,
            remaining_in=new_check_in,
            remaining_out=_CHECK_OUT,
        )

    def test_both_edges_echoing_current_values_cancels_whole_stay(self) -> None:
        """Both edges restated unchanged is equivalent to no edges set."""
        instruction = CancelRoomInstruction(
            booking_room_id=1, new_check_in=_CHECK_IN, new_check_out=_CHECK_OUT,
        )
        assert (
            resolve_trim(instruction, check_in=_CHECK_IN, check_out=_CHECK_OUT)
            is None
        )
