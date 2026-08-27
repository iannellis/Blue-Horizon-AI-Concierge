"""Shared database invariant checks used by the eval harness and stress test.

`eval.evaluators._booking` and `eval.stress.db` both need to answer the same
question against the same schema -- does `booking_rooms` contain two rows
for the same room with overlapping stays? -- so that query lives here once
and both call it, instead of each carrying its own copy that could silently
drift out of sync with the other.

`booking_rooms` carries a GiST exclusion constraint,
`booking_rooms_no_overlap` (see
`blue_horizon.load_data.booking_pgsql.setup_schema`'s `schema.sql`), that
makes the overlap this module looks for impossible to insert or update into
existence in the first place. This query is therefore belt-and-suspenders
against real agent-driven usage, not the primary line of defense: it would
only ever find a match if the constraint were dropped, disabled, or
bypassed by something other than a normal INSERT/UPDATE (e.g. a future
schema change, or historical rows loaded before the constraint existed).
Its true-positive path is accordingly no longer exercisable through normal
agent usage -- see `tests/booking/test_db_invariants.py`, which tests the
constraint's own refusal directly, and separately proves this query would
still catch a violation by temporarily dropping the constraint inside a
rolled-back transaction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from psycopg.rows import tuple_row

if TYPE_CHECKING:
    import psycopg


class BookingRoomOverlap(NamedTuple):
    """One pair of `booking_rooms` rows on the same room with overlapping stays.

    Attributes:
        a_booking_room_id: `booking_room_id` of the earlier-inserted row in
            the pair (the one with the smaller id).
        b_booking_room_id: `booking_room_id` of the later-inserted row in the
            pair.
        room_id: The `rooms.room_id` both rows share.

    """

    a_booking_room_id: int
    b_booking_room_id: int
    room_id: int


async def find_overlapping_booking_rooms(
    cur: psycopg.AsyncCursor,
) -> list[BookingRoomOverlap]:
    """Find pairs of `booking_rooms` rows on the same room with overlapping stays.

    A double-booking is two `booking_rooms` rows for the same `room_id` whose
    `[check_in, check_out)` spans share at least one night. Adjacent stays
    (one's `check_out` equals the other's `check_in`, e.g. same-day turnover)
    are not an overlap and never match.

    Args:
        cur: An open async cursor. The caller is responsible for the
            connection's `search_path` already resolving `booking_rooms` to
            the intended schema (e.g. via `SET search_path`). Its
            `row_factory` is temporarily switched to `tuple_row` for this
            query and restored before returning, so this works regardless of
            what row shape the caller's cursor was opened with (e.g.
            `dict_row`, used elsewhere in this codebase).

    Returns:
        One `BookingRoomOverlap` per violating pair. Empty when no two
        `booking_rooms` rows on the same room overlap.

    """
    original_row_factory = cur.row_factory
    cur.row_factory = tuple_row
    try:
        await cur.execute(
            """
            SELECT a.booking_room_id AS a_id, b.booking_room_id AS b_id,
                   a.room_id
            FROM booking_rooms a
            JOIN booking_rooms b
              ON a.room_id = b.room_id
             AND a.booking_room_id < b.booking_room_id
             AND a.check_in < b.check_out
             AND b.check_in < a.check_out
            """,
        )
        rows = await cur.fetchall()
    finally:
        cur.row_factory = original_row_factory
    return [BookingRoomOverlap(*row) for row in rows]
