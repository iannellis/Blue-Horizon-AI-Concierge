"""Tests for `eval.db_invariants.find_overlapping_booking_rooms` against a real DB.

Exercises the shared overlap-detection query that both the eval harness
(`eval.evaluators._booking`) and the stress test (`eval.stress.db`) rely on
to catch double-bookings, via positive and negative controls: a
hand-inserted overlapping pair must be flagged (previously verified only
once, ad hoc -- see the `398ee08` commit message -- this makes that
verification a permanent regression test instead), and legitimate
non-overlapping stays, including the back-to-back same-day-turnover
boundary, must not be.

Rows are inserted directly into `bookings`/`booking_rooms` via raw SQL,
bypassing `blue_horizon.agents.booking.write_ops` entirely:
`write_ops.commit_booking` already refuses to create an overlap (see
`TestCommitBooking` in `test_write_ops.py`), so exercising the query itself
requires getting an overlapping pair into the table some other way. There is
deliberately no schema-level constraint preventing this today -- if one is
ever added (e.g. a GiST exclusion constraint on `booking_rooms`), the
positive control below stops being possible to set up and both this file and
`find_overlapping_booking_rooms` should be revisited together.

Marked `db_integration` and excluded from the default `pytest` run -- see
`.github/workflows/ci.yml`'s `db-integration-tests` job, which resets the
Development branch before running these.
"""
# ruff: noqa: S101

from __future__ import annotations

import asyncio
import datetime as dt
import os
import platform
from contextlib import asynccontextmanager
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pytest
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from eval.db_invariants import find_overlapping_booking_rooms

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

pytestmark = pytest.mark.db_integration

if platform.system() == "Windows":
    # psycopg's async mode cannot run on Windows's default ProactorEventLoop
    # (the loop `asyncio.run()` otherwise selects). Matches the same fix in
    # `test_write_ops.py` / `eval/booking_db_manager.py`.
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Placeholder price for hand-inserted rows -- the overlap query never reads
# `total_amount`, so its value is irrelevant beyond satisfying the column's
# `NOT NULL` constraint.
_DUMMY_TOTAL_AMOUNT = Decimal("100.00")


@pytest.fixture
def rw_db_url() -> str:
    """Return `PGSQL_RW_DB_URL`, skipping the test if it is not set.

    Returns:
        The read-write (`bh_agent_rw`) database URL.

    """
    url = os.environ.get("PGSQL_RW_DB_URL")
    if not url:
        pytest.skip("PGSQL_RW_DB_URL not set; skipping db_integration test.")
    return url


@asynccontextmanager
async def _rw_pool(db_url: str) -> AsyncIterator[AsyncConnectionPool[Any]]:
    """Open a small read-write connection pool for the duration of one test.

    Args:
        db_url: Read-write (`bh_agent_rw`) database URL.

    Yields:
        An open `AsyncConnectionPool`.

    """
    pool: AsyncConnectionPool[Any] = AsyncConnectionPool(
        conninfo=db_url, min_size=0, max_size=2, open=False,
    )
    await pool.open()
    try:
        yield pool
    finally:
        await pool.close()


async def _first_customer_id(pool: AsyncConnectionPool[Any]) -> int:
    """Fetch a seeded customer id to attribute hand-inserted bookings to.

    Args:
        pool: Read-write booking database pool.

    Returns:
        A `customer_id` value.

    Raises:
        RuntimeError: If no customers are seeded.

    """
    async with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(
            "SELECT customer_id FROM customers ORDER BY customer_id LIMIT 1",
        )
        row = await cur.fetchone()
    if row is None:
        msg = "No seeded customers; cannot attribute a hand-inserted booking."
        raise RuntimeError(msg)
    return row["customer_id"]


async def _first_room_id(pool: AsyncConnectionPool[Any]) -> int:
    """Fetch a seeded room id to hand-insert bookings against.

    Args:
        pool: Read-write booking database pool.

    Returns:
        A `room_id` value.

    Raises:
        RuntimeError: If no rooms are seeded.

    """
    async with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        await cur.execute("SELECT room_id FROM rooms ORDER BY room_id LIMIT 1")
        row = await cur.fetchone()
    if row is None:
        msg = "No seeded rooms; cannot hand-insert a booking against one."
        raise RuntimeError(msg)
    return row["room_id"]


@asynccontextmanager
async def _hand_inserted_booking_rooms(
    pool: AsyncConnectionPool[Any],
    *,
    customer_id: int,
    room_id: int,
    spans: list[tuple[dt.date, dt.date]],
) -> AsyncIterator[list[int]]:
    """Insert `booking_rooms` rows directly, bypassing `write_ops` entirely.

    Args:
        pool: Read-write booking database pool.
        customer_id: Owner of the hand-inserted `bookings` row.
        room_id: Room every span in `spans` is booked against.
        spans: `(check_in, check_out)` pairs to insert as separate
            `booking_rooms` rows, all under one new `bookings` row.

    Yields:
        list[int]: The `booking_room_id` of each inserted row, in the same
        order as `spans`.

    """
    async with (
        pool.connection() as conn,
        conn.transaction(),
        conn.cursor(row_factory=dict_row) as cur,
    ):
        await cur.execute(
            "INSERT INTO bookings (customer_id) VALUES (%s) RETURNING booking_id",
            (customer_id,),
        )
        booking_id = (await cur.fetchone())["booking_id"]

        booking_room_ids: list[int] = []
        for check_in, check_out in spans:
            await cur.execute(
                """
                INSERT INTO booking_rooms
                    (booking_id, room_id, check_in, check_out, total_amount)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING booking_room_id
                """,
                (booking_id, room_id, check_in, check_out, _DUMMY_TOTAL_AMOUNT),
            )
            booking_room_ids.append((await cur.fetchone())["booking_room_id"])

    try:
        yield booking_room_ids
    finally:
        # `bh_agent_rw` has no DELETE grant on `bookings` (see
        # `regrant_booking_agent_role.sql`) -- a `bookings` row is never
        # deleted, only marked cancelled, exactly like a real full cancel
        # (`write_ops.cancel_booking`). `booking_rooms` rows are deleted,
        # matching that same real cancel path.
        async with pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
            await cur.execute(
                "DELETE FROM booking_rooms WHERE booking_id = %s", (booking_id,),
            )
            await cur.execute(
                "UPDATE bookings SET status = 'cancelled', cancelled_at = now()"
                " WHERE booking_id = %s",
                (booking_id,),
            )


class TestFindOverlappingBookingRooms:
    """Positive/negative controls for `find_overlapping_booking_rooms`."""

    def test_hand_inserted_overlapping_pair_is_flagged(self, rw_db_url: str) -> None:
        """Positive control: two overlapping spans on one room are flagged.

        Regression test for the "confirmed to go red against a hand-inserted
        overlapping pair" verification the README describes -- previously
        only done once, ad hoc, before the query was trusted.
        """

        async def _run() -> None:
            async with _rw_pool(rw_db_url) as pool:
                customer_id = await _first_customer_id(pool)
                room_id = await _first_room_id(pool)
                spans = [
                    (dt.date(2099, 1, 10), dt.date(2099, 1, 15)),
                    (dt.date(2099, 1, 12), dt.date(2099, 1, 18)),  # overlaps above
                ]
                async with _hand_inserted_booking_rooms(
                    pool, customer_id=customer_id, room_id=room_id, spans=spans,
                ) as booking_room_ids:
                    async with pool.connection() as conn, conn.cursor() as cur:
                        overlaps = await find_overlapping_booking_rooms(cur)

                    flagged_pairs = {
                        frozenset((o.a_booking_room_id, o.b_booking_room_id))
                        for o in overlaps
                    }
                    assert frozenset(booking_room_ids) in flagged_pairs

        asyncio.run(_run())

    def test_hand_inserted_adjacent_pair_is_not_flagged(self, rw_db_url: str) -> None:
        """Negative control: back-to-back same-day-turnover spans are not flagged.

        Guards against a check that reports a violation (or none at all)
        regardless of its input -- the same tautology mistake the historical
        `room_availability` version of this check made.
        """

        async def _run() -> None:
            async with _rw_pool(rw_db_url) as pool:
                customer_id = await _first_customer_id(pool)
                room_id = await _first_room_id(pool)
                spans = [
                    (dt.date(2098, 1, 10), dt.date(2098, 1, 15)),
                    (dt.date(2098, 1, 15), dt.date(2098, 1, 20)),  # adjacent, no gap
                ]
                async with _hand_inserted_booking_rooms(
                    pool, customer_id=customer_id, room_id=room_id, spans=spans,
                ) as booking_room_ids:
                    async with pool.connection() as conn, conn.cursor() as cur:
                        overlaps = await find_overlapping_booking_rooms(cur)

                    flagged_ids = {
                        oid
                        for o in overlaps
                        for oid in (o.a_booking_room_id, o.b_booking_room_id)
                    }
                    assert not (set(booking_room_ids) & flagged_ids)

        asyncio.run(_run())
