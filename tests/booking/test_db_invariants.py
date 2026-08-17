"""Tests for `booking_rooms`'s constraint and trigger backstops against a real DB.

`booking_rooms` carries a GiST exclusion constraint (see
`blue_horizon.load_data.booking_pgsql.setup_booking_rooms_schema`) that makes
two rows for the same room with overlapping `[check_in, check_out)` spans
impossible to insert or update into existence -- a double-booking is now a
schema-level impossibility, not merely something detected after the fact. It
also carries a `prevent_maintenance_booking` trigger (see
`setup_maintenance_booking_guard`) refusing any insert or update that would
cover a night `room_availability` marks `Maintenance`. This file is the
permanent positive/negative-control regression test for both guarantees
(the GiST one previously verified only once, ad hoc -- see the `398ee08`
commit message).

Before the constraint existed, the positive control here hand-inserted an
overlapping pair and asserted `eval.db_invariants.find_overlapping_booking_rooms`
flagged it after the fact (see history at
https://github.com/iannellis/Blue-Horizon-AI-Concierge/blob/e67fb9b33279219a0d4b6b8fc8a85bbb92733dfa/tests/booking/test_db_invariants.py,
exactly the follow-up that file's own docstring predicted would be needed).
That is no longer possible to set up through a normal insert -- the second
insert now fails outright -- so `TestBookingRoomsNoOverlapConstraint` covers
three things instead: the constraint itself refuses an overlapping insert
(`psycopg.errors.ExclusionViolation`); a legitimate back-to-back (same-day
turnover) insert is not mistakenly refused, and `find_overlapping_booking_rooms`
does not flag it either; and, since the constraint makes a real violation
unreachable through a normal insert, `find_overlapping_booking_rooms`'s
true-positive path is proven separately by temporarily dropping the
constraint inside a transaction that is always rolled back, so it never
survives the test. The constraint is deliberately not made `DEFERRABLE` to
make that last check easier: that would let overlapping rows exist
transiently mid-transaction in production too, weakening the guarantee the
constraint exists to provide.

Rows are inserted directly into `bookings`/`booking_rooms` via raw SQL,
bypassing `blue_horizon.agents.booking.write_ops` entirely:
`write_ops.commit_booking` already refuses to create an overlap, or to price
a night under `Maintenance`, before ever reaching either backstop (see
`TestCommitBooking` in `test_write_ops.py`), so exercising the backstops
themselves requires getting a write attempted some other way.

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

import psycopg
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


@pytest.fixture
def root_db_url() -> str:
    """Return `PGSQL_ROOT_DB_URL`, skipping the test if it is not set.

    `bh_agent_rw` has no DDL privilege on `booking_rooms` (see
    `regrant_booking_agent_role.sql`), so the one test here that needs to
    drop and recreate the `booking_rooms_no_overlap` constraint connects as
    the schema-owning root role instead. This must point at whatever branch
    the rest of this test run targets (e.g. the same branch behind
    `rw_db_url`) -- it is deliberately a different environment variable from
    `PGSQL_ROOT_PARENT_DB_URL`, which `booking_pgsql.reload_sql_tables` reads
    and which must never point at a branch a test run touches, since that
    function drops and recreates every table it loads.

    Returns:
        The root database URL.

    """
    url = os.environ.get("PGSQL_ROOT_DB_URL")
    if not url:
        pytest.skip("PGSQL_ROOT_DB_URL not set; skipping db_integration test.")
    return url


@asynccontextmanager
async def _pool(db_url: str) -> AsyncIterator[AsyncConnectionPool[Any]]:
    """Open a small connection pool for the duration of one test.

    Args:
        db_url: Database URL to connect with.

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


async def _first_room_night_with_status(
    pool: AsyncConnectionPool[Any], status: str,
) -> tuple[int, dt.date]:
    """Fetch a seeded `room_availability` row currently marked with the given status.

    Args:
        pool: Read-write booking database pool.
        status: `room_availability.status` value to match (e.g. `Maintenance`).

    Returns:
        The row's `room_id` and `date`.

    Raises:
        RuntimeError: If no row has that status.

    """
    async with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(
            "SELECT room_id, date FROM room_availability "
            "WHERE status = %s ORDER BY room_id, date LIMIT 1",
            (status,),
        )
        row = await cur.fetchone()
    if row is None:
        msg = f"No room_availability row is marked {status}."
        raise RuntimeError(msg)
    return row["room_id"], row["date"]


@asynccontextmanager
async def _hand_inserted_booking_room(
    pool: AsyncConnectionPool[Any],
    *,
    customer_id: int,
    room_id: int,
    check_in: dt.date,
    check_out: dt.date,
) -> AsyncIterator[int]:
    """Insert one `booking_rooms` row directly, bypassing `write_ops` entirely.

    If the insert violates `booking_rooms_no_overlap`, the exception
    propagates out of this context manager before it ever yields -- both the
    `bookings` row and the `booking_rooms` row this function attempted are
    rolled back together as one transaction, so callers do not need to (and
    should not) clean up after a caught violation.

    Args:
        pool: Read-write booking database pool.
        customer_id: Owner of the hand-inserted `bookings` row.
        room_id: Room booked.
        check_in: First night of the stay (inclusive).
        check_out: Departure date (exclusive).

    Yields:
        int: The inserted row's `booking_room_id`.

    Raises:
        psycopg.errors.ExclusionViolation: If the insert overlaps an
            existing `booking_rooms` row on the same room.

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

        await cur.execute(
            """
            INSERT INTO booking_rooms
                (booking_id, room_id, check_in, check_out, total_amount)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING booking_room_id
            """,
            (booking_id, room_id, check_in, check_out, _DUMMY_TOTAL_AMOUNT),
        )
        booking_room_id = (await cur.fetchone())["booking_room_id"]

    try:
        yield booking_room_id
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


class TestBookingRoomsNoOverlapConstraint:
    """Positive/negative controls for the `booking_rooms_no_overlap` constraint."""

    def test_second_overlapping_booking_is_rejected(self, rw_db_url: str) -> None:
        """Positive control: a second, overlapping booking on one room is refused.

        Regression test for the "confirmed to go red against a hand-inserted
        overlapping pair" verification the README describes -- previously
        only done once, ad hoc, before the query it originally exercised was
        trusted; now it exercises the schema-level constraint that replaced
        that query as the actual enforcement mechanism.
        """

        async def _run() -> None:
            async with _pool(rw_db_url) as pool:
                customer_id = await _first_customer_id(pool)
                room_id = await _first_room_id(pool)
                async with _hand_inserted_booking_room(
                    pool,
                    customer_id=customer_id,
                    room_id=room_id,
                    check_in=dt.date(2099, 1, 10),
                    check_out=dt.date(2099, 1, 15),
                ):
                    with pytest.raises(psycopg.errors.ExclusionViolation):
                        async with _hand_inserted_booking_room(
                            pool,
                            customer_id=customer_id,
                            room_id=room_id,
                            check_in=dt.date(2099, 1, 12),  # overlaps above
                            check_out=dt.date(2099, 1, 18),
                        ):
                            pass

        asyncio.run(_run())

    def test_second_adjacent_booking_is_not_rejected(self, rw_db_url: str) -> None:
        """Negative control: back-to-back same-day-turnover bookings both succeed.

        Guards against a constraint that is too strict (rejecting legitimate
        turnover) as well as against a `find_overlapping_booking_rooms` false
        positive on the same data -- the same tautology-avoidance goal the
        historical `room_availability` version of this check existed for.
        """

        async def _run() -> None:
            async with _pool(rw_db_url) as pool:
                customer_id = await _first_customer_id(pool)
                room_id = await _first_room_id(pool)
                async with (
                    _hand_inserted_booking_room(
                        pool,
                        customer_id=customer_id,
                        room_id=room_id,
                        check_in=dt.date(2098, 1, 10),
                        check_out=dt.date(2098, 1, 15),
                    ) as first_id,
                    _hand_inserted_booking_room(
                        pool,
                        customer_id=customer_id,
                        room_id=room_id,
                        check_in=dt.date(2098, 1, 15),  # adjacent, no gap
                        check_out=dt.date(2098, 1, 20),
                    ) as second_id,
                ):
                    async with pool.connection() as conn, conn.cursor() as cur:
                        overlaps = await find_overlapping_booking_rooms(cur)
                    flagged_ids = {
                        oid
                        for o in overlaps
                        for oid in (o.a_booking_room_id, o.b_booking_room_id)
                    }
                    assert not ({first_id, second_id} & flagged_ids)

        asyncio.run(_run())

    def test_dropped_constraint_overlap_is_still_detected(
        self, root_db_url: str,
    ) -> None:
        """True-positive control: with the constraint gone, detection still fires.

        `booking_rooms_no_overlap` makes an overlapping pair impossible to
        create via a normal insert, so `find_overlapping_booking_rooms`'s
        true-positive path can no longer be exercised through one (see the
        two tests above). This manufactures the state the schema now
        forbids by dropping the constraint inside a transaction that always
        rolls back -- Postgres DDL is transactional, so `DROP CONSTRAINT`
        undoes cleanly on rollback just like any other statement, and the
        overlapping rows never survive to be seen by another test. Requires
        `PGSQL_ROOT_DB_URL`: `bh_agent_rw` has no DDL privilege on
        `booking_rooms` (see `regrant_booking_agent_role.sql`), so the
        `rw_db_url`-authenticated connection the other tests here use cannot
        run the `ALTER TABLE` this needs.
        """

        async def _run() -> None:
            async with _pool(root_db_url) as pool, pool.connection() as conn:
                customer_id = await _first_customer_id(pool)
                room_id = await _first_room_id(pool)
                async with (
                    conn.transaction(force_rollback=True),
                    conn.cursor(row_factory=dict_row) as cur,
                ):
                    await cur.execute(
                        "ALTER TABLE booking_rooms "
                        "DROP CONSTRAINT booking_rooms_no_overlap",
                    )

                    await cur.execute(
                        "INSERT INTO bookings (customer_id) VALUES (%s) "
                        "RETURNING booking_id",
                        (customer_id,),
                    )
                    row = await cur.fetchone()
                    assert row is not None
                    booking_id = row["booking_id"]

                    await cur.execute(
                        """
                        INSERT INTO booking_rooms
                            (booking_id, room_id, check_in, check_out, total_amount)
                        VALUES
                            (%s, %s, %s, %s, %s),
                            (%s, %s, %s, %s, %s)
                        RETURNING booking_room_id
                        """,
                        (
                            booking_id, room_id,
                            dt.date(2097, 1, 10), dt.date(2097, 1, 15),
                            _DUMMY_TOTAL_AMOUNT,
                            booking_id, room_id,
                            dt.date(2097, 1, 12), dt.date(2097, 1, 18),
                            _DUMMY_TOTAL_AMOUNT,
                        ),
                    )
                    inserted_ids = {
                        row["booking_room_id"] for row in await cur.fetchall()
                    }

                    overlaps = await find_overlapping_booking_rooms(cur)
                    flagged_ids = {
                        oid
                        for o in overlaps
                        for oid in (o.a_booking_room_id, o.b_booking_room_id)
                    }
                    assert inserted_ids <= flagged_ids

        asyncio.run(_run())


class TestMaintenanceBookingGuard:
    """Positive/negative controls for the `prevent_maintenance_booking` trigger."""

    def test_maintenance_night_is_rejected(self, rw_db_url: str) -> None:
        """Positive control: booking over a Maintenance night is refused.

        `write_ops._price_one_room` already refuses to price a night whose
        `room_availability.status` isn't `Available`, so a real
        `commit_booking` call never reaches this trigger; this test bypasses
        `write_ops` entirely (see the module docstring) to exercise the
        trigger itself.
        """

        async def _run() -> None:
            async with _pool(rw_db_url) as pool:
                customer_id = await _first_customer_id(pool)
                room_id, night = await _first_room_night_with_status(
                    pool, "Maintenance",
                )
                with pytest.raises(psycopg.errors.CheckViolation):
                    async with _hand_inserted_booking_room(
                        pool,
                        customer_id=customer_id,
                        room_id=room_id,
                        check_in=night,
                        check_out=night + dt.timedelta(days=1),
                    ):
                        pass

        asyncio.run(_run())

    def test_available_night_is_not_rejected(self, rw_db_url: str) -> None:
        """Negative control: booking an Available night is not refused.

        `room_availability` rows marked `Available` have no matching
        `booking_rooms` reservation by construction (see
        `reconcile_room_availability_status`), so this also cannot collide
        with `booking_rooms_no_overlap`.
        """

        async def _run() -> None:
            async with _pool(rw_db_url) as pool:
                customer_id = await _first_customer_id(pool)
                room_id, night = await _first_room_night_with_status(
                    pool, "Available",
                )
                async with _hand_inserted_booking_room(
                    pool,
                    customer_id=customer_id,
                    room_id=room_id,
                    check_in=night,
                    check_out=night + dt.timedelta(days=1),
                ):
                    pass  # No exception means the trigger did not over-fire.

        asyncio.run(_run())
