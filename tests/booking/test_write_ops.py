"""Tests for `blue_horizon.agents.booking.write_ops` against a real database.

These exercise the only code paths permitted to mutate `bookings`,
`booking_rooms`, or `room_availability.status`, against a real Postgres
connected as `bh_agent_rw` (`PGSQL_RW_DB_URL`). Every test that leaves a
booking behind cleans it up in a `finally` block, so a single run does not
permanently consume inventory even though the shared CI branch is reset
before the whole session anyway.

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

from blue_horizon.agents.booking import write_ops

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

pytestmark = pytest.mark.db_integration

if platform.system() == "Windows":
    # psycopg's async mode cannot run on Windows's default ProactorEventLoop
    # (the loop `asyncio.run()` otherwise selects). Matches the same fix in
    # `eval/booking_db_manager.py`.
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

_SEARCH_CANDIDATE_ROOMS = 60
# The seeded `room_availability` dataset spans a fixed, roughly year-long
# window that is not anchored to the real-world current date (it was seeded
# once, at rest, rather than regenerated relative to "today"), so searches
# scan the whole span rather than starting from `dt.date.today()`.
_SEARCH_WINDOW_DAYS = 366


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

    A fresh pool is opened and closed inside each test's own `asyncio.run`
    call rather than shared across tests, since `AsyncConnectionPool`'s
    internal asyncio primitives are bound to the event loop they were
    created in.

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


async def _find_available_block(
    pool: AsyncConnectionPool[Any],
    *,
    nights: int,
    not_before: dt.date | None = None,
) -> tuple[write_ops.RoomRequest, Decimal]:
    """Find a room and a contiguous block of currently-bookable nights.

    Scans a bounded number of rooms, most likely to succeed rather than
    depend on hardcoded dates that may already be consumed by other work on
    a shared branch.

    Args:
        pool: Read-write booking database pool.
        nights: Number of consecutive nights required.
        not_before: Earliest acceptable check-in date. Defaults to the
            earliest date in `room_availability` -- the seeded dataset spans
            a fixed window that is not anchored to the real-world current
            date, so `dt.date.today()` is not a usable lower bound here.

    Returns:
        tuple: A `RoomRequest` for the found block, and the total price of
        those nights at the time of the search (informational; re-priced
        under lock by every write_ops function that uses it).

    Raises:
        RuntimeError: If no qualifying block is found among the candidates
        searched -- indicates the shared branch is in an unusually
        fully-booked state.

    """
    async with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        if not_before is None:
            await cur.execute("SELECT MIN(date) AS earliest FROM room_availability")
            not_before = (await cur.fetchone())["earliest"]

        await cur.execute(
            "SELECT room_id, room_number FROM rooms ORDER BY room_number LIMIT %s",
            (_SEARCH_CANDIDATE_ROOMS,),
        )
        candidates = await cur.fetchall()

        for candidate in candidates:
            await cur.execute(
                """
                SELECT date, status, price FROM room_availability
                WHERE room_id = %s AND date >= %s
                ORDER BY date LIMIT %s
                """,
                (candidate["room_id"], not_before, _SEARCH_WINDOW_DAYS),
            )
            rows = await cur.fetchall()
            block = _first_contiguous_free_block(rows, nights=nights)
            if block is not None:
                check_in, check_out, total = block
                return (
                    write_ops.RoomRequest(
                        room_id=candidate["room_id"],
                        room_number=candidate["room_number"],
                        check_in=check_in,
                        check_out=check_out,
                    ),
                    total,
                )

    msg = (
        f"No {nights}-night available block found among the first "
        f"{_SEARCH_CANDIDATE_ROOMS} rooms; the shared branch may need a reset."
    )
    raise RuntimeError(msg)


def _first_contiguous_free_block(
    rows: list[dict[str, Any]], *, nights: int,
) -> tuple[dt.date, dt.date, Decimal] | None:
    """Find the first run of `nights` consecutive, bookable rows.

    Args:
        rows: Availability rows for one room, ordered by date, with no gaps
            assumed between consecutive dates in the underlying data.
        nights: Run length required.

    Returns:
        `(check_in, check_out, total_price)` for the first qualifying run,
        or `None` if no run of that length is bookable.

    """
    run: list[dict[str, Any]] = []
    for row in rows:
        is_bookable = row["status"] == "Available" and row["price"] is not None
        continues_run = run and row["date"] == run[-1]["date"] + dt.timedelta(days=1)
        if is_bookable and (not run or continues_run):
            run.append(row)
        elif is_bookable:
            run = [row]
        else:
            run = []
        if len(run) == nights:
            total = sum((r["price"] for r in run), Decimal("0.00"))
            return run[0]["date"], run[-1]["date"] + dt.timedelta(days=1), total
    return None


async def _first_two_customer_ids(pool: AsyncConnectionPool[Any]) -> tuple[int, int]:
    """Fetch two distinct seeded customer ids.

    Args:
        pool: Read-write booking database pool.

    Returns:
        tuple[int, int]: Two distinct `customer_id` values.

    Raises:
        RuntimeError: If fewer than two customers are seeded.

    """
    async with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(
            "SELECT customer_id FROM customers ORDER BY customer_id LIMIT 2",
        )
        rows = await cur.fetchall()
    if len(rows) < 2:  # noqa: PLR2004
        msg = "Fewer than two seeded customers; cannot test cross-customer refusal."
        raise RuntimeError(msg)
    return rows[0]["customer_id"], rows[1]["customer_id"]


async def _first_booking_room_id(
    pool: AsyncConnectionPool[Any], *, customer_id: int, booking_id: int,
) -> int:
    """Look up the real `booking_room_id` for a just-committed booking's first room.

    `CommitResult.rooms[i].booking_room_id` is never populated by
    `commit_booking` -- it inserts each `booking_rooms` row without reading
    the generated id back into the `PricedRoomStay` it returns -- so, exactly
    like the real `propose_cancellation`/`propose_modification` tools in
    `blue_horizon.agents.booking.factory`, this resolves it via
    `list_bookings` instead.

    Args:
        pool: Read-write booking database pool.
        customer_id: Owner of the booking.
        booking_id: Booking to look up.

    Returns:
        int: `booking_room_id` of the booking's first room-stay.

    """
    bookings = await write_ops.list_bookings(pool, customer_id=customer_id)
    matching = next(b for b in bookings if b.booking_id == booking_id)
    return matching.rooms[0].booking_room_id


async def _availability_status(
    pool: AsyncConnectionPool[Any], *, room_id: int, on_date: dt.date,
) -> tuple[str, Decimal | None]:
    """Read one night's `status`/`price` directly.

    Args:
        pool: Read-write booking database pool.
        room_id: Room to check.
        on_date: Date to check.

    Returns:
        tuple[str, Decimal | None]: `(status, price)`.

    """
    async with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(
            "SELECT status, price FROM room_availability "
            "WHERE room_id = %s AND date = %s",
            (room_id, on_date),
        )
        row = await cur.fetchone()
    return row["status"], row["price"]


async def _hand_insert_conflicting_booking_room(
    pool: AsyncConnectionPool[Any],
    *,
    customer_id: int,
    room_id: int,
    check_in: dt.date,
    check_out: dt.date,
) -> int:
    """Insert a `booking_rooms` row directly, leaving `room_availability` untouched.

    Simulates the only scenario `write_ops`'s `booking_rooms_no_overlap`
    catch (`write_ops._refuse_on_overlap`) exists for: `room_availability`
    and `booking_rooms` disagreeing about who holds a night. `commit_booking`
    only ever consults `room_availability` before writing, so it cannot
    detect a conflict recorded solely in `booking_rooms` -- this bypasses
    `write_ops` entirely to create exactly that drift.

    Args:
        pool: Read-write booking database pool.
        customer_id: Owner of the hand-inserted `bookings` row.
        room_id: Room booked.
        check_in: First night of the stay (inclusive).
        check_out: Departure date (exclusive).

    Returns:
        int: The new `bookings` row's id, for later cleanup.

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
            """,
            (booking_id, room_id, check_in, check_out, Decimal("100.00")),
        )
    return booking_id


async def _cancel_hand_inserted_booking(
    pool: AsyncConnectionPool[Any], *, booking_id: int,
) -> None:
    """Clean up a booking created by `_hand_insert_conflicting_booking_room`.

    Mirrors `write_ops.cancel_booking`'s real full-cancel behavior:
    `bh_agent_rw` has no DELETE grant on `bookings` (see
    `regrant_booking_agent_role.sql`), so the row is marked cancelled rather
    than deleted.

    Args:
        pool: Read-write booking database pool.
        booking_id: Booking to delete `booking_rooms` rows from and cancel.

    """
    async with (
        pool.connection() as conn,
        conn.transaction(),
        conn.cursor() as cur,
    ):
        await cur.execute(
            "DELETE FROM booking_rooms WHERE booking_id = %s", (booking_id,),
        )
        await cur.execute(
            "UPDATE bookings SET status = 'cancelled', cancelled_at = now()"
            " WHERE booking_id = %s",
            (booking_id,),
        )


class TestCommitBooking:
    """`commit_booking` -- the only path that ever books a night."""

    def test_happy_path_prices_persists_and_marks_nights_booked(
        self, rw_db_url: str,
    ) -> None:
        """A commit returns a receipt and the DB agrees with it."""

        async def _run() -> None:
            async with _rw_pool(rw_db_url) as pool:
                customer_id, _ = await _first_two_customer_ids(pool)
                request, expected_total = await _find_available_block(pool, nights=2)

                result = await write_ops.commit_booking(
                    pool, customer_id=customer_id, rooms=[request],
                )
                try:
                    assert result.total_amount == expected_total
                    assert result.confirmation_number == f"BH{result.booking_id:06d}"
                    status, price = await _availability_status(
                        pool, room_id=request.room_id, on_date=request.check_in,
                    )
                    assert status == "Booked"
                    assert price is not None
                finally:
                    await write_ops.cancel_booking(
                        pool, customer_id=customer_id, booking_id=result.booking_id,
                    )

        asyncio.run(_run())

    def test_unavailable_night_raises_and_leaves_no_partial_writes(
        self, rw_db_url: str,
    ) -> None:
        """Booking an already-booked night fails, leaving no dangling booking."""

        async def _run() -> None:
            async with _rw_pool(rw_db_url) as pool:
                customer_id, _ = await _first_two_customer_ids(pool)
                request, _ = await _find_available_block(pool, nights=1)

                before = await write_ops.list_bookings(
                    pool, customer_id=customer_id,
                )
                active_before = len(
                    [b for b in before if b.status != "cancelled"],
                )

                first = await write_ops.commit_booking(
                    pool, customer_id=customer_id, rooms=[request],
                )
                try:
                    with pytest.raises(write_ops.BookingWriteError):
                        await write_ops.commit_booking(
                            pool, customer_id=customer_id, rooms=[request],
                        )
                    bookings = await write_ops.list_bookings(
                        pool, customer_id=customer_id,
                    )
                    active = [b for b in bookings if b.status != "cancelled"]
                    assert len(active) == active_before + 1
                    assert any(b.booking_id == first.booking_id for b in active)
                finally:
                    await write_ops.cancel_booking(
                        pool, customer_id=customer_id, booking_id=first.booking_id,
                    )

        asyncio.run(_run())

    def test_all_or_nothing_across_multiple_rooms(self, rw_db_url: str) -> None:
        """One unavailable room in a multi-room request rolls back every room."""

        async def _run() -> None:
            async with _rw_pool(rw_db_url) as pool:
                customer_id, _ = await _first_two_customer_ids(pool)
                good_request, _ = await _find_available_block(pool, nights=1)
                # An already-booked night: book it first so the second
                # `commit_booking` call below sees it as unavailable.
                bad_request, _ = await _find_available_block(
                    pool,
                    nights=1,
                    not_before=good_request.check_in + dt.timedelta(days=1),
                )
                blocker = await write_ops.commit_booking(
                    pool, customer_id=customer_id, rooms=[bad_request],
                )
                try:
                    with pytest.raises(write_ops.BookingWriteError):
                        await write_ops.commit_booking(
                            pool,
                            customer_id=customer_id,
                            rooms=[good_request, bad_request],
                        )
                    status, _ = await _availability_status(
                        pool,
                        room_id=good_request.room_id,
                        on_date=good_request.check_in,
                    )
                    assert status == "Available"
                finally:
                    await write_ops.cancel_booking(
                        pool, customer_id=customer_id, booking_id=blocker.booking_id,
                    )

        asyncio.run(_run())


class TestCancelBooking:
    """`cancel_booking` -- whole-stay cancel and end-trim."""

    def test_full_cancel_round_trip_preserves_price_and_reopens_night(
        self, rw_db_url: str,
    ) -> None:
        """Book then fully cancel: the night returns Available at the same price.

        Regression test for a real bug found during manual verification:
        `cancel_booking`'s default-instructions comprehension iterated the
        `_lock_booking_rooms` dict's keys (`int`s) instead of its values,
        raising `TypeError: 'int' object is not subscriptable` on every
        full-booking cancellation.
        """

        async def _run() -> None:
            async with _rw_pool(rw_db_url) as pool:
                customer_id, _ = await _first_two_customer_ids(pool)
                request, expected_total = await _find_available_block(pool, nights=2)
                committed = await write_ops.commit_booking(
                    pool, customer_id=customer_id, rooms=[request],
                )

                result = await write_ops.cancel_booking(
                    pool, customer_id=customer_id, booking_id=committed.booking_id,
                )

                assert result.fully_cancelled is True
                assert result.refunded_amount == expected_total
                status, price = await _availability_status(
                    pool, room_id=request.room_id, on_date=request.check_in,
                )
                assert status == "Available"
                assert price is not None  # price is never cleared by a cancel

        asyncio.run(_run())

    def test_mid_stay_hole_is_refused(self, rw_db_url: str) -> None:
        """Trimming both ends of a stay at once -- a mid-stay hole -- is refused."""

        async def _run() -> None:
            async with _rw_pool(rw_db_url) as pool:
                customer_id, _ = await _first_two_customer_ids(pool)
                request, _ = await _find_available_block(pool, nights=4)
                committed = await write_ops.commit_booking(
                    pool, customer_id=customer_id, rooms=[request],
                )
                booking_room_id = await _first_booking_room_id(
                    pool, customer_id=customer_id, booking_id=committed.booking_id,
                )
                try:
                    instruction = write_ops.CancelRoomInstruction(
                        booking_room_id=booking_room_id,
                        new_check_in=request.check_in + dt.timedelta(days=1),
                        new_check_out=request.check_out - dt.timedelta(days=1),
                    )
                    with pytest.raises(write_ops.BookingWriteError):
                        await write_ops.cancel_booking(
                            pool,
                            customer_id=customer_id,
                            booking_id=committed.booking_id,
                            rooms=[instruction],
                        )
                finally:
                    await write_ops.cancel_booking(
                        pool, customer_id=customer_id, booking_id=committed.booking_id,
                    )

        asyncio.run(_run())

    def test_trim_from_front_refunds_only_trimmed_nights(self, rw_db_url: str) -> None:
        """Trimming the front of a stay refunds and reopens only that night."""

        async def _run() -> None:
            async with _rw_pool(rw_db_url) as pool:
                customer_id, _ = await _first_two_customer_ids(pool)
                request, _ = await _find_available_block(pool, nights=3)
                committed = await write_ops.commit_booking(
                    pool, customer_id=customer_id, rooms=[request],
                )
                booking_room_id = await _first_booking_room_id(
                    pool, customer_id=customer_id, booking_id=committed.booking_id,
                )
                # Price is never cleared by booking, so this is the same
                # price a full cancel would refund for this one night.
                _, first_night_price = await _availability_status(
                    pool, room_id=request.room_id, on_date=request.check_in,
                )
                try:
                    new_check_in = request.check_in + dt.timedelta(days=1)
                    instruction = write_ops.CancelRoomInstruction(
                        booking_room_id=booking_room_id, new_check_in=new_check_in,
                    )
                    result = await write_ops.cancel_booking(
                        pool,
                        customer_id=customer_id,
                        booking_id=committed.booking_id,
                        rooms=[instruction],
                    )
                    assert result.fully_cancelled is False
                    assert result.refunded_amount == first_night_price
                    trimmed_status, trimmed_price = await _availability_status(
                        pool, room_id=request.room_id, on_date=request.check_in,
                    )
                    assert trimmed_status == "Available"
                    assert trimmed_price is not None
                    kept_status, _ = await _availability_status(
                        pool, room_id=request.room_id, on_date=new_check_in,
                    )
                    assert kept_status == "Booked"
                finally:
                    await write_ops.cancel_booking(
                        pool, customer_id=customer_id, booking_id=committed.booking_id,
                    )

        asyncio.run(_run())

    def test_trim_echoing_unchanged_edge_is_not_treated_as_both_ends(
        self, rw_db_url: str,
    ) -> None:
        """A trim that restates the untouched edge still resolves as a trim.

        Regression test for a real bug found by running the eval suite: a
        guest saying "shorten to <check-in> through <new check-out>"
        naturally restates the unchanged check-in alongside the edge that
        actually moved. `CancelRoomInstruction` carrying both dates was
        rejected outright as an unsupported both-ends trim, even though one
        of them was identical to the stay's current check-in.
        """

        async def _run() -> None:
            async with _rw_pool(rw_db_url) as pool:
                customer_id, _ = await _first_two_customer_ids(pool)
                request, _ = await _find_available_block(pool, nights=3)
                committed = await write_ops.commit_booking(
                    pool, customer_id=customer_id, rooms=[request],
                )
                booking_room_id = await _first_booking_room_id(
                    pool, customer_id=customer_id, booking_id=committed.booking_id,
                )
                try:
                    new_check_out = request.check_out - dt.timedelta(days=1)
                    instruction = write_ops.CancelRoomInstruction(
                        booking_room_id=booking_room_id,
                        new_check_in=request.check_in,  # unchanged, merely echoed
                        new_check_out=new_check_out,
                    )
                    result = await write_ops.cancel_booking(
                        pool,
                        customer_id=customer_id,
                        booking_id=committed.booking_id,
                        rooms=[instruction],
                    )
                    assert result.fully_cancelled is False
                    trimmed_status, _ = await _availability_status(
                        pool, room_id=request.room_id, on_date=new_check_out,
                    )
                    assert trimmed_status == "Available"
                    kept_status, _ = await _availability_status(
                        pool, room_id=request.room_id, on_date=request.check_in,
                    )
                    assert kept_status == "Booked"
                finally:
                    await write_ops.cancel_booking(
                        pool, customer_id=customer_id, booking_id=committed.booking_id,
                    )

        asyncio.run(_run())

    def test_wrong_customer_is_refused(self, rw_db_url: str) -> None:
        """Cancelling another guest's booking is refused."""

        async def _run() -> None:
            async with _rw_pool(rw_db_url) as pool:
                owner_id, other_id = await _first_two_customer_ids(pool)
                request, _ = await _find_available_block(pool, nights=1)
                committed = await write_ops.commit_booking(
                    pool, customer_id=owner_id, rooms=[request],
                )
                try:
                    with pytest.raises(write_ops.BookingWriteError):
                        await write_ops.cancel_booking(
                            pool,
                            customer_id=other_id,
                            booking_id=committed.booking_id,
                        )
                finally:
                    await write_ops.cancel_booking(
                        pool, customer_id=owner_id, booking_id=committed.booking_id,
                    )

        asyncio.run(_run())

    def test_unknown_booking_raises(self, rw_db_url: str) -> None:
        """Cancelling a nonexistent booking id raises `BookingWriteError`."""

        async def _run() -> None:
            async with _rw_pool(rw_db_url) as pool:
                customer_id, _ = await _first_two_customer_ids(pool)
                with pytest.raises(write_ops.BookingWriteError):
                    await write_ops.cancel_booking(
                        pool, customer_id=customer_id, booking_id=-1,
                    )

        asyncio.run(_run())


class TestModifyBooking:
    """`modify_booking` -- atomic release-and-reacquire."""

    def test_modify_releases_old_nights_and_books_new_ones(
        self, rw_db_url: str,
    ) -> None:
        """A modification frees the old room-stay and books the replacement."""

        async def _run() -> None:
            async with _rw_pool(rw_db_url) as pool:
                customer_id, _ = await _first_two_customer_ids(pool)
                old_request, _ = await _find_available_block(pool, nights=2)
                new_request, _ = await _find_available_block(
                    pool,
                    nights=2,
                    not_before=old_request.check_out + dt.timedelta(days=30),
                )
                committed = await write_ops.commit_booking(
                    pool, customer_id=customer_id, rooms=[old_request],
                )
                booking_room_id = await _first_booking_room_id(
                    pool, customer_id=customer_id, booking_id=committed.booking_id,
                )
                try:
                    change = write_ops.ModifyRoomInstruction(
                        booking_room_id=booking_room_id,
                        new_room_id=new_request.room_id,
                        new_room_number=new_request.room_number,
                        new_check_in=new_request.check_in,
                        new_check_out=new_request.check_out,
                    )
                    result = await write_ops.modify_booking(
                        pool,
                        customer_id=customer_id,
                        booking_id=committed.booking_id,
                        changes=[change],
                    )
                    assert len(result.rooms) == 1
                    assert result.rooms[0].room_id == new_request.room_id
                    assert result.rooms[0].check_in == new_request.check_in

                    old_status, _ = await _availability_status(
                        pool, room_id=old_request.room_id, on_date=old_request.check_in,
                    )
                    assert old_status == "Available"
                    new_status, _ = await _availability_status(
                        pool, room_id=new_request.room_id, on_date=new_request.check_in,
                    )
                    assert new_status == "Booked"
                finally:
                    await write_ops.cancel_booking(
                        pool, customer_id=customer_id, booking_id=committed.booking_id,
                    )

        asyncio.run(_run())


class TestPriceModificationRoom:
    """`price_modification_room` -- the propose-time preview for a modification.

    Unlike `modify_booking`, this must price the replacement range without
    writing, so it cannot release the existing room-stay first. It has to
    treat nights the guest already holds under that room-stay as available on
    its own -- these tests are the regression case for when it did not: an
    in-place extension raised `BookingWriteError` on nights the guest already
    owned, even though `modify_booking` would have released and rebooked them
    without issue.
    """

    def test_extending_a_stay_in_place_prices_without_raising(
        self, rw_db_url: str,
    ) -> None:
        """Extending a booked stay's checkout in the same room does not raise."""

        async def _run() -> None:
            async with _rw_pool(rw_db_url) as pool:
                customer_id, _ = await _first_two_customer_ids(pool)
                original, _ = await _find_available_block(pool, nights=4)
                # Book only the first half; the second half stays `Available`
                # so the extension below has somewhere free to grow into.
                held_check_out = original.check_in + dt.timedelta(days=2)
                held = write_ops.RoomRequest(
                    room_id=original.room_id,
                    room_number=original.room_number,
                    check_in=original.check_in,
                    check_out=held_check_out,
                )
                committed = await write_ops.commit_booking(
                    pool, customer_id=customer_id, rooms=[held],
                )
                try:
                    extended = write_ops.RoomRequest(
                        room_id=original.room_id,
                        room_number=original.room_number,
                        check_in=original.check_in,
                        check_out=original.check_out,
                    )
                    priced = await write_ops.price_modification_room(
                        pool,
                        existing_room_id=held.room_id,
                        existing_check_in=held.check_in,
                        existing_check_out=held.check_out,
                        request=extended,
                    )
                    assert priced.check_out == original.check_out
                    assert priced.total_amount > Decimal("0.00")
                finally:
                    await write_ops.cancel_booking(
                        pool, customer_id=customer_id, booking_id=committed.booking_id,
                    )

        asyncio.run(_run())

    def test_nights_outside_owned_range_still_require_available(
        self, rw_db_url: str,
    ) -> None:
        """A replacement range is still refused if its new nights are taken."""

        async def _run() -> None:
            async with _rw_pool(rw_db_url) as pool:
                first_customer, second_customer = await _first_two_customer_ids(pool)
                original, _ = await _find_available_block(pool, nights=4)
                held = write_ops.RoomRequest(
                    room_id=original.room_id,
                    room_number=original.room_number,
                    check_in=original.check_in,
                    check_out=original.check_in + dt.timedelta(days=2),
                )
                # Someone else takes the second half of the same block, so
                # extending `held` into it must still be refused.
                blocker = write_ops.RoomRequest(
                    room_id=original.room_id,
                    room_number=original.room_number,
                    check_in=held.check_out,
                    check_out=original.check_out,
                )
                first_commit = await write_ops.commit_booking(
                    pool, customer_id=first_customer, rooms=[held],
                )
                second_commit = await write_ops.commit_booking(
                    pool, customer_id=second_customer, rooms=[blocker],
                )
                try:
                    extended = write_ops.RoomRequest(
                        room_id=held.room_id,
                        room_number=held.room_number,
                        check_in=held.check_in,
                        check_out=blocker.check_out,
                    )
                    with pytest.raises(write_ops.BookingWriteError):
                        await write_ops.price_modification_room(
                            pool,
                            existing_room_id=held.room_id,
                            existing_check_in=held.check_in,
                            existing_check_out=held.check_out,
                            request=extended,
                        )
                finally:
                    await write_ops.cancel_booking(
                        pool,
                        customer_id=first_customer,
                        booking_id=first_commit.booking_id,
                    )
                    await write_ops.cancel_booking(
                        pool,
                        customer_id=second_customer,
                        booking_id=second_commit.booking_id,
                    )

        asyncio.run(_run())


class TestStaleProposalCommit:
    """A commit whose priced nights were taken by someone else in the meantime."""

    def test_commit_after_nights_taken_elsewhere_fails_without_partial_writes(
        self, rw_db_url: str,
    ) -> None:
        """A second commit for the same nights fails cleanly once the first commits.

        This is the scenario a stale, unconfirmed proposal hits if the guest
        confirms it after someone else already took the nights: `write_ops`
        re-validates under lock at commit time regardless of what a
        proposal's dialog showed.
        """

        async def _run() -> None:
            async with _rw_pool(rw_db_url) as pool:
                first_customer, second_customer = await _first_two_customer_ids(pool)
                request, _ = await _find_available_block(pool, nights=1)

                first_commit = await write_ops.commit_booking(
                    pool, customer_id=first_customer, rooms=[request],
                )
                try:
                    with pytest.raises(write_ops.BookingWriteError):
                        await write_ops.commit_booking(
                            pool, customer_id=second_customer, rooms=[request],
                        )
                    second_bookings = await write_ops.list_bookings(
                        pool, customer_id=second_customer,
                    )
                    assert all(
                        b.booking_id != first_commit.booking_id
                        for b in second_bookings
                    )
                finally:
                    await write_ops.cancel_booking(
                        pool,
                        customer_id=first_customer,
                        booking_id=first_commit.booking_id,
                    )

        asyncio.run(_run())


class TestCommitBookingConstraintTranslation:
    """`commit_booking` translates a `booking_rooms` constraint hit cleanly."""

    def test_data_drift_overlap_is_refused_not_leaked(self, rw_db_url: str) -> None:
        """A `booking_rooms`/`room_availability` drift is refused, not leaked raw.

        `_price_one_room`'s lock only ever consults `room_availability`, so
        it cannot itself detect a night already held in `booking_rooms` but
        never marked `Booked` there -- exactly the drift
        `booking_rooms_no_overlap` (and `write_ops._refuse_on_overlap`)
        exist to catch as a second line of defense. This proves that catch
        surfaces as `write_ops.BookingWriteError`, not a raw
        `psycopg.errors.ExclusionViolation` escaping `commit_booking`, and
        that the attempted commit leaves no partial write behind.
        """

        async def _run() -> None:
            async with _rw_pool(rw_db_url) as pool:
                first_customer, second_customer = await _first_two_customer_ids(pool)
                request, _ = await _find_available_block(pool, nights=1)

                drift_booking_id = await _hand_insert_conflicting_booking_room(
                    pool,
                    customer_id=first_customer,
                    room_id=request.room_id,
                    check_in=request.check_in,
                    check_out=request.check_out,
                )
                try:
                    before = await write_ops.list_bookings(
                        pool, customer_id=second_customer,
                    )
                    with pytest.raises(write_ops.BookingWriteError):
                        await write_ops.commit_booking(
                            pool, customer_id=second_customer, rooms=[request],
                        )
                    after = await write_ops.list_bookings(
                        pool, customer_id=second_customer,
                    )
                    assert len(after) == len(before)
                    status, _ = await _availability_status(
                        pool, room_id=request.room_id, on_date=request.check_in,
                    )
                    assert status == "Available"
                finally:
                    await _cancel_hand_inserted_booking(
                        pool, booking_id=drift_booking_id,
                    )

        asyncio.run(_run())


class TestListCustomers:
    """`list_customers` -- the read path the UI's identity dropdown relies on."""

    def test_result_is_capped_at_seeded_customer_count(self, rw_db_url: str) -> None:
        """`customers` may hold far more rows than the dropdown should offer.

        `booking_pgsql.reload_sql_tables` loads every source customer, not
        just the dropdown identities, so this asserts `list_customers` caps
        its result at the `seeded_customer_count` it is given, rather than
        trusting the table to already be small.
        """
        seeded_customer_count = 3

        async def _run() -> None:
            async with _rw_pool(rw_db_url) as pool:
                async with pool.connection() as conn, conn.cursor() as cur:
                    await cur.execute("SELECT count(*) FROM customers;")
                    row = await cur.fetchone()
                total_customers = row[0] if row is not None else 0

                customers = await write_ops.list_customers(
                    pool, seeded_customer_count=seeded_customer_count,
                )

                expected_count = min(seeded_customer_count, total_customers)
                assert len(customers) == expected_count
                assert [c.customer_id for c in customers] == list(
                    range(1, expected_count + 1),
                )

        asyncio.run(_run())


class TestListBookings:
    """`list_bookings` -- the read path the confirmation panel relies on."""

    def test_list_bookings_includes_a_freshly_committed_booking(
        self, rw_db_url: str,
    ) -> None:
        """A just-committed booking is immediately visible to `list_bookings`."""

        async def _run() -> None:
            async with _rw_pool(rw_db_url) as pool:
                customer_id, _ = await _first_two_customer_ids(pool)
                request, expected_total = await _find_available_block(pool, nights=1)
                committed = await write_ops.commit_booking(
                    pool, customer_id=customer_id, rooms=[request],
                )
                try:
                    bookings = await write_ops.list_bookings(
                        pool, customer_id=customer_id,
                    )
                    matching = [
                        b for b in bookings if b.booking_id == committed.booking_id
                    ]
                    assert len(matching) == 1
                    assert matching[0].total_amount == expected_total
                    assert matching[0].confirmation_number == (
                        committed.confirmation_number
                    )
                finally:
                    await write_ops.cancel_booking(
                        pool, customer_id=customer_id, booking_id=committed.booking_id,
                    )

        asyncio.run(_run())
