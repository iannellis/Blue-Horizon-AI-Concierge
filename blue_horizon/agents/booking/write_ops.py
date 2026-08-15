"""Server-owned booking write operations.

These functions are the only code paths permitted to mutate `bookings`,
`booking_rooms`, or `room_availability.status`. The LLM never calls them
directly: it can only create a *proposal* (see
`blue_horizon.agents.booking.proposals`), and only the `/v1/booking/confirm`
endpoint -- triggered by a human clicking Confirm -- invokes them.

Every function opens exactly one explicit transaction and leaves no partial
writes behind on failure. `price` is never written by any function here:
`commit_booking` only ever sets `status = 'Booked'`, `cancel_booking` only
ever sets it back to `'Available'`, and `modify_booking` does both. This
keeps the nightly rate intact across a book-then-cancel round trip and is why
ownership is tracked through `booking_rooms.booking_id`, not through a
`price IS NULL` heuristic.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, replace
from decimal import Decimal
from typing import TYPE_CHECKING, Any, LiteralString, cast

from psycopg import errors as psycopg_errors
from psycopg.rows import dict_row

if TYPE_CHECKING:
    import datetime as dt
    from collections.abc import Iterator, Sequence

    import psycopg
    from psycopg_pool import AsyncConnectionPool

logger = logging.getLogger(__name__)


class BookingWriteError(Exception):
    """Raised when a booking write cannot be completed as requested.

    Callers should treat this as a clean refusal: no partial writes are left
    behind, because every write_ops function performs its work inside one
    transaction and this exception is only ever raised before that
    transaction commits.
    """


@dataclass(frozen=True, slots=True)
class RoomRequest:
    """One room, date range request used to build a booking.

    Attributes:
        room_id: Target room's primary key.
        room_number: Target room's number, carried along for display so
            callers never need a second lookup to render a receipt.
        check_in: First night of the stay (inclusive).
        check_out: Departure date (exclusive of the final night).

    """

    room_id: int
    room_number: int
    check_in: dt.date
    check_out: dt.date


@dataclass(frozen=True, slots=True)
class PricedRoomStay:
    """A priced, contiguous stay in a single room.

    Attributes:
        booking_room_id: Primary key of the `booking_rooms` row, when one
            exists yet (unset -- ``None`` -- for a stay still being priced).
        room_id: Room's primary key.
        room_number: Room's number.
        check_in: First night of the stay (inclusive).
        check_out: Departure date (exclusive of the final night).
        total_amount: Sum of the nightly `price` across every night in range.

    """

    room_id: int
    room_number: int
    check_in: dt.date
    check_out: dt.date
    total_amount: Decimal
    booking_room_id: int | None = None


@dataclass(frozen=True, slots=True)
class CommitResult:
    """Outcome of a successful `commit_booking` call.

    Attributes:
        booking_id: Primary key of the new `bookings` row.
        confirmation_number: App-authored receipt identifier, ``BH######``.
        rooms: The priced room-stays that were booked.
        total_amount: Sum of every room-stay's `total_amount`.

    """

    booking_id: int
    confirmation_number: str
    rooms: tuple[PricedRoomStay, ...]
    total_amount: Decimal


@dataclass(frozen=True, slots=True)
class CancelRoomInstruction:
    """One trim-or-cancel instruction for a single `booking_rooms` row.

    Leaving both `new_check_in` and `new_check_out` unset cancels the room-stay
    entirely. Setting exactly one of them shrinks the stay from that end.
    Setting both, or setting a value that does not touch either original
    edge, is rejected: the `booking_rooms` shape is one contiguous range per
    row, so a mid-stay hole cannot be expressed without splitting it in two.

    Attributes:
        booking_room_id: Primary key of the `booking_rooms` row to modify.
        new_check_in: New (later) check-in date, trimming from the front.
        new_check_out: New (earlier) check-out date, trimming from the back.

    """

    booking_room_id: int
    new_check_in: dt.date | None = None
    new_check_out: dt.date | None = None


@dataclass(frozen=True, slots=True)
class CancelResult:
    """Outcome of a successful `cancel_booking` call.

    Attributes:
        booking_id: Primary key of the affected `bookings` row.
        refunded_amount: Sum of the `price` of every night released.
        fully_cancelled: True when no `booking_rooms` rows remain for this
            booking, in which case `bookings.status` was set to `'cancelled'`.

    """

    booking_id: int
    refunded_amount: Decimal
    fully_cancelled: bool


@dataclass(frozen=True, slots=True)
class ModifyRoomInstruction:
    """One release-and-reacquire instruction for a single `booking_rooms` row.

    Attributes:
        booking_room_id: Primary key of the `booking_rooms` row to replace.
        new_room_id: Room to book instead.
        new_room_number: Room number, carried along for display.
        new_check_in: First night of the replacement stay (inclusive).
        new_check_out: Departure date of the replacement stay (exclusive).

    """

    booking_room_id: int
    new_room_id: int
    new_room_number: int
    new_check_in: dt.date
    new_check_out: dt.date


@dataclass(frozen=True, slots=True)
class ModifyResult:
    """Outcome of a successful `modify_booking` call.

    Attributes:
        booking_id: Primary key of the affected `bookings` row.
        rooms: The priced room-stays after modification.
        total_amount: Sum of every room-stay's `total_amount` after
            modification.

    """

    booking_id: int
    rooms: tuple[PricedRoomStay, ...]
    total_amount: Decimal


@dataclass(frozen=True, slots=True)
class RoomStaySummary:
    """One room-stay within a guest's booking, for display.

    Attributes:
        booking_room_id: Primary key of the `booking_rooms` row.
        room_id: Room's primary key.
        room_number: Room's number.
        check_in: First night of the stay (inclusive).
        check_out: Departure date (exclusive of the final night).
        total_amount: Sum of the nightly `price` across the stay.

    """

    booking_room_id: int
    room_id: int
    room_number: int
    check_in: dt.date
    check_out: dt.date
    total_amount: Decimal


@dataclass(frozen=True, slots=True)
class CustomerSummary:
    """One guest available for the identity dropdown.

    Attributes:
        customer_id: Guest's primary key.
        first_name: Guest's first name.
        last_name: Guest's last name.

    """

    customer_id: int
    first_name: str
    last_name: str


@dataclass(frozen=True, slots=True)
class BookingSummary:
    """One booking and its room-stays, for display.

    Attributes:
        booking_id: Primary key of the `bookings` row.
        status: Current booking status (`confirmed`, `cancelled`, etc.).
        confirmation_number: App-authored receipt identifier, or ``None`` for
            a booking that was never committed through `commit_booking`.
        booked_at: Timestamp the booking was created.
        cancelled_at: Timestamp the booking was cancelled, or ``None``.
        rooms: This booking's room-stays.
        total_amount: Sum of every room-stay's `total_amount`.

    """

    booking_id: int
    status: str
    confirmation_number: str | None
    booked_at: dt.datetime
    cancelled_at: dt.datetime | None
    rooms: tuple[RoomStaySummary, ...]
    total_amount: Decimal


@dataclass(frozen=True, slots=True)
class _OwnedRange:
    """A room-stay range the caller already holds, priced as if released.

    Passed into `_price_one_room` so a modification preview can price a
    replacement range without requiring `status = 'Available'` on nights the
    guest already owns under the room-stay being replaced -- see
    `price_modification_room`.

    Attributes:
        room_id: Room id of the currently held stay.
        check_in: Start of the currently held range (inclusive).
        check_out: End of the currently held range (exclusive).

    """

    room_id: int
    check_in: dt.date
    check_out: dt.date


def _confirmation_number(booking_id: int) -> str:
    """Format the app-authored confirmation number for a booking.

    Args:
        booking_id: Primary key of the `bookings` row.

    Returns:
        str: Confirmation number of the form ``BH######``.

    """
    return f"BH{booking_id:06d}"


async def price_rooms(
    pool: AsyncConnectionPool[Any],
    rooms: Sequence[RoomRequest],
) -> list[PricedRoomStay]:
    """Price a set of room, date-range requests without locking or writing.

    Used by the `propose_booking` tool: a proposal reserves nothing, so this
    reads current availability and pricing without `FOR UPDATE`. The same
    nights are re-validated (this time under lock) inside `commit_booking`.

    Args:
        pool: Read-write booking database pool (`bh_agent_rw`).
        rooms: Room, date-range requests to price.

    Returns:
        list[PricedRoomStay]: One priced stay per request, in the same order.

    Raises:
        BookingWriteError: If any requested night is not `Available` with a
            price set.

    """
    async with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        return [await _price_one_room(cur, room, lock=False) for room in rooms]


async def price_modification_room(
    pool: AsyncConnectionPool[Any],
    *,
    existing_room_id: int,
    existing_check_in: dt.date,
    existing_check_out: dt.date,
    request: RoomRequest,
) -> PricedRoomStay:
    """Price a modification's replacement range without locking or writing.

    Used by the `propose_modification` tool. `modify_booking` releases the
    existing room-stay before pricing its replacement, so a same-room
    extension or any new range overlapping the guest's current nights prices
    cleanly at commit time. A preview must not write, so it cannot release
    first; instead, nights within `[existing_check_in, existing_check_out)`
    on `existing_room_id` are priced as already held by the guest rather than
    requiring `status = 'Available'`, mirroring that release-then-reacquire
    order without an intervening write.

    Args:
        pool: Read-write booking database pool (`bh_agent_rw`).
        existing_room_id: Room id of the room-stay being replaced.
        existing_check_in: Start of the currently held range (inclusive).
        existing_check_out: End of the currently held range (exclusive).
        request: Replacement room, date-range request to price.

    Returns:
        PricedRoomStay: The replacement stay with its summed total.

    Raises:
        BookingWriteError: If any replacement night is neither `Available`
            nor already held by this room-stay, or has no price set.

    """
    owned = _OwnedRange(
        room_id=existing_room_id,
        check_in=existing_check_in,
        check_out=existing_check_out,
    )
    async with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        return await _price_one_room(cur, request, lock=False, owned=owned)


async def commit_booking(
    pool: AsyncConnectionPool[Any],
    *,
    customer_id: int,
    rooms: Sequence[RoomRequest],
) -> CommitResult:
    """Book every requested room-night, all-or-nothing, in one transaction.

    Args:
        pool: Read-write booking database pool (`bh_agent_rw`).
        customer_id: Server-injected identity of the booking guest.
        rooms: Room, date-range requests to book.

    Returns:
        CommitResult: The new booking's id, confirmation number, and priced
        room-stays.

    Raises:
        BookingWriteError: If any requested night is no longer `Available`
            with a price set, or the insert would violate `booking_rooms`'s
            no-overlap exclusion constraint (only reachable if
            `room_availability` and `booking_rooms` have drifted out of
            sync, since the `FOR UPDATE` lock above already re-validates
            availability). No partial writes are made either way.

    """
    async with pool.connection() as conn:
        async with conn.transaction(), conn.cursor(row_factory=dict_row) as cur:
            priced = [
                await _price_one_room(cur, room, lock=True) for room in rooms
            ]

            booking_id = await _insert_booking(cur, customer_id=customer_id)
            for stay in priced:
                await _lock_and_book_nights(cur, stay)
                with _refuse_on_overlap(stay.room_number):
                    await cur.execute(
                        """
                        INSERT INTO booking_rooms
                            (booking_id, room_id, check_in, check_out, total_amount)
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        (
                            booking_id,
                            stay.room_id,
                            stay.check_in,
                            stay.check_out,
                            stay.total_amount,
                        ),
                    )

            confirmation_number = _confirmation_number(booking_id)
            await cur.execute(
                "UPDATE bookings SET confirmation_number = %s WHERE booking_id = %s",
                (confirmation_number, booking_id),
            )

        total_amount = sum((stay.total_amount for stay in priced), Decimal("0.00"))
        return CommitResult(
            booking_id=booking_id,
            confirmation_number=confirmation_number,
            rooms=tuple(priced),
            total_amount=total_amount,
        )


async def cancel_booking(
    pool: AsyncConnectionPool[Any],
    *,
    customer_id: int,
    booking_id: int,
    rooms: Sequence[CancelRoomInstruction] | None = None,
) -> CancelResult:
    """Cancel whole room-stays, or shrink one from either end, in one transaction.

    Args:
        pool: Read-write booking database pool (`bh_agent_rw`).
        customer_id: Server-injected identity; must own `booking_id`.
        booking_id: Booking to cancel or shrink.
        rooms: Per-room-stay instructions. ``None`` cancels every room-stay on
            the booking outright.

    Returns:
        CancelResult: Refunded amount and whether the booking is now fully
        cancelled.

    Raises:
        BookingWriteError: If the booking does not belong to `customer_id`,
            is already cancelled, references an unknown `booking_room_id`, or
            any instruction would leave a mid-stay hole. No partial writes
            are made.

    """
    async with pool.connection() as conn, conn.transaction(), conn.cursor(
        row_factory=dict_row,
    ) as cur:
        existing = await _lock_booking_rooms(
            cur, booking_id=booking_id, customer_id=customer_id,
        )

        instructions = rooms if rooms is not None else [
            CancelRoomInstruction(booking_room_id=row["booking_room_id"])
            for row in existing.values()
        ]

        refunded = Decimal("0.00")
        for instruction in instructions:
            row = existing.get(instruction.booking_room_id)
            if row is None:
                msg = (
                    f"booking_room_id {instruction.booking_room_id} "
                    "is not on this booking."
                )
                raise BookingWriteError(msg)
            refunded += await _apply_cancel_instruction(cur, row, instruction)

        await cur.execute(
            "SELECT COUNT(*) AS remaining FROM booking_rooms WHERE booking_id = %s",
            (booking_id,),
        )
        remaining = cast("dict[str, Any]", await cur.fetchone())["remaining"]
        fully_cancelled = remaining == 0
        if fully_cancelled:
            await cur.execute(
                """
                UPDATE bookings SET status = 'cancelled', cancelled_at = now()
                WHERE booking_id = %s
                """,
                (booking_id,),
            )

    return CancelResult(
        booking_id=booking_id,
        refunded_amount=refunded,
        fully_cancelled=fully_cancelled,
    )


async def modify_booking(
    pool: AsyncConnectionPool[Any],
    *,
    customer_id: int,
    booking_id: int,
    changes: Sequence[ModifyRoomInstruction],
) -> ModifyResult:
    """Release and reacquire room-stays atomically; nothing changes unless all succeed.

    Args:
        pool: Read-write booking database pool (`bh_agent_rw`).
        customer_id: Server-injected identity; must own `booking_id`.
        booking_id: Booking to modify.
        changes: Replacement room, date-range requests, one per existing
            `booking_rooms` row being changed.

    Returns:
        ModifyResult: The booking's priced room-stays after modification.

    Raises:
        BookingWriteError: If the booking does not belong to `customer_id`, a
            `booking_room_id` is unknown, any replacement night is not
            `Available` with a price set, or the replacement would violate
            `booking_rooms`'s no-overlap exclusion constraint (only
            reachable if `room_availability` and `booking_rooms` have
            drifted out of sync, since the `FOR UPDATE` lock above already
            re-validates availability). No partial writes are made either
            way.

    """
    async with pool.connection() as conn, conn.transaction(), conn.cursor(
        row_factory=dict_row,
    ) as cur:
        existing = await _lock_booking_rooms(
            cur, booking_id=booking_id, customer_id=customer_id,
        )

        for change in changes:
            row = existing.get(change.booking_room_id)
            if row is None:
                msg = (
                    f"booking_room_id {change.booking_room_id} "
                    "is not on this booking."
                )
                raise BookingWriteError(msg)
            await _release_nights(
                cur,
                room_id=row["room_id"],
                check_in=row["check_in"],
                check_out=row["check_out"],
            )

        for change in changes:
            request = RoomRequest(
                room_id=change.new_room_id,
                room_number=change.new_room_number,
                check_in=change.new_check_in,
                check_out=change.new_check_out,
            )
            stay = await _price_one_room(cur, request, lock=True)
            await _lock_and_book_nights(cur, stay)
            with _refuse_on_overlap(stay.room_number):
                await cur.execute(
                    """
                    UPDATE booking_rooms
                    SET room_id = %s, check_in = %s, check_out = %s, total_amount = %s
                    WHERE booking_room_id = %s
                    """,
                    (
                        stay.room_id,
                        stay.check_in,
                        stay.check_out,
                        stay.total_amount,
                        change.booking_room_id,
                    ),
                )

        await cur.execute(
            """
            SELECT br.booking_room_id, br.room_id, r.room_number,
                   br.check_in, br.check_out, br.total_amount
            FROM booking_rooms br
            JOIN rooms r ON r.room_id = br.room_id
            WHERE br.booking_id = %s
            ORDER BY br.booking_room_id
            """,
            (booking_id,),
        )
        rows = cast("list[dict[str, Any]]", await cur.fetchall())

    rooms_out = tuple(
        PricedRoomStay(
            booking_room_id=row["booking_room_id"],
            room_id=row["room_id"],
            room_number=row["room_number"],
            check_in=row["check_in"],
            check_out=row["check_out"],
            total_amount=row["total_amount"],
        )
        for row in rows
    )
    total_amount = sum((stay.total_amount for stay in rooms_out), Decimal("0.00"))
    return ModifyResult(
        booking_id=booking_id, rooms=rooms_out, total_amount=total_amount,
    )


async def list_bookings(
    pool: AsyncConnectionPool[Any],
    *,
    customer_id: int,
) -> list[BookingSummary]:
    """List every booking belonging to one guest, most recent first.

    Args:
        pool: Read-write booking database pool (`bh_agent_rw`).
        customer_id: Server-injected identity of the requesting guest.

    Returns:
        list[BookingSummary]: One entry per booking, each carrying its
        room-stays. Bookings with no remaining `booking_rooms` rows (fully
        cancelled) are still included, with an empty `rooms` tuple.

    """
    async with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(
            """
            SELECT booking_id, status, confirmation_number, booked_at, cancelled_at
            FROM bookings
            WHERE customer_id = %s
            ORDER BY booked_at DESC
            """,
            (customer_id,),
        )
        booking_rows = cast("list[dict[str, Any]]", await cur.fetchall())

        await cur.execute(
            """
            SELECT br.booking_id, br.booking_room_id, br.room_id, r.room_number,
                   br.check_in, br.check_out, br.total_amount
            FROM booking_rooms br
            JOIN rooms r ON r.room_id = br.room_id
            JOIN bookings b ON b.booking_id = br.booking_id
            WHERE b.customer_id = %s
            ORDER BY br.check_in
            """,
            (customer_id,),
        )
        room_rows = cast("list[dict[str, Any]]", await cur.fetchall())

    rooms_by_booking: dict[int, list[RoomStaySummary]] = {}
    for row in room_rows:
        rooms_by_booking.setdefault(row["booking_id"], []).append(
            RoomStaySummary(
                booking_room_id=row["booking_room_id"],
                room_id=row["room_id"],
                room_number=row["room_number"],
                check_in=row["check_in"],
                check_out=row["check_out"],
                total_amount=row["total_amount"],
            ),
        )

    summaries: list[BookingSummary] = []
    for row in booking_rows:
        stays = tuple(rooms_by_booking.get(row["booking_id"], []))
        total_amount = sum((stay.total_amount for stay in stays), Decimal("0.00"))
        summaries.append(
            BookingSummary(
                booking_id=row["booking_id"],
                status=row["status"],
                confirmation_number=row["confirmation_number"],
                booked_at=row["booked_at"],
                cancelled_at=row["cancelled_at"],
                rooms=stays,
                total_amount=total_amount,
            ),
        )
    return summaries


async def list_customers(pool: AsyncConnectionPool[Any]) -> list[CustomerSummary]:
    """List every guest available for the identity dropdown.

    Args:
        pool: Read-write booking database pool (`bh_agent_rw`).

    Returns:
        list[CustomerSummary]: All seeded guests, ordered by id.

    """
    async with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(
            "SELECT customer_id, first_name, last_name "
            "FROM customers ORDER BY customer_id",
        )
        rows = cast("list[dict[str, Any]]", await cur.fetchall())
    return [CustomerSummary(**row) for row in rows]


def fmt_money(amount: Decimal) -> str:
    """Format a decimal amount as a fixed-2-decimal string for display.

    Args:
        amount: Amount to format.

    Returns:
        str: Amount formatted to exactly two decimal places.

    """
    return f"{amount:.2f}"


def serialize_priced_stay(stay: PricedRoomStay) -> dict[str, Any]:
    """Convert a priced room-stay into JSON-safe display fields.

    Args:
        stay: Priced room-stay to serialize.

    Returns:
        dict[str, Any]: `room_number`, `check_in`, `check_out`, `nights`, and
        `amount` (a fixed-2-decimal string).

    """
    nights = (stay.check_out - stay.check_in).days
    return {
        "room_number": stay.room_number,
        "check_in": stay.check_in.isoformat(),
        "check_out": stay.check_out.isoformat(),
        "nights": nights,
        "amount": fmt_money(stay.total_amount),
    }


def serialize_room_stay(stay: RoomStaySummary) -> dict[str, Any]:
    """Convert an existing room-stay into JSON-safe display fields.

    Args:
        stay: Room-stay summary to serialize.

    Returns:
        dict[str, Any]: `booking_room_id`, `room_number`, `check_in`,
        `check_out`, and `total_amount` (a fixed-2-decimal string).

    """
    return {
        "booking_room_id": stay.booking_room_id,
        "room_number": stay.room_number,
        "check_in": stay.check_in.isoformat(),
        "check_out": stay.check_out.isoformat(),
        "total_amount": fmt_money(stay.total_amount),
    }


def serialize_booking(booking: BookingSummary) -> dict[str, Any]:
    """Convert a booking summary into JSON-safe display fields.

    Args:
        booking: Booking summary to serialize.

    Returns:
        dict[str, Any]: Booking fields plus a serialized `rooms` list.

    """
    return {
        "booking_id": booking.booking_id,
        "status": booking.status,
        "confirmation_number": booking.confirmation_number,
        "total_amount": fmt_money(booking.total_amount),
        "rooms": [serialize_room_stay(room) for room in booking.rooms],
    }


async def _price_one_room(
    cur: psycopg.AsyncCursor[dict[str, Any]],
    room: RoomRequest,
    *,
    lock: bool,
    owned: _OwnedRange | None = None,
) -> PricedRoomStay:
    """Read and sum the nightly prices for one room over one date range.

    Args:
        cur: Open cursor inside the caller's transaction (or an ad hoc
            connection for `lock=False`).
        room: Room, date-range request to price.
        lock: Whether to take `FOR UPDATE` row locks (commit path) or read
            without locking (proposal path).
        owned: A room-stay the caller already holds and will release as part
            of the same operation, or `None`. Nights that fall inside its
            range on its room are priced without requiring
            `status = 'Available'`, since they read back as `Booked` under
            the guest's own reservation until that release happens. Used by
            modification previews, which must not write, to mirror
            `modify_booking`'s release-then-reacquire order without an
            intervening release.

    Returns:
        PricedRoomStay: The requested stay with its summed total.

    Raises:
        BookingWriteError: If any night in range is missing, or is neither
            `Available` nor already held by the caller under `owned`, or has
            no price set.

    """
    nights = (room.check_out - room.check_in).days
    if nights <= 0:
        msg = "check_out must be after check_in."
        raise BookingWriteError(msg)

    query = (
        "SELECT id, date, status, price FROM room_availability "
        "WHERE room_id = %s AND date >= %s AND date < %s"
    )
    if lock:
        query += " ORDER BY date FOR UPDATE"
    else:
        query += " ORDER BY date"

    await cur.execute(
        cast("LiteralString", query),
        (room.room_id, room.check_in, room.check_out),
    )
    found = cast("list[dict[str, Any]]", await cur.fetchall())

    if len(found) != nights:
        msg = f"Room {room.room_number} is not available for every night requested."
        raise BookingWriteError(msg)

    total = Decimal("0.00")
    for night in found:
        already_owned = (
            owned is not None
            and owned.room_id == room.room_id
            and owned.check_in <= night["date"] < owned.check_out
        )
        if night["price"] is None or (
            night["status"] != "Available" and not already_owned
        ):
            msg = f"Room {room.room_number} is not available for every night requested."
            raise BookingWriteError(msg)
        total += night["price"]

    return PricedRoomStay(
        room_id=room.room_id,
        room_number=room.room_number,
        check_in=room.check_in,
        check_out=room.check_out,
        total_amount=total,
    )


@contextmanager
def _refuse_on_overlap(room_number: int) -> Iterator[None]:
    """Translate a `booking_rooms` exclusion-constraint hit into `BookingWriteError`.

    `booking_rooms_no_overlap` (see `setup_booking_rooms_schema`) is the
    database's own backstop against two rows for one room holding
    overlapping nights. `_price_one_room`'s `FOR UPDATE` lock on
    `room_availability` is expected to catch every real double-booking
    attempt first, so this constraint should only ever fire if that table
    and `booking_rooms` have drifted out of sync -- but if it does fire, the
    guest should see the same clean refusal as any other unavailable-night
    rejection, not a raw `psycopg` error.

    Args:
        room_number: Room number to name in the refusal message.

    Yields:
        None. Wrap exactly one `booking_rooms` insert or update statement.

    Raises:
        BookingWriteError: If the wrapped statement violates
            `booking_rooms_no_overlap`.

    """
    try:
        yield
    except psycopg_errors.ExclusionViolation as exc:
        msg = f"Room {room_number} is not available for every night requested."
        raise BookingWriteError(msg) from exc


async def _lock_and_book_nights(
    cur: psycopg.AsyncCursor[dict[str, Any]],
    stay: PricedRoomStay,
) -> None:
    """Mark every night of a priced stay `Booked`.

    Args:
        cur: Open cursor inside the caller's transaction. The rows were
            already locked and validated by `_price_one_room(..., lock=True)`.
        stay: The priced stay whose nights should be marked `Booked`.

    """
    await cur.execute(
        """
        UPDATE room_availability
        SET status = 'Booked'
        WHERE room_id = %s AND date >= %s AND date < %s
        """,
        (stay.room_id, stay.check_in, stay.check_out),
    )


async def _release_nights(
    cur: psycopg.AsyncCursor[dict[str, Any]],
    *,
    room_id: int,
    check_in: dt.date,
    check_out: dt.date,
) -> None:
    """Return a range of nights to `Available`, leaving `price` untouched.

    Args:
        cur: Open cursor inside the caller's transaction.
        room_id: Room whose nights are being released.
        check_in: First night to release (inclusive).
        check_out: End of the range to release (exclusive).

    """
    await cur.execute(
        """
        UPDATE room_availability
        SET status = 'Available'
        WHERE room_id = %s AND date >= %s AND date < %s
        """,
        (room_id, check_in, check_out),
    )


async def _insert_booking(
    cur: psycopg.AsyncCursor[dict[str, Any]],
    *,
    customer_id: int,
) -> int:
    """Insert a new `bookings` row and return its id.

    Args:
        cur: Open cursor inside the caller's transaction.
        customer_id: Guest the booking belongs to.

    Returns:
        int: The new booking's primary key.

    """
    await cur.execute(
        "INSERT INTO bookings (customer_id) VALUES (%s) RETURNING booking_id",
        (customer_id,),
    )
    row = cast("dict[str, Any]", await cur.fetchone())
    return cast("int", row["booking_id"])


async def _lock_booking_rooms(
    cur: psycopg.AsyncCursor[dict[str, Any]],
    *,
    booking_id: int,
    customer_id: int,
) -> dict[int, dict[str, Any]]:
    """Lock and load a booking's room-stays, verifying ownership and status.

    Args:
        cur: Open cursor inside the caller's transaction.
        booking_id: Booking to load.
        customer_id: Expected owner of the booking.

    Returns:
        dict[int, dict[str, Any]]: Room-stay rows keyed by `booking_room_id`.

    Raises:
        BookingWriteError: If the booking does not exist, belongs to a
            different guest, or is already cancelled.

    """
    await cur.execute(
        "SELECT customer_id, status FROM bookings WHERE booking_id = %s FOR UPDATE",
        (booking_id,),
    )
    booking_row = await cur.fetchone()
    if booking_row is None:
        msg = f"Booking {booking_id} does not exist."
        raise BookingWriteError(msg)
    if booking_row["customer_id"] != customer_id:
        msg = "This booking does not belong to this guest."
        raise BookingWriteError(msg)
    if booking_row["status"] == "cancelled":
        msg = f"Booking {booking_id} is already cancelled."
        raise BookingWriteError(msg)

    await cur.execute(
        """
        SELECT booking_room_id, room_id, check_in, check_out, total_amount
        FROM booking_rooms
        WHERE booking_id = %s
        FOR UPDATE
        """,
        (booking_id,),
    )
    rows = cast("list[dict[str, Any]]", await cur.fetchall())
    return {row["booking_room_id"]: row for row in rows}


async def _apply_cancel_instruction(
    cur: psycopg.AsyncCursor[dict[str, Any]],
    row: dict[str, Any],
    instruction: CancelRoomInstruction,
) -> Decimal:
    """Apply one cancel-or-trim instruction to a locked `booking_rooms` row.

    Args:
        cur: Open cursor inside the caller's transaction.
        row: The locked `booking_rooms` row (`room_id`, `check_in`,
            `check_out`, `total_amount`).
        instruction: The requested cancellation or trim.

    Returns:
        Decimal: The amount refunded by this instruction.

    Raises:
        BookingWriteError: If the instruction would leave a mid-stay hole
            (neither edge of the original range is preserved, or both are
            changed at once).

    """
    check_in, check_out = row["check_in"], row["check_out"]
    instruction = normalize_cancel_instruction(
        instruction, check_in=check_in, check_out=check_out,
    )

    if instruction.new_check_in is None and instruction.new_check_out is None:
        await _release_nights(
            cur, room_id=row["room_id"], check_in=check_in, check_out=check_out,
        )
        await cur.execute(
            "DELETE FROM booking_rooms WHERE booking_room_id = %s",
            (row["booking_room_id"],),
        )
        return cast("Decimal", row["total_amount"])

    if instruction.new_check_in is not None and instruction.new_check_out is not None:
        msg = "Cancelling a range from both ends at once is not supported."
        raise BookingWriteError(msg)

    if instruction.new_check_in is not None:
        new_in = instruction.new_check_in
        if not (check_in < new_in < check_out):
            msg = "Trim start must fall strictly within the existing stay."
            raise BookingWriteError(msg)
        released_start, released_end = check_in, new_in
        remaining_in, remaining_out = new_in, check_out
    else:
        new_out = cast("dt.date", instruction.new_check_out)
        if not (check_in < new_out < check_out):
            msg = "Trim end must fall strictly within the existing stay."
            raise BookingWriteError(msg)
        released_start, released_end = new_out, check_out
        remaining_in, remaining_out = check_in, new_out

    await cur.execute(
        """
        SELECT COALESCE(SUM(price), 0) AS refund
        FROM room_availability
        WHERE room_id = %s AND date >= %s AND date < %s
        """,
        (row["room_id"], released_start, released_end),
    )
    refund_row = cast("dict[str, Any]", await cur.fetchone())
    refund = cast("Decimal", refund_row["refund"])

    await _release_nights(
        cur, room_id=row["room_id"], check_in=released_start, check_out=released_end,
    )
    await cur.execute(
        """
        UPDATE booking_rooms
        SET check_in = %s, check_out = %s, total_amount = total_amount - %s
        WHERE booking_room_id = %s
        """,
        (remaining_in, remaining_out, refund, row["booking_room_id"]),
    )
    return refund


def normalize_cancel_instruction(
    instruction: CancelRoomInstruction,
    *,
    check_in: dt.date,
    check_out: dt.date,
) -> CancelRoomInstruction:
    """Drop a trim edge that merely echoes the stay's current value.

    A caller describing "shorten to <check-in> through <new check-out>" (or
    the mirror case for a later check-in) naturally repeats the unchanged
    edge alongside the edge that actually moved -- the instruction then
    carries both `new_check_in` and `new_check_out`, even though only one
    edge is a real change. Only two edges that both genuinely differ from
    the stay's current dates describe an actual both-ends trim, which stays
    rejected: the `booking_rooms` shape is one contiguous range per row, so
    a mid-stay hole cannot be expressed without splitting it in two. This
    normalization runs before that check on both the propose-time preview
    (`factory._refund_preview`) and the commit path
    (`_apply_cancel_instruction`), so a caller is never rejected merely for
    restating a date it did not intend to change.

    Args:
        instruction: The requested trim/cancel instruction, as supplied.
        check_in: The room-stay's current check-in date.
        check_out: The room-stay's current check-out date.

    Returns:
        CancelRoomInstruction: `instruction`, or a copy with any edge equal
        to its current value cleared.

    """
    new_check_in = instruction.new_check_in
    new_check_out = instruction.new_check_out
    if new_check_in == check_in:
        new_check_in = None
    if new_check_out == check_out:
        new_check_out = None
    if new_check_in == instruction.new_check_in and (
        new_check_out == instruction.new_check_out
    ):
        return instruction
    return replace(instruction, new_check_in=new_check_in, new_check_out=new_check_out)
