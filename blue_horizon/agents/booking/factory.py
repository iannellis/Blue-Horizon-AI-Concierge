"""Factory for building the booking SQL agent.

Every tool built here is read-only or propose-only. Nothing the model calls
mutates `bookings`, `booking_rooms`, or `room_availability` -- that authority
belongs entirely to `blue_horizon.agents.booking.write_ops`, reached only
through a human confirming a proposal via `POST /v1/booking/confirm`.
`customer_id` and `thread_id` arrive through the injected `RunnableConfig`,
which LangChain excludes from the model-facing tool schema: the model can
neither see nor set them.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Required, TypedDict

from langchain.agents import create_agent

# Kept out of the TYPE_CHECKING block despite TC002: `@tool` resolves its
# schema at runtime via `get_type_hints()`, which needs `RunnableConfig` to
# exist in this module's globals -- a TYPE_CHECKING-only import raises
# `NameError` the first time a tool with a `config: RunnableConfig`
# parameter is built.
from langchain_core.runnables import RunnableConfig  # noqa: TC002
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from psycopg.rows import dict_row

from blue_horizon.agents.booking import write_ops

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph
    from psycopg_pool import AsyncConnectionPool

    from blue_horizon.agents.booking.resources import BookingSqlResources
    from blue_horizon.config import BookingSqlConfig


class BookingRoomRequest(TypedDict):
    """One room, date-range request as supplied by the model.

    Attributes:
        room_number: Room the guest wants.
        check_in: First night of the stay, ISO ``YYYY-MM-DD``.
        check_out: Departure date, ISO ``YYYY-MM-DD``, exclusive of the final
            night.

    """

    room_number: int
    check_in: str
    check_out: str


class CancelRoomRequest(TypedDict, total=False):
    """One trim-or-cancel instruction as supplied by the model.

    Attributes:
        booking_room_id: Row to modify, from a prior `list_my_bookings` call.
        new_check_in: New (later) check-in date, ISO ``YYYY-MM-DD``, trimming
            from the front. Omit to leave the front untouched.
        new_check_out: New (earlier) check-out date, ISO ``YYYY-MM-DD``,
            trimming from the back. Omit to leave the back untouched.

    """

    booking_room_id: Required[int]
    new_check_in: str
    new_check_out: str


class ModifyRoomRequest(TypedDict):
    """One room-stay replacement as supplied by the model.

    Attributes:
        booking_room_id: Row to replace, from a prior `list_my_bookings` call.
        new_room_number: Room to book instead.
        new_check_in: First night of the replacement stay, ISO ``YYYY-MM-DD``.
        new_check_out: Departure date of the replacement stay, ISO
            ``YYYY-MM-DD``, exclusive of the final night.

    """

    booking_room_id: int
    new_room_number: int
    new_check_in: str
    new_check_out: str


def _identity_from_config(config: RunnableConfig) -> tuple[int, str]:
    """Read the server-injected customer and thread identity from tool config.

    Args:
        config: `RunnableConfig` LangChain injects into the tool call, never
            exposed to the model.

    Returns:
        tuple[int, str]: `(customer_id, thread_id)`.

    Raises:
        RuntimeError: If either value is missing, which indicates a wiring
            bug upstream (the API layer failed to set them), not a request
            the model could have caused.

    """
    configurable = config.get("configurable", {})
    customer_id = configurable.get("customer_id")
    thread_id = configurable.get("thread_id")
    if customer_id is None or thread_id is None:
        msg = "Booking tool invoked without a server-injected customer_id/thread_id."
        raise RuntimeError(msg)
    return customer_id, thread_id


class BookingAgentFactory:
    """Build the booking agent using initialized resources.

    Attributes:
        config: Parsed configuration.
        resources: Initialized resources instance.

    """

    __slots__ = ("config", "resources")

    config: BookingSqlConfig
    resources: BookingSqlResources

    def __init__(
        self,
        *,
        config: BookingSqlConfig,
        resources: BookingSqlResources,
    ) -> None:
        """Construct the booking agent factory.

        Args:
            config: Parsed configuration loaded from TOML.
            resources: Initialized resources instance.

        """
        self.config = config
        self.resources = resources

    def build(self) -> CompiledStateGraph:
        """Build and return a compiled booking agent.

        Returns:
            Compiled LangGraph agent.

        Raises:
            RuntimeError: If resources were not initialized.

        """
        system_prompt = self.resources.get_system_prompt()
        resources = self.resources

        llm = ChatOpenAI(
            model=self.config.llm.model,
            reasoning={"effort": self.config.llm.reasoning_effort},
            timeout=self.config.llm.timeout_s,
            max_retries=self.config.llm.max_retries,
        )

        @tool(parse_docstring=True)
        async def run_sql(query: str) -> dict[str, Any]:
            """Execute a single read-only SQL statement and return rows.

            Args:
                query: One SELECT statement (no semicolons).

            Returns:
                Dict with keys:
                  - status: str ("ok" or "error")
                  - rows: list[dict[str, Any]]
                  - truncated: bool
                  - rowcount: int
                  - error: str (only present on failure)

            """
            return await resources.execute_sql(query)

        @tool(parse_docstring=True)
        async def list_my_bookings(config: RunnableConfig) -> dict[str, Any]:
            """List the current guest's bookings, including cancelled ones.

            Use this to find `booking_id` and `booking_room_id` values before
            proposing a cancellation or modification.

            Args:
                config: Server-injected identity, hidden from the model.

            Returns:
                Dict with key `bookings`: a list of booking summaries, each
                with `booking_id`, `status`, `confirmation_number`, `rooms`
                (each with `booking_room_id`, `room_number`, `check_in`,
                `check_out`, `total_amount`), and `total_amount`.

            """
            customer_id, _ = _identity_from_config(config)
            bookings = await write_ops.list_bookings(
                resources.get_write_pool(), customer_id=customer_id,
            )
            return {"bookings": [write_ops.serialize_booking(b) for b in bookings]}

        @tool(parse_docstring=True)
        async def propose_booking(
            rooms: list[BookingRoomRequest],
            config: RunnableConfig,
        ) -> dict[str, Any]:
            """Price a booking and store it for the guest to review and confirm.

            This does not book anything. The application shows the guest a
            confirmation dialog built from the prices this call computes; the
            booking only happens if they click Confirm.

            Args:
                rooms: Rooms and date ranges to book.
                config: Server-injected identity, hidden from the model.

            Returns:
                Dict with `proposal_id` on success, or `status: "error"` and
                `error` if any requested night is unavailable.

            """
            customer_id, thread_id = _identity_from_config(config)
            try:
                requests = [
                    write_ops.RoomRequest(
                        room_id=await _room_id_for_number(
                            resources.get_write_pool(), r["room_number"],
                        ),
                        room_number=r["room_number"],
                        check_in=dt.date.fromisoformat(r["check_in"]),
                        check_out=dt.date.fromisoformat(r["check_out"]),
                    )
                    for r in rooms
                ]
                priced = await write_ops.price_rooms(
                    resources.get_write_pool(), requests,
                )
            except write_ops.BookingWriteError as exc:
                return {"status": "error", "error": str(exc)}

            total = sum((stay.total_amount for stay in priced), Decimal("0.00"))
            summary = {
                "rooms": [write_ops.serialize_priced_stay(stay) for stay in priced],
                "total": write_ops.fmt_money(total),
            }
            proposal = resources.proposals.create(
                thread_id=thread_id,
                customer_id=customer_id,
                action="book",
                summary=summary,
                details=requests,
            )
            return {"proposal_id": proposal.proposal_id, "status": "proposed"}

        @tool(parse_docstring=True)
        async def propose_cancellation(
            booking_id: int,
            config: RunnableConfig,
            rooms: list[CancelRoomRequest] | None = None,
        ) -> dict[str, Any]:
            """Price a cancellation (or partial trim) and store it for review.

            This does not cancel anything. A stay can be cancelled outright
            (omit `rooms`) or shortened from either end. A gap in the middle
            of a stay is not supported.

            Args:
                booking_id: Booking to cancel or shrink, from
                    `list_my_bookings`.
                config: Server-injected identity, hidden from the model.
                rooms: Per-room-stay trim instructions. Omit to cancel every
                    room-stay on the booking outright.

            Returns:
                Dict with `proposal_id` on success, or `status: "error"` and
                `error` if the booking is not found or a trim is invalid.

            """
            customer_id, thread_id = _identity_from_config(config)
            try:
                booking = await _find_owned_booking(
                    resources.get_write_pool(),
                    customer_id=customer_id,
                    booking_id=booking_id,
                )
                instructions, summary_rooms, total = await _price_cancellation(
                    resources.get_write_pool(), booking=booking, rooms=rooms,
                )
            except write_ops.BookingWriteError as exc:
                return {"status": "error", "error": str(exc)}

            summary = {"rooms": summary_rooms, "total": write_ops.fmt_money(total)}
            proposal = resources.proposals.create(
                thread_id=thread_id,
                customer_id=customer_id,
                action="cancel",
                summary=summary,
                details=(booking_id, instructions),
            )
            return {"proposal_id": proposal.proposal_id, "status": "proposed"}

        @tool(parse_docstring=True)
        async def propose_modification(
            booking_id: int,
            changes: list[ModifyRoomRequest],
            config: RunnableConfig,
        ) -> dict[str, Any]:
            """Price a room/date change and store it for the guest to review.

            This does not change anything. Each room-stay is released and
            reacquired together: nothing changes unless every replacement is
            available.

            Args:
                booking_id: Booking to modify, from `list_my_bookings`.
                changes: Replacement room and date range for each room-stay
                    being changed.
                config: Server-injected identity, hidden from the model.

            Returns:
                Dict with `proposal_id` on success, or `status: "error"` and
                `error` if the booking is not found or a replacement is
                unavailable.

            """
            customer_id, thread_id = _identity_from_config(config)
            try:
                booking = await _find_owned_booking(
                    resources.get_write_pool(),
                    customer_id=customer_id,
                    booking_id=booking_id,
                )
                instructions, summary_changes, total = await _price_modification(
                    resources.get_write_pool(), booking=booking, changes=changes,
                )
            except write_ops.BookingWriteError as exc:
                return {"status": "error", "error": str(exc)}

            summary = {"changes": summary_changes, "total": write_ops.fmt_money(total)}
            proposal = resources.proposals.create(
                thread_id=thread_id,
                customer_id=customer_id,
                action="modify",
                summary=summary,
                details=(booking_id, instructions),
            )
            return {"proposal_id": proposal.proposal_id, "status": "proposed"}

        return create_agent(
            model=llm,
            tools=[
                run_sql,
                list_my_bookings,
                propose_booking,
                propose_cancellation,
                propose_modification,
            ],
            system_prompt=system_prompt,
        )


async def _room_id_for_number(pool: Any, room_number: int) -> int:  # noqa: ANN401
    """Resolve a guest-facing room number into its internal room_id.

    Args:
        pool: Read-write booking database pool (`bh_agent_rw`).
        room_number: Room number as the guest/model refers to it.

    Returns:
        int: The room's primary key.

    Raises:
        write_ops.BookingWriteError: If no room has that number.

    """
    async with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(
            "SELECT room_id FROM rooms WHERE room_number = %s", (room_number,),
        )
        row = await cur.fetchone()
    if row is None:
        msg = f"Room {room_number} does not exist."
        raise write_ops.BookingWriteError(msg)
    return row["room_id"]


async def _find_owned_booking(
    pool: AsyncConnectionPool[Any],
    *,
    customer_id: int,
    booking_id: int,
) -> write_ops.BookingSummary:
    """Load one of a guest's bookings by id, refusing bookings they don't own.

    Args:
        pool: Read-write booking database pool (`bh_agent_rw`).
        customer_id: Guest requesting the booking.
        booking_id: Booking to look up.

    Returns:
        write_ops.BookingSummary: The matching booking.

    Raises:
        write_ops.BookingWriteError: If no booking with that id belongs to
            this guest.

    """
    bookings = await write_ops.list_bookings(pool, customer_id=customer_id)
    for booking in bookings:
        if booking.booking_id == booking_id:
            return booking
    msg = f"Booking {booking_id} does not belong to this guest."
    raise write_ops.BookingWriteError(msg)


async def _price_cancellation(
    pool: AsyncConnectionPool[Any],
    *,
    booking: write_ops.BookingSummary,
    rooms: list[CancelRoomRequest] | None,
) -> tuple[list[write_ops.CancelRoomInstruction] | None, list[dict[str, Any]], Decimal]:
    """Price a cancellation without locking or writing, for the confirmation dialog.

    Args:
        pool: Read-write booking database pool (`bh_agent_rw`).
        booking: The guest's existing booking.
        rooms: Requested trim instructions, or ``None`` for a full cancel.

    Returns:
        tuple: `(instructions, summary_rooms, total_refund)`, where
        `instructions` is the payload `write_ops.cancel_booking` expects.

    Raises:
        write_ops.BookingWriteError: If a referenced `booking_room_id` is not
            on this booking, or a trim does not touch an original edge.

    """
    existing = {room.booking_room_id: room for room in booking.rooms}

    if rooms is None:
        summary_rooms = [
            write_ops.serialize_priced_stay(_as_priced(room)) for room in booking.rooms
        ]
        return None, summary_rooms, booking.total_amount

    instructions: list[write_ops.CancelRoomInstruction] = []
    summary_rooms = []
    total = Decimal("0.00")
    for request in rooms:
        row = existing.get(request["booking_room_id"])
        if row is None:
            msg = (
                f"booking_room_id {request['booking_room_id']} "
                "is not on this booking."
            )
            raise write_ops.BookingWriteError(msg)

        new_check_in = (
            dt.date.fromisoformat(request["new_check_in"])
            if "new_check_in" in request
            else None
        )
        new_check_out = (
            dt.date.fromisoformat(request["new_check_out"])
            if "new_check_out" in request
            else None
        )
        instruction = write_ops.CancelRoomInstruction(
            booking_room_id=row.booking_room_id,
            new_check_in=new_check_in,
            new_check_out=new_check_out,
        )
        refund = await _refund_preview(pool, row, instruction)
        instructions.append(instruction)
        total += refund
        summary_rooms.append(
            {
                "room_number": row.room_number,
                "amount": write_ops.fmt_money(refund),
            },
        )

    return instructions, summary_rooms, total


async def _refund_preview(
    pool: AsyncConnectionPool[Any],
    row: write_ops.RoomStaySummary,
    instruction: write_ops.CancelRoomInstruction,
) -> Decimal:
    """Preview the refund a cancel instruction would produce, without locking.

    Mirrors `write_ops._apply_cancel_instruction`'s edge-validation and refund
    math, but reads without `FOR UPDATE` since a proposal never writes. The
    commit re-validates under lock; this is display-only.

    Args:
        pool: Read-write booking database pool (`bh_agent_rw`).
        row: The existing room-stay being cancelled or trimmed.
        instruction: The requested cancellation or trim.

    Returns:
        Decimal: The amount this instruction would refund.

    Raises:
        write_ops.BookingWriteError: If the instruction would leave a
            mid-stay hole.

    """
    instruction = write_ops.normalize_cancel_instruction(
        instruction, check_in=row.check_in, check_out=row.check_out,
    )

    if instruction.new_check_in is None and instruction.new_check_out is None:
        return row.total_amount

    if instruction.new_check_in is not None and instruction.new_check_out is not None:
        msg = "Cancelling a range from both ends at once is not supported."
        raise write_ops.BookingWriteError(msg)

    if instruction.new_check_in is not None:
        new_in = instruction.new_check_in
        if not (row.check_in < new_in < row.check_out):
            msg = "Trim start must fall strictly within the existing stay."
            raise write_ops.BookingWriteError(msg)
        released_start, released_end = row.check_in, new_in
    else:
        new_out = instruction.new_check_out
        if new_out is None or not (row.check_in < new_out < row.check_out):
            msg = "Trim end must fall strictly within the existing stay."
            raise write_ops.BookingWriteError(msg)
        released_start, released_end = new_out, row.check_out

    async with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(
            """
            SELECT COALESCE(SUM(price), 0) AS refund
            FROM room_availability
            WHERE room_id = %s AND date >= %s AND date < %s
            """,
            (row.room_id, released_start, released_end),
        )
        result = await cur.fetchone()
    return result["refund"]


def _as_priced(room: write_ops.RoomStaySummary) -> write_ops.PricedRoomStay:
    """Adapt a `RoomStaySummary` for reuse with `write_ops.serialize_priced_stay`.

    Args:
        room: Existing room-stay summary.

    Returns:
        write_ops.PricedRoomStay: Equivalent priced stay.

    """
    return write_ops.PricedRoomStay(
        booking_room_id=room.booking_room_id,
        room_id=room.room_id,
        room_number=room.room_number,
        check_in=room.check_in,
        check_out=room.check_out,
        total_amount=room.total_amount,
    )


async def _price_modification(
    pool: AsyncConnectionPool[Any],
    *,
    booking: write_ops.BookingSummary,
    changes: list[ModifyRoomRequest],
) -> tuple[list[write_ops.ModifyRoomInstruction], list[dict[str, Any]], Decimal]:
    """Price a modification without locking or writing, for the confirmation dialog.

    Args:
        pool: Read-write booking database pool (`bh_agent_rw`).
        booking: The guest's existing booking.
        changes: Requested replacements.

    Returns:
        tuple: `(instructions, summary_changes, new_total)`, where
        `instructions` is the payload `write_ops.modify_booking` expects.

    Raises:
        write_ops.BookingWriteError: If a referenced `booking_room_id` is not
            on this booking, or a replacement room/range is unavailable.

    """
    existing = {room.booking_room_id: room for room in booking.rooms}
    changed_ids = {c["booking_room_id"] for c in changes}
    unchanged_total = sum(
        (
            room.total_amount
            for room in booking.rooms
            if room.booking_room_id not in changed_ids
        ),
        Decimal("0.00"),
    )

    instructions: list[write_ops.ModifyRoomInstruction] = []
    summary_changes: list[dict[str, Any]] = []
    changed_total = Decimal("0.00")
    for change in changes:
        before = existing.get(change["booking_room_id"])
        if before is None:
            msg = f"booking_room_id {change['booking_room_id']} is not on this booking."
            raise write_ops.BookingWriteError(msg)

        new_room_id = await _room_id_for_number(pool, change["new_room_number"])
        request = write_ops.RoomRequest(
            room_id=new_room_id,
            room_number=change["new_room_number"],
            check_in=dt.date.fromisoformat(change["new_check_in"]),
            check_out=dt.date.fromisoformat(change["new_check_out"]),
        )
        priced = await write_ops.price_modification_room(
            pool,
            existing_room_id=before.room_id,
            existing_check_in=before.check_in,
            existing_check_out=before.check_out,
            request=request,
        )

        instructions.append(
            write_ops.ModifyRoomInstruction(
                booking_room_id=before.booking_room_id,
                new_room_id=new_room_id,
                new_room_number=change["new_room_number"],
                new_check_in=priced.check_in,
                new_check_out=priced.check_out,
            ),
        )
        changed_total += priced.total_amount
        summary_changes.append(
            {
                "before": write_ops.serialize_priced_stay(_as_priced(before)),
                "after": write_ops.serialize_priced_stay(priced),
            },
        )

    return instructions, summary_changes, unchanged_total + changed_total
