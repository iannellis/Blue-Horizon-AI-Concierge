"""Database utilities for the booking agent.

Provides metadata fetching, transient-error detection, and the `search_rooms`
tool-error text handed to the model. None of these reach a guest directly;
see `_tool_error_message_for_model`.
"""

from __future__ import annotations

from typing import Any, Final

import psycopg
from psycopg import sql
from psycopg_pool import PoolTimeout

from blue_horizon.agents.booking.search import RoomsMetadata
from blue_horizon.agents.exceptions import OperationalError

# Postgres enum types used to populate the system prompt template.
ENUM_TYPES: Final[tuple[str, ...]] = (
    "availability_status_type",
    "room_bed_type",
    "room_status_type",
    "room_type",
)


async def fetch_rooms_metadata(db_url: str) -> RoomsMetadata:
    """Fetch the search vocabulary and bounds from the database.

    The result fills both the system prompt and the `search_rooms` argument
    model, so the values the model may search for are exactly the values the
    database holds.

    Args:
        db_url: Database URL. The read-only role is sufficient.

    Returns:
        Enum values, amenity and view type vocabularies (NULLs dropped), the
        floor range, and the first and last nights with availability data.

    Raises:
        OperationalError: If metadata queries fail or a table is empty.

    """
    enum_values: dict[str, list[str]] = {}

    try:
        async with (
            await psycopg.AsyncConnection.connect(db_url) as conn,
            conn.cursor() as cur,
        ):
            await cur.execute("SET search_path TO public;")
            for enum_type in ENUM_TYPES:
                query = sql.SQL("SELECT unnest(enum_range(NULL::{}));").format(
                    sql.Identifier(enum_type),
                )
                await cur.execute(query)
                enum_values[enum_type] = [
                    str(row[0])
                    for row in await cur.fetchall()
                    if row and row[0] is not None
                ]

            await cur.execute(
                "SELECT DISTINCT unnest(basic_amenities || additional_amenities) "
                "FROM rooms;",
            )
            amenities = await _distinct_strings(cur)

            await cur.execute("SELECT DISTINCT unnest(view_type) FROM rooms;")
            view_types = await _distinct_strings(cur)

            await cur.execute("SELECT MIN(floor), MAX(floor) FROM rooms;")
            floor_row = await cur.fetchone()

            await cur.execute("SELECT MIN(date), MAX(date) FROM room_availability;")
            date_row = await cur.fetchone()

        if floor_row is None or date_row is None or None in (*floor_row, *date_row):
            msg = "rooms or room_availability is empty"
            raise ValueError(msg)  # noqa: TRY301

    except Exception as exc:
        msg = "Failed to fetch rooms metadata from the database"
        raise OperationalError(msg) from exc

    else:
        return RoomsMetadata(
            enum_values=enum_values,
            amenities=amenities,
            view_types=view_types,
            min_floor=floor_row[0],
            max_floor=floor_row[1],
            first_night=date_row[0],
            last_night=date_row[1],
        )


async def _distinct_strings(cur: psycopg.AsyncCursor[Any]) -> tuple[str, ...]:
    """Collect a single-column result as sorted strings, dropping NULLs.

    Args:
        cur: Cursor that has just executed a one-column query.

    Returns:
        The non-NULL values, sorted.

    """
    rows = await cur.fetchall()
    return tuple(sorted(str(row[0]) for row in rows if row[0] is not None))


def is_transient_conn_error(exc: BaseException) -> bool:
    """Determine whether an exception looks like a transient failure.

    Covers both Neon cold-start cases. A *suspended* compute dies mid
    connection, which surfaces as one of the message patterns below. A
    connection that instead fails to *establish* at all (compute still
    waking up) does not raise those -- but `psycopg_pool.AsyncConnectionPool`
    never lets that raw connect exception reach a checkout call either: a
    failed background connect attempt is retried internally
    (`reconnect_timeout` defaults to 5 minutes, far longer than this
    project's per-checkout `pool.timeout_s`), so the checkout's own
    wait simply times out and raises `PoolTimeout` first. Confirmed
    empirically against `psycopg_pool` 3.3.1 in
    `tests/booking/test_resources.py`; unconditionally treating
    `PoolTimeout` as transient below therefore already covers the
    wake-up case, and no connect-side message pattern needs adding here.

    Args:
        exc: Exception raised by psycopg/psycopg_pool.

    Returns:
        True if the exception looks retryable (e.g., SSL close, pool timeout).

    """
    if isinstance(exc, (PoolTimeout, TimeoutError)):
        return True

    msg = str(exc).lower()
    patterns = (
        "ssl connection has been closed unexpectedly",
        "server closed the connection unexpectedly",
        "connection is closed",
        "connection not open",
        "terminating connection",
    )
    return any(p in msg for p in patterns)


def _tool_error_message_for_model() -> str:
    """Return the `search_rooms` error text handed back to the booking model.

    This is prompt text, not UX copy: it is consumed only as a `search_rooms`
    tool result, and no guest ever sees it directly. `booking.txt` tells
    the model to relay a `propose_*` failure "in guest-facing terms" for
    that tool family, but for `search_rooms` specifically the adjacent rule
    (see the `DATABASE_UNAVAILABLE` marker below) tells the model this is
    not a search error at all, so the audience of this string's *wording*
    is the model's reasoning about what to do next, not the guest's eyes.

    Returns:
        Tool-result text carrying a stable leading `DATABASE_UNAVAILABLE`
        marker the system prompt's retry rule keys off, followed by
        readable English so a verbatim relay is not gibberish.

    """
    return (
        "DATABASE_UNAVAILABLE: the database is temporarily unreachable "
        "after retrying. This is not a problem with your search -- do not "
        "change or retry it yourself. Tell the guest the booking system is "
        "temporarily unavailable right now."
    )
