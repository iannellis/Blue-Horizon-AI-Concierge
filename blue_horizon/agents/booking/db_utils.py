"""Database utilities for the booking SQL agent.

Provides metadata fetching, backoff helpers, transient-error detection, row
truncation, and the `run_sql` tool-error text handed to the model. None of
these reach a guest directly; see `_tool_error_message_for_model`.
"""

from __future__ import annotations

from typing import Any, Final

import psycopg
from psycopg import sql
from psycopg_pool import PoolTimeout

from blue_horizon.agents.exceptions import OperationalError

# Postgres enum types used to populate the system prompt template.
ENUM_TYPES: Final[tuple[str, ...]] = (
    "availability_status_type",
    "room_bed_type",
    "room_status_type",
    "room_type",
)


async def fetch_rooms_metadata(
    pgsql_rw_db_url: str,
) -> tuple[dict[str, list[str]], list[str], list[str], list[str]]:
    """Fetch database metadata used to fill the system prompt.

    The system prompt template is populated with:
      - Enum values for known enum types.
      - Distinct values from array columns in the rooms table.

    Args:
        pgsql_rw_db_url: Database URL.

    Returns:
        Tuple of (enum_values, basic_amenities, additional_amenities, view_types).

    Raises:
        OperationalError: If metadata queries fail.

    """
    enum_values: dict[str, list[str]] = {}

    try:
        async with (
            await psycopg.AsyncConnection.connect(pgsql_rw_db_url) as conn,
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

            await cur.execute("SELECT DISTINCT unnest(basic_amenities) FROM rooms;")
            basic_amenities = [r[0] for r in await cur.fetchall()]

            await cur.execute(
                "SELECT DISTINCT unnest(additional_amenities) FROM rooms;",
            )
            additional_amenities = [r[0] for r in await cur.fetchall()]

            await cur.execute("SELECT DISTINCT unnest(view_type) FROM rooms;")
            view_types = [r[0] for r in await cur.fetchall()]

    except Exception as exc:
        msg = "Failed to fetch rooms metadata from the database"
        raise OperationalError(msg) from exc

    else:
        return enum_values, basic_amenities, additional_amenities, view_types


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


def _truncate_rows(
    rows: list[dict[str, Any]],
    *,
    max_rows: int,
) -> tuple[list[dict[str, Any]], bool]:
    """Truncate DB result rows to a maximum length.

    Args:
        rows: Result rows.
        max_rows: Maximum number of rows to keep.

    Returns:
        Tuple of (rows_out, truncated).

    """
    if len(rows) <= max_rows:
        return rows, False
    return rows[:max_rows], True


def _tool_error_message_for_model() -> str:
    """Return the `run_sql` error text handed back to the booking model.

    This is prompt text, not UX copy: it is consumed only as a `run_sql`
    tool result, and no guest ever sees it directly. `booking.txt` tells
    the model to relay a `propose_*` failure "in guest-facing terms" for
    that tool family, but for `run_sql` specifically the adjacent rule
    (see the `DATABASE_UNAVAILABLE` marker below) tells the model this is
    not a query error at all, so the audience of this string's *wording*
    is the model's reasoning about what to do next, not the guest's eyes.

    Returns:
        Tool-result text carrying a stable leading `DATABASE_UNAVAILABLE`
        marker the system prompt's retry rule keys off, followed by
        readable English so a verbatim relay is not gibberish.

    """
    return (
        "DATABASE_UNAVAILABLE: the database is temporarily unreachable "
        "after retrying. This is not a query error -- do not rewrite or "
        "retry this query yourself. Tell the guest the booking system is "
        "temporarily unavailable right now."
    )
