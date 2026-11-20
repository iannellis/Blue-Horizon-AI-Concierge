"""Database utilities for the booking agent.

Provides metadata fetching, the startup credential and read-only checks,
transient-error detection, and the `search_rooms` tool-error text handed to
the model. None
of these reach a guest directly; see `_tool_error_message_for_model`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

import psycopg
from psycopg import sql
from psycopg_pool import PoolTimeout

from blue_horizon.agents.booking.search import RoomsMetadata
from blue_horizon.agents.exceptions import ConfigurationError, OperationalError

if TYPE_CHECKING:
    from psycopg_pool import AsyncConnectionPool

# Postgres enum types used to populate the system prompt template.
ENUM_TYPES: Final[tuple[str, ...]] = (
    "availability_status_type",
    "room_bed_type",
    "room_status_type",
    "room_type",
)

# libpq reports a rejected password with no SQLSTATE at connection time, so
# this message fragment is the only signal available.
_AUTH_REJECTED_FRAGMENT: Final = "password authentication failed"

# A no-op UPDATE that matches no row: used only to prove the read-only pool's
# role is actually refused write privileges at startup. Never mutates data
# even if the assertion this guards against has already failed.
_READ_ONLY_PROBE_SQL: Final = (
    "UPDATE room_availability SET status = status WHERE id = -1"
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


async def check_credentials(url: str, label: str, *, timeout_s: float) -> None:
    """Raise `ConfigurationError` if the database rejects a URL's password.

    Opens one direct connection instead of going through a pool, because
    `psycopg_pool` retries a rejected password internally and a checkout only
    ever reports `PoolTimeout`, which is indistinguishable from an outage.
    Any other failure propagates unchanged, for the caller to treat as
    transient.

    Args:
        url: The database URL to authenticate with.
        label: Human-readable name for the URL in the error message, such as
            the role it should authenticate as. Deliberately not an
            environment variable name: the same caller may be handed URLs
            from different variables (the application's, or an eval
            override), so only the caller's caller knows the source.
        timeout_s: Connection timeout in seconds, rounded down to whole
            seconds with a floor of 1, as libpq requires.

    Raises:
        ConfigurationError: If the server rejects the password.
        psycopg.OperationalError: If the connection fails for any other
            reason, such as the database being unreachable.

    """
    try:
        conn = await psycopg.AsyncConnection.connect(
            url, connect_timeout=max(1, int(timeout_s)),
        )
    except psycopg.OperationalError as exc:
        if _AUTH_REJECTED_FRAGMENT in str(exc):
            msg = (
                f"The {label} database URL was rejected: password "
                "authentication failed."
            )
            raise ConfigurationError(msg) from exc
        raise
    await conn.close()


async def assert_pool_is_read_only(pool: AsyncConnectionPool[Any]) -> None:
    """Prove that a read-only pool's role cannot write.

    Guards against the read-only URL being misconfigured to point at the
    same (writable) role as the read-write URL -- everything would
    otherwise work, and the guarantee this whole design depends on would
    be gone with nothing to notice.

    Args:
        pool: An open pool that should authenticate as `bh_agent_ro`.

    Raises:
        ConfigurationError: If the probe write is not refused. This is the
            read-only URL pointed at a writable role, which no retry fixes;
            only a corrected environment variable and a restart do.
        psycopg.Error: If the probe cannot be run for another reason, such
            as the database being unreachable.
        psycopg_pool.PoolTimeout: If no connection can be checked out.

    """
    try:
        async with pool.connection() as conn, conn.transaction():
            await conn.execute(_READ_ONLY_PROBE_SQL)
    except (
        psycopg.errors.ReadOnlySqlTransaction,
        psycopg.errors.InsufficientPrivilege,
    ):
        return
    msg = (
        "The read-only database URL permitted a write. It must "
        "authenticate as a role with no write privileges (bh_agent_ro) "
        "-- refusing to start with the read-only guarantee unverified."
    )
    raise ConfigurationError(msg)


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
