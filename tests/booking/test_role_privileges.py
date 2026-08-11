"""Tests that the two booking database roles enforce their intended contract.

These tests connect directly to a real Postgres as `bh_agent_ro` (via
`PGSQL_RO_DB_URL`) and `bh_agent_rw` (via `PGSQL_RW_DB_URL`) and assert what
Postgres itself will and will not allow -- with the AST guardrail
(`blue_horizon.agents.booking.guardrails`) and the application code entirely
out of the picture. That is the point: the guardrail is a redundant,
code-level restatement of the same rule, not the thing actually keeping the
model from writing (see `blue_horizon/load_data/regrant_booking_agent_role.sql`
and `blue_horizon/agents/booking/resources.py`).

Marked `db_integration` and excluded from the default `pytest` run -- see
`.github/workflows/ci.yml`'s `db-integration-tests` job, which resets the
Development branch before running these.
"""
# ruff: noqa: S101

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import psycopg
import pytest
from psycopg.rows import dict_row

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.db_integration

# A no-op UPDATE that matches no row -- the same probe
# `blue_horizon.agents.booking.resources._assert_read_pool_is_read_only` uses
# at startup to prove the read-only pool's role really cannot write.
_READ_ONLY_PROBE_SQL = "UPDATE room_availability SET status = status WHERE room_id = -1"

_PRIVILEGE_PROBE_SQL = """
SELECT
    has_table_privilege('public.rooms', 'SELECT') AS rooms_select,
    has_table_privilege('public.rooms', 'UPDATE') AS rooms_update,
    has_table_privilege('public.room_availability', 'SELECT') AS availability_select,
    has_table_privilege('public.room_availability', 'UPDATE') AS availability_update,
    has_table_privilege('public.customers', 'SELECT') AS customers_select,
    has_table_privilege('public.customers', 'INSERT') AS customers_insert,
    has_table_privilege('public.bookings', 'SELECT') AS bookings_select,
    has_table_privilege('public.bookings', 'INSERT') AS bookings_insert,
    has_table_privilege('public.bookings', 'UPDATE') AS bookings_update,
    has_table_privilege('public.bookings', 'DELETE') AS bookings_delete,
    has_table_privilege('public.booking_rooms', 'SELECT') AS booking_rooms_select,
    has_table_privilege('public.booking_rooms', 'INSERT') AS booking_rooms_insert,
    has_table_privilege('public.booking_rooms', 'UPDATE') AS booking_rooms_update,
    has_table_privilege('public.booking_rooms', 'DELETE') AS booking_rooms_delete
"""


def _connect(url_env_var: str) -> psycopg.Connection[Any]:
    """Open a sync connection using a role-scoped URL from the environment.

    Args:
        url_env_var: Name of the environment variable holding the DSN.

    Returns:
        An open, autocommit `psycopg.Connection`.

    """
    url = os.environ.get(url_env_var)
    if not url:
        pytest.skip(f"{url_env_var} not set; skipping db_integration test.")
    conn = psycopg.connect(url, row_factory=dict_row)
    conn.autocommit = True
    return conn


@pytest.fixture
def ro_conn() -> Iterator[psycopg.Connection[Any]]:
    """Yield a connection authenticated as the `bh_agent_ro` role.

    Yields:
        An open connection using `PGSQL_RO_DB_URL`.

    """
    conn = _connect("PGSQL_RO_DB_URL")
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture
def rw_conn() -> Iterator[psycopg.Connection[Any]]:
    """Yield a connection authenticated as the `bh_agent_rw` role.

    Yields:
        An open connection using `PGSQL_RW_DB_URL`.

    """
    conn = _connect("PGSQL_RW_DB_URL")
    try:
        yield conn
    finally:
        conn.close()


def _privileges(conn: psycopg.Connection[Any]) -> dict[str, bool]:
    """Read the connected role's effective privileges on every booking table.

    Uses Postgres's own `has_table_privilege()`, which resolves role
    membership/inheritance the same way an actual statement would -- so this
    reflects the role's real, effective privileges regardless of whether the
    grants in `regrant_booking_agent_role.sql` are direct or via a NOLOGIN
    grant role.

    Args:
        conn: Open connection to probe.

    Returns:
        Mapping of privilege-check name to whether it is granted.

    """
    with conn.cursor() as cur:
        cur.execute(_PRIVILEGE_PROBE_SQL)
        return cur.fetchone()  # type: ignore[return-value]


class TestReadOnlyRolePrivileges:
    """`bh_agent_ro` -- the only role the model's `run_sql` tool ever uses."""

    def test_can_select_hotel_tables(
        self, ro_conn: psycopg.Connection[Any],
    ) -> None:
        """`rooms` and `room_availability` are readable, exactly the allowlist."""
        privileges = _privileges(ro_conn)
        assert privileges["rooms_select"] is True
        assert privileges["availability_select"] is True

    def test_cannot_write_hotel_tables(
        self, ro_conn: psycopg.Connection[Any],
    ) -> None:
        """Neither allowlisted table is writable."""
        privileges = _privileges(ro_conn)
        assert privileges["rooms_update"] is False
        assert privileges["availability_update"] is False

    def test_cannot_read_customer_or_booking_tables(
        self, ro_conn: psycopg.Connection[Any],
    ) -> None:
        """`customers`, `bookings`, and `booking_rooms` are unreachable.

        This is what stops a guest's identity and reservation history from
        ever reaching a third-party inference provider through the model's
        free-text `run_sql` tool, even in principle.
        """
        privileges = _privileges(ro_conn)
        assert privileges["customers_select"] is False
        assert privileges["bookings_select"] is False
        assert privileges["booking_rooms_select"] is False

    def test_update_is_refused_by_postgres(
        self, ro_conn: psycopg.Connection[Any],
    ) -> None:
        """A live write attempt is refused at the database, guardrail bypassed."""
        with (
            pytest.raises(psycopg.errors.InsufficientPrivilege),
            ro_conn.transaction(),
            ro_conn.cursor() as cur,
        ):
            cur.execute(_READ_ONLY_PROBE_SQL)

    def test_cross_table_read_is_refused_by_postgres(
        self, ro_conn: psycopg.Connection[Any],
    ) -> None:
        """A live read of `customers` is refused at the database, guardrail bypassed."""
        with (
            pytest.raises(psycopg.errors.InsufficientPrivilege),
            ro_conn.cursor() as cur,
        ):
            cur.execute("SELECT * FROM customers LIMIT 1")


class TestReadWriteRolePrivileges:
    """`bh_agent_rw` -- used only by server-side `write_ops`, never the model."""

    def test_can_read_rooms_and_customers(
        self, rw_conn: psycopg.Connection[Any],
    ) -> None:
        """`rooms` and `customers` are readable (for pricing and identity)."""
        privileges = _privileges(rw_conn)
        assert privileges["rooms_select"] is True
        assert privileges["customers_select"] is True

    def test_cannot_write_rooms_or_customers(
        self, rw_conn: psycopg.Connection[Any],
    ) -> None:
        """`rooms` and `customers` are read-only even for the write role."""
        privileges = _privileges(rw_conn)
        assert privileges["rooms_update"] is False
        assert privileges["customers_insert"] is False

    def test_can_read_and_update_room_availability(
        self, rw_conn: psycopg.Connection[Any],
    ) -> None:
        """`room_availability` is writable, needed to flip `status`."""
        privileges = _privileges(rw_conn)
        assert privileges["availability_select"] is True
        assert privileges["availability_update"] is True

    def test_can_select_insert_update_bookings(
        self, rw_conn: psycopg.Connection[Any],
    ) -> None:
        """`bookings` supports the writes `commit_booking`/`cancel_booking` need."""
        privileges = _privileges(rw_conn)
        assert privileges["bookings_select"] is True
        assert privileges["bookings_insert"] is True
        assert privileges["bookings_update"] is True

    def test_cannot_delete_bookings(self, rw_conn: psycopg.Connection[Any]) -> None:
        """A `bookings` row is never deleted, only status-flipped to cancelled."""
        privileges = _privileges(rw_conn)
        assert privileges["bookings_delete"] is False

    def test_can_select_insert_update_delete_booking_rooms(
        self, rw_conn: psycopg.Connection[Any],
    ) -> None:
        """`booking_rooms` grants include `DELETE`.

        Regression test for a real gap found during manual verification:
        `write_ops.cancel_booking`'s full-cancel path deletes the
        `booking_rooms` row rather than soft-flagging it, which raised
        `psycopg.errors.InsufficientPrivilege` in production until `DELETE`
        was added to this role's grant in
        `regrant_booking_agent_role.sql`.
        """
        privileges = _privileges(rw_conn)
        assert privileges["booking_rooms_select"] is True
        assert privileges["booking_rooms_insert"] is True
        assert privileges["booking_rooms_update"] is True
        assert privileges["booking_rooms_delete"] is True
