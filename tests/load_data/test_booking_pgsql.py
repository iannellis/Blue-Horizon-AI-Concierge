"""Tests for the booking PostgreSQL loader helpers."""

# ruff: noqa: S101

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, cast

import pandas as pd
import psycopg.sql

from blue_horizon.load_data.booking_pgsql import (
    CUSTOMERS_COLUMNS,
    ROOM_AVAIL_COLUMNS,
    ROOMS_COLUMNS,
    _remap_bookings_to_new_customer_ids,
    _select_non_overlapping_bookings,
    _select_richest_customer_ids,
    copy_dataframe_into_table,
    prepare_accepted_bookings,
    prepare_customers_dataframe,
    prepare_room_availability_dataframe,
    prepare_rooms_dataframe,
    regrant_booking_agent_role,
    setup_maintenance_booking_guard,
    setup_schema,
)

if TYPE_CHECKING:
    from types import TracebackType


class TestPrepareRoomsDataFrame:
    """prepare_rooms_dataframe preserves data-driven enum values."""

    def test_unseen_room_type_and_status_are_preserved(self, tmp_path: Path) -> None:
        """The loader no longer depends on hard-coded Python enum mappings."""
        rooms_path = tmp_path / "rooms.pkl"
        pd.DataFrame(
            {
                "room_id": ["RM101", "RM202"],
                "room_number": [101, 202],
                "floor": [1, 2],
                "type": ["Sky Villa", "Standard"],
                "square_feet": [500, 350],
                "basic_amenities": [["WiFi"], ["WiFi"]],
                "additional_amenities": [["Sauna"], ["Desk"]],
                "max_occupancy": [4, 2],
                "bed_type": ["Murphy Bed", "Queen"],
                "view_type": [["Ocean"], ["Garden"]],
                "accessibility": [True, False],
                "status": ["Out of Service", "Available"],
                "last_renovation": ["2024-01-01", "2023-01-01"],
                "base_rate": [350.0, 199.0],
                "max_rate": [650.0, 299.0],
            },
        ).to_pickle(rooms_path)

        df, room_types, bed_types, room_statuses = prepare_rooms_dataframe(tmp_path)

        assert df["room_id"].tolist() == [101, 202]
        assert df["type"].tolist() == ["Sky Villa", "Standard"]
        assert room_types == ["Sky Villa", "Standard"]
        assert bed_types == ["Murphy Bed", "Queen"]
        assert room_statuses == ["Out of Service", "Available"]
        assert list(df.columns) == list(ROOMS_COLUMNS)


class TestPrepareRoomAvailabilityDataFrame:
    """prepare_room_availability_dataframe preserves data-driven statuses."""

    def test_unseen_status_is_preserved(self, tmp_path: Path) -> None:
        """Availability statuses come from the source data instead of a code map."""
        availability_path = tmp_path / "room_availability.pkl"
        pd.DataFrame(
            {
                "room_id": ["RM101", "RM101"],
                "room_number": [101, 101],
                "date": ["2026-06-01", "2026-06-02"],
                "status": ["Held", "Available"],
                "price": [325.0, 350.0],
                "max_occupancy": [4, 4],
            },
        ).to_pickle(availability_path)

        df, statuses = prepare_room_availability_dataframe(tmp_path)

        assert df["room_id"].tolist() == [101, 101]
        assert df["status"].tolist() == ["Held", "Available"]
        assert statuses == ["Held", "Available"]
        assert list(df.columns) == list(ROOM_AVAIL_COLUMNS)


class TestSelectNonOverlappingBookings:
    """_select_non_overlapping_bookings keeps the maximum non-overlapping subset."""

    def test_overlap_loses_to_earlier_check_out_kept_first(self) -> None:
        """Of two overlapping rows on one room, the earliest-ending one is kept."""
        df = pd.DataFrame(
            {
                "room_id": [1, 1],
                "check_in": pd.to_datetime(["2025-01-02", "2025-01-04"]),
                "check_out": pd.to_datetime(["2025-01-05", "2025-01-06"]),
            },
        )

        kept = _select_non_overlapping_bookings(df)

        assert kept.tolist() == [True, False]

    def test_adjacent_rows_are_both_kept(self) -> None:
        """Back-to-back rows (one's check_out equals the other's check_in) both fit."""
        df = pd.DataFrame(
            {
                "room_id": [1, 1],
                "check_in": pd.to_datetime(["2025-01-02", "2025-01-05"]),
                "check_out": pd.to_datetime(["2025-01-05", "2025-01-07"]),
            },
        )

        kept = _select_non_overlapping_bookings(df)

        assert kept.tolist() == [True, True]

    def test_different_rooms_do_not_affect_each_other(self) -> None:
        """Overlapping date ranges on different rooms are independent."""
        df = pd.DataFrame(
            {
                "room_id": [1, 2],
                "check_in": pd.to_datetime(["2025-01-02", "2025-01-02"]),
                "check_out": pd.to_datetime(["2025-01-05", "2025-01-05"]),
            },
        )

        kept = _select_non_overlapping_bookings(df)

        assert kept.tolist() == [True, True]


class TestPrepareAcceptedBookings:
    """prepare_accepted_bookings maps, clamps, and filters pre-existing bookings."""

    def test_maps_clamps_and_filters(self, tmp_path: Path) -> None:
        """Room numbers are mapped, dates clamped/dropped, overlaps resolved."""
        bookings_path = tmp_path / "room_bookings.pkl"
        pd.DataFrame(
            {
                "customer_id": [1, 2, 3, 1, 2],
                "room_number": [101, 101, 101, 102, 102],
                "check_in": pd.to_datetime(
                    [
                        "2025-01-02", "2025-01-04", "2025-01-05",
                        "2024-12-30", "2025-01-20",
                    ],
                ),
                "check_out": pd.to_datetime(
                    [
                        "2025-01-05", "2025-01-06", "2025-01-07",
                        "2025-01-02", "2025-01-25",
                    ],
                ),
                "total_amount": [100.0, 100.0, 100.0, 100.0, 100.0],
            },
            index=pd.Index(
                ["BK1", "BK2", "BK3", "BK4", "BK5"], name="booking_id",
            ),
        ).to_pickle(bookings_path)

        df_rooms = pd.DataFrame({"room_number": [101, 102], "room_id": [1, 2]})
        df_availability = pd.DataFrame(
            {"date": pd.to_datetime(["2025-01-01", "2025-01-10"])},
        )

        accepted = prepare_accepted_bookings(tmp_path, df_rooms, df_availability)

        # BK2 loses to BK1's earlier check_out (both room 101); BK5 is
        # entirely outside room_availability's range and is dropped.
        assert set(accepted["source_booking_id"]) == {"BK1", "BK3", "BK4"}
        bk4 = accepted.set_index("source_booking_id").loc["BK4"]
        assert bk4["room_id"] == 2  # noqa: PLR2004
        # BK4's check_in (2024-12-30) was clamped up to the covered range's start.
        assert bk4["check_in"] == pd.Timestamp("2025-01-01")


class TestSelectRichestCustomerIds:
    """_select_richest_customer_ids ranks customers by accepted-booking count."""

    def test_picks_top_count_breaking_ties_by_ascending_id(self) -> None:
        """More bookings outranks fewer; ties among equal counts go to the lower id."""
        df_accepted_bookings = pd.DataFrame(
            {"customer_id": [5, 5, 5, 2, 2, 2, 9, 1]},
        )

        selected = _select_richest_customer_ids(df_accepted_bookings, count=2)

        # Customers 5 and 2 both have 3 bookings (a tie); 9 and 1 have one
        # each. The tie is broken by ascending customer_id, and the final
        # result is sorted ascending regardless of booking count.
        assert selected == [2, 5]


class TestPrepareCustomersDataframe:
    """prepare_customers_dataframe loads every customer, richest history first."""

    def test_remaps_richest_first_then_remaining_in_ascending_order(
        self, tmp_path: Path,
    ) -> None:
        """Customers with accepted bookings get ids 1..N; the rest follow, ascending."""
        customers_path = tmp_path / "customers.pkl"
        pd.DataFrame(
            {
                "first_name": ["Ana", "Bo", "Cy", "Dee", "Eli"],
                "last_name": ["Adams", "Blake", "Cruz", "Diaz", "Evans"],
            },
            index=pd.Index([10, 20, 30, 40, 50], name="customer_id"),
        ).to_pickle(customers_path)
        # Only customers 10/20/30 have accepted bookings; 40/50 have none.
        df_accepted_bookings = pd.DataFrame(
            {"customer_id": [30, 30, 10, 20]},
        )

        df_customers, new_id_by_source_id = prepare_customers_dataframe(
            tmp_path, df_accepted_bookings, seeded_customer_count=25,
        )

        assert new_id_by_source_id == {10: 1, 20: 2, 30: 3, 40: 4, 50: 5}
        assert list(df_customers.columns) == list(CUSTOMERS_COLUMNS)
        assert df_customers["customer_id"].tolist() == [1, 2, 3, 4, 5]
        assert df_customers.set_index("customer_id").loc[3, "first_name"] == "Cy"
        assert df_customers.set_index("customer_id").loc[5, "first_name"] == "Eli"


class TestRemapBookingsToNewCustomerIds:
    """_remap_bookings_to_new_customer_ids applies the loaded customer id remap."""

    def test_remaps_mapped_rows_and_drops_unmapped_ones(self) -> None:
        """Every mapped row gets its new id; a row with no entry is dropped."""
        df_accepted_bookings = pd.DataFrame({"customer_id": [1, 2, 3, 4]})
        new_customer_id_by_source_id = {1: 10, 2: 11, 3: 12}

        result = _remap_bookings_to_new_customer_ids(
            df_accepted_bookings, new_customer_id_by_source_id,
        )

        assert result["customer_id"].tolist() == [10, 11, 12]


class _FakeCopy:
    """Simple COPY context manager capturing rows written by the loader."""

    def __init__(self) -> None:
        """Initialize an empty row capture buffer."""
        self.rows: list[tuple[object, ...]] = []

    def __enter__(self) -> Self:
        """Return the active fake COPY object."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        """Do not suppress exceptions raised inside the context."""
        return False

    def write_row(self, row: tuple[object, ...]) -> None:
        """Record one row sent through COPY."""
        self.rows.append(tuple(row))


class _FakeCursor:
    """Simple cursor context manager exposing a fake ``copy()`` method."""

    def __init__(self) -> None:
        """Initialize the cursor with a fake COPY target."""
        self.copy_stmt: object | None = None
        self.copy_context = _FakeCopy()

    def __enter__(self) -> Self:
        """Return the active fake cursor."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        """Do not suppress exceptions raised inside the context."""
        return False

    def copy(self, statement: object) -> _FakeCopy:
        """Capture the COPY statement and return the fake COPY writer."""
        self.copy_stmt = statement
        return self.copy_context


class _FakeConnection:
    """Connection wrapper returning a single fake cursor."""

    def __init__(self) -> None:
        """Initialize the connection with one reusable fake cursor."""
        self.cursor_obj = _FakeCursor()

    def cursor(self) -> _FakeCursor:
        """Return the fake cursor instance."""
        return self.cursor_obj


class TestCopyDataFrameIntoTable:
    """copy_dataframe_into_table uses COPY row loading instead of executemany."""

    def test_rows_are_written_via_copy(self) -> None:
        """COPY receives rows in column order for the requested table."""
        conn = _FakeConnection()
        df = pd.DataFrame(
            {
                "room_id": [101],
                "room_number": [101],
                "date": ["2026-06-01"],
                "status": ["Available"],
                "price": [325.0],
                "max_occupancy": [4],
            },
        )

        # _FakeConnection implements only the .cursor()/.copy() surface
        # copy_dataframe_into_table actually exercises, not the full
        # psycopg.Connection interface.
        copy_dataframe_into_table(
            cast("psycopg.Connection[Any]", conn),
            table_name="room_availability",
            columns=ROOM_AVAIL_COLUMNS,
            df=df,
        )

        assert conn.cursor_obj.copy_stmt is not None
        assert conn.cursor_obj.copy_context.rows == [
            (101, 101, "2026-06-01", "Available", 325.0, 4),
        ]


class _FakeExecuteConnection:
    """Connection wrapper recording the SQL text passed to ``execute``."""

    def __init__(self) -> None:
        """Initialize the connection with no recorded SQL yet."""
        self.executed_sql: str | None = None

    def execute(self, sql: str | psycopg.sql.Composable) -> None:
        """Record the SQL text this connection was asked to execute.

        Args:
            sql: The SQL text, or a `psycopg.sql.Composable` wrapping it,
                passed to ``execute``. Normalized to plain text the same way
                psycopg itself resolves either form before sending it to the
                database.

        """
        self.executed_sql = psycopg.sql.as_string(sql)


class TestRegrantBookingAgentRole:
    """regrant_booking_agent_role executes the checked-in grants SQL file."""

    def test_executes_the_sql_file_contents(self) -> None:
        """The connection receives the full text of the .sql file, verbatim."""
        conn = _FakeExecuteConnection()

        # _FakeExecuteConnection implements only the .execute() surface
        # regrant_booking_agent_role actually exercises, not the full
        # psycopg.Connection interface.
        regrant_booking_agent_role(cast("psycopg.Connection[Any]", conn))

        sql_path = (
            Path(__file__).parents[2]
            / "blue_horizon"
            / "load_data"
            / "regrant_booking_agent_role.sql"
        )
        assert conn.executed_sql == sql_path.read_text(encoding="utf-8")
        assert conn.executed_sql is not None
        assert "bh_agent_ro" in conn.executed_sql
        assert "bh_agent_rw" in conn.executed_sql


class TestSetupSchema:
    """setup_schema executes schema.sql with its enum placeholders filled in."""

    def test_fills_in_enum_placeholders(self) -> None:
        """Each of the four enum-label lists appears, quoted, in the executed SQL."""
        conn = _FakeExecuteConnection()

        # _FakeExecuteConnection implements only the .execute() surface
        # setup_schema actually exercises, not the full psycopg.Connection
        # interface.
        setup_schema(
            cast("psycopg.Connection[Any]", conn),
            room_type_values=["Sky Villa"],
            bed_type_values=["Murphy Bed"],
            room_status_values=["Available"],
            availability_status_values=["Booked"],
        )

        assert conn.executed_sql is not None
        assert "'Sky Villa'" in conn.executed_sql
        assert "'Murphy Bed'" in conn.executed_sql
        assert "'Available'" in conn.executed_sql
        assert "'Booked'" in conn.executed_sql
        assert "CREATE TABLE rooms" in conn.executed_sql
        assert "CREATE TABLE booking_rooms" in conn.executed_sql
        # The maintenance-guard trigger is a separate, later step (see
        # setup_maintenance_booking_guard) so it must not be in this file.
        assert "booking_rooms_no_maintenance" not in conn.executed_sql

    def test_placeholders_are_not_left_literally_in_the_sql(self) -> None:
        """Every {..._values} placeholder is substituted, none survive verbatim."""
        conn = _FakeExecuteConnection()

        setup_schema(
            cast("psycopg.Connection[Any]", conn),
            room_type_values=["Sky Villa"],
            bed_type_values=["Murphy Bed"],
            room_status_values=["Available"],
            availability_status_values=["Booked"],
        )

        assert conn.executed_sql is not None
        assert "{room_type_values}" not in conn.executed_sql
        assert "{bed_type_values}" not in conn.executed_sql
        assert "{room_status_values}" not in conn.executed_sql
        assert "{availability_status_values}" not in conn.executed_sql


class TestSetupMaintenanceBookingGuard:
    """setup_maintenance_booking_guard executes the checked-in trigger SQL file."""

    def test_executes_the_sql_file_contents(self) -> None:
        """The connection receives the full text of the .sql file, verbatim."""
        conn = _FakeExecuteConnection()

        # _FakeExecuteConnection implements only the .execute() surface
        # setup_maintenance_booking_guard actually exercises, not the full
        # psycopg.Connection interface.
        setup_maintenance_booking_guard(cast("psycopg.Connection[Any]", conn))

        sql_path = (
            Path(__file__).parents[2]
            / "blue_horizon"
            / "load_data"
            / "maintenance_booking_guard.sql"
        )
        assert conn.executed_sql == sql_path.read_text(encoding="utf-8")
        assert conn.executed_sql is not None
        assert "booking_rooms_no_maintenance" in conn.executed_sql
        assert "prevent_maintenance_booking" in conn.executed_sql
