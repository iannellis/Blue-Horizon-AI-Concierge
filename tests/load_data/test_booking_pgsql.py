"""Tests for the booking PostgreSQL loader helpers."""

# ruff: noqa: S101

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Self

import pandas as pd

from blue_horizon.load_data.booking_pgsql import (
    ROOM_AVAIL_COLUMNS,
    ROOMS_COLUMNS,
    copy_dataframe_into_table,
    prepare_room_availability_dataframe,
    prepare_rooms_dataframe,
    regrant_booking_agent_role,
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

        copy_dataframe_into_table(
            conn,
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

    def execute(self, sql: str) -> None:
        """Record the SQL text this connection was asked to execute.

        Args:
            sql: The SQL text passed to ``execute``.

        """
        self.executed_sql = sql


class TestRegrantBookingAgentRole:
    """regrant_booking_agent_role executes the checked-in grants SQL file."""

    def test_executes_the_sql_file_contents(self) -> None:
        """The connection receives the full text of the .sql file, verbatim."""
        conn = _FakeExecuteConnection()

        regrant_booking_agent_role(conn)

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
