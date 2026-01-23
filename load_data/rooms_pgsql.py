"""Utilities to rebuild the PostgreSQL rooms and availability tables from pickled data."""  # noqa: E501

import logging
import os
import sys
from collections.abc import Iterable, Sequence
from enum import Enum
from pathlib import Path
from typing import LiteralString, cast

import pandas as pd
import psycopg
from dotenv import load_dotenv
from psycopg.types.enum import EnumInfo, register_enum

DATA_PATH = Path(__file__).parents[1] / "data/pandas"

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def ensure_pickle_path(path: Path) -> None:
    """Verify that a pickled data file exists before it is read.

    Args:
        path: File path that must exist before attempting to unpickle.

    Raises:
        FileNotFoundError: If the target file is missing.

    """
    if not path.exists():
        msg = f"Missing expected pickle at {path}"
        logger.error(msg)
        raise FileNotFoundError(msg)


ROOMS_COLUMNS: Sequence[str] = [
    "room_id",
    "room_number",
    "floor",
    "type",
    "square_feet",
    "basic_amenities",
    "additional_amenities",
    "max_occupancy",
    "bed_type",
    "view_type",
    "accessibility",
    "status",
    "last_renovation",
    "base_rate",
    "max_rate",
]

ROOM_AVAIL_COLUMNS: Sequence[str] = [
    "room_id",
    "room_number",
    "date",
    "status",
    "price",
    "max_occupancy",
]


class RoomType(Enum):
    """Enum mapping for room type labels stored in Postgres."""

    Standard = "Standard"
    Deluxe = "Deluxe"
    Suite = "Suite"
    Presidential_Suite = "Presidential Suite"


class AvailabilityStatusType(Enum):
    """Enum mapping for availability status values stored in Postgres."""

    Booked = "Booked"
    Available = "Available"
    Maintenance = "Maintenance"


enum_room_type_mapping = {
    "Standard": RoomType.Standard,
    "Deluxe": RoomType.Deluxe,
    "Suite": RoomType.Suite,
    "Presidential Suite": RoomType.Presidential_Suite,
}

sql_room_type_mapping = [
    (RoomType.Standard, "Standard"),
    (RoomType.Deluxe, "Deluxe"),
    (RoomType.Suite, "Suite"),
    (RoomType.Presidential_Suite, "Presidential Suite"),
]

enum_availability_status_type_mapping = {
    "Booked": AvailabilityStatusType.Booked,
    "Available": AvailabilityStatusType.Available,
    "Maintenance": AvailabilityStatusType.Maintenance,
}


def get_pgsql_conn_string() -> str:
    """Load the PGSQL connection string from the environment.

    Returns:
        str: The PGSQL connection string stored in PGSQL_DB_URL.

    Raises:
        RuntimeError: If PGSQL_DB_URL is unset.

    """
    load_dotenv()
    conn_string = os.getenv("PGSQL_DB_URL")
    if not conn_string:
        error_msg = "PGSQL_DB_URL is not set in the environment"
        raise RuntimeError(error_msg)
    return conn_string


def build_enum_definition(values: Iterable[str]) -> str:
    """Generate an ordered SQL enum definition string.

    Args:
        values: Iterable of enum labels that must be preserved in order.

    Returns:
        Parenthesized, quoted list suitable for CREATE TYPE AS ENUM.

    """
    unique_values = list(dict.fromkeys(values))
    if not unique_values:
        error_msg = "Enum definitions cannot be empty"
        logger.error(error_msg)
        raise ValueError(error_msg)
    quoted_values = "', '".join(unique_values)
    return f"('{quoted_values}')"


def prepare_rooms_dataframe(
    data_path: Path,
) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    """Load, normalize, and prune rooms data for insertion.

    Args:
        data_path: Directory containing rooms.pkl.

    Returns:
        Tuple containing the prepared DataFrame followed by lists of unique room types,
        bed types, and statuses (in discovery order).

    """
    rooms_path = data_path / "rooms.pkl"
    ensure_pickle_path(rooms_path)
    df = pd.read_pickle(rooms_path)  # noqa: S301
    df["room_id"] = df["room_id"].apply(lambda raw: int(raw[2:]))

    room_types = df["type"].unique().tolist()
    room_bed_types = df["bed_type"].unique().tolist()
    room_statuses = df["status"].unique().tolist()

    df["type"] = df["type"].map(enum_room_type_mapping)
    df = df.loc[:, ROOMS_COLUMNS]

    return df, room_types, room_bed_types, room_statuses


def prepare_room_availability_dataframe(
    data_path: Path,
) -> tuple[pd.DataFrame, list[str]]:
    """Load and normalize room availability records for the database.

    Args:
        data_path: Directory containing room_availability.pkl.

    Returns:
        Tuple with the prepared availability DataFrame and a list of unique statuses.

    """
    availability_path = data_path / "room_availability.pkl"
    ensure_pickle_path(availability_path)
    df = pd.read_pickle(availability_path)  # noqa: S301
    df["room_id"] = df["room_id"].apply(lambda raw: int(raw[2:]))

    availability_statuses = df["status"].unique().tolist()
    df["status"] = df["status"].map(enum_availability_status_type_mapping)
    df = df.loc[:, ROOM_AVAIL_COLUMNS]

    return df, availability_statuses


def setup_rooms_schema(
    conn: psycopg.Connection,
    room_type_def: str,
    bed_type_def: str,
    status_def: str,
) -> None:
    """Reset rooms-related enums and recreate the `rooms` table.

    Args:
        conn: Active database connection.
        room_type_def: Enum definition string for room_type.
        bed_type_def: Enum definition string for room_bed_type.
        status_def: Enum definition string for room_status_type.

    """
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS rooms CASCADE;")
        cur.execute("DROP TYPE IF EXISTS room_type CASCADE;")
        cur.execute("DROP TYPE IF EXISTS room_bed_type CASCADE;")
        cur.execute("DROP TYPE IF EXISTS room_status_type CASCADE;")

        query_str = f"CREATE TYPE room_type AS ENUM {room_type_def};"
        cur.execute(cast("LiteralString", query_str))
        query_str = f"CREATE TYPE room_bed_type AS ENUM {bed_type_def};"
        cur.execute(cast("LiteralString", query_str))
        query_str = f"CREATE TYPE room_status_type AS ENUM {status_def};"
        cur.execute(cast("LiteralString", query_str))

        cur.execute(
            """
            CREATE TABLE rooms (
                room_id INT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                room_number INT NOT NULL,
                floor INT NOT NULL,
                type room_type,
                square_feet INT,
                basic_amenities TEXT[],
                additional_amenities TEXT[],
                max_occupancy INT,
                bed_type room_bed_type,
                view_type TEXT[],
                accessibility BOOLEAN,
                status room_status_type,
                last_renovation DATE,
                base_rate NUMERIC(10, 2),
                max_rate NUMERIC(10, 2)
            );
            """,
        )


def register_room_type_enum(conn: psycopg.Connection) -> None:
    """Register the Python RoomType enum with the Postgres enum metadata.

    This particular enum causes problems if not registered this way.

    Args:
        conn: Active database connection.

    """
    info = EnumInfo.fetch(conn, "room_type")
    if info is None:
        error_msg = "room_type enum metadata could not be fetched"
        raise RuntimeError(error_msg)
    register_enum(info, context=conn, enum=RoomType, mapping=sql_room_type_mapping)


def insert_rooms_data(conn: psycopg.Connection, df_rooms: pd.DataFrame) -> None:
    """Insert normalized rooms data into the database.

    Args:
        conn: Active database connection.
        df_rooms: DataFrame keyed to `ROOMS_COLUMNS`.

    """
    columns_sql = ", ".join(ROOMS_COLUMNS)
    placeholders = ", ".join(["%s"] * len(ROOMS_COLUMNS))
    insert_sql = f"INSERT INTO rooms ({columns_sql}) VALUES ({placeholders});"  # noqa: S608

    if df_rooms.empty:
        logger.warning("No rooms records to insert; skipping.")
        return

    with conn.cursor() as cur:
        cur.executemany(cast("LiteralString", insert_sql), df_rooms.to_numpy().tolist())
        logger.info("Inserted rooms data.")


def setup_room_availability_schema(conn: psycopg.Connection, status_def: str) -> None:
    """Reset availability enums and recreate the `room_availability` table.

    Args:
        conn: Active database connection.
        status_def: Enum definition string for availability_status_type.

    """
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS room_availability;")
        cur.execute("DROP TYPE IF EXISTS availability_status_type CASCADE;")
        query_str = f"CREATE TYPE availability_status_type AS ENUM {status_def};"
        cur.execute(cast("LiteralString", query_str))

        cur.execute(
            """
            CREATE TABLE room_availability (
                id INT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                room_id INT NOT NULL,
                room_number INT NOT NULL,
                date DATE NOT NULL,
                status availability_status_type,
                price NUMERIC(8,2),
                max_occupancy INT,
                FOREIGN KEY (room_id) REFERENCES rooms(room_id),
                CONSTRAINT room_availability_room_id_date_uniq UNIQUE (room_id, date)
            );
            """,
        )


def register_room_availability_enum(conn: psycopg.Connection) -> None:
    """Register the Python AvailabilityStatusType enum with Postgres metadata.

    Args:
        conn: Active database connection.

    """
    info = EnumInfo.fetch(conn, "availability_status_type")
    if info is None:
        error_msg = "availability_status_type enum metadata could not be fetched"
        raise RuntimeError(error_msg)
    register_enum(info, context=conn, enum=AvailabilityStatusType)


def insert_room_availability_data(
    conn: psycopg.Connection,
    df_availability: pd.DataFrame,
) -> None:
    """Insert normalized availability records into `room_availability`.

    Args:
        conn: Active database connection.
        df_availability: DataFrame keyed to `ROOM_AVAIL_COLUMNS`.

    """
    columns_sql = ", ".join(ROOM_AVAIL_COLUMNS)
    placeholders = ", ".join(["%s"] * len(ROOM_AVAIL_COLUMNS))
    insert_sql = (
        f"INSERT INTO room_availability ({columns_sql}) VALUES ({placeholders});"  # noqa: S608
    )

    if df_availability.empty:
        logger.warning("No room availability records to insert; skipping.")
        return

    with conn.cursor() as cur:
        cur.executemany(
            cast("LiteralString", insert_sql),
            df_availability.to_numpy().tolist(),
        )
        logger.info("Inserted room_availability data.")


def reload_sql_tables() -> None:
    """Prepare data and rebuild the room and availability tables.

    The function reads pickled source data, reconstructs the required enums,
    and inserts records into both `rooms` and `room_availability`.
    """
    try:
        conn_string = get_pgsql_conn_string()
        df_rooms, room_type_values, bed_type_values, room_status_values = (
            prepare_rooms_dataframe(DATA_PATH)
        )
        df_availability, availability_status_values = (
            prepare_room_availability_dataframe(
                DATA_PATH,
            )
        )

        room_type_def = build_enum_definition(room_type_values)
        bed_type_def = build_enum_definition(bed_type_values)
        room_status_def = build_enum_definition(room_status_values)
        availability_status_def = build_enum_definition(availability_status_values)

        with psycopg.connect(conn_string) as conn:
            setup_rooms_schema(conn, room_type_def, bed_type_def, room_status_def)
            register_room_type_enum(conn)
            insert_rooms_data(conn, df_rooms)

            setup_room_availability_schema(conn, availability_status_def)
            register_room_availability_enum(conn)
            insert_room_availability_data(conn, df_availability)
    except Exception:  # pragma: no cover - retries logged
        logger.exception("Failed to rebuild rooms tables")
        raise


if __name__ == "__main__":
    reload_sql_tables()
