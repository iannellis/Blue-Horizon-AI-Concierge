"""Utilities to rebuild the PostgreSQL rooms and availability tables from pickled data."""  # noqa: E501

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

import pandas as pd
import psycopg
from psycopg.sql import SQL, Composed, Identifier, Literal

from blue_horizon.config import load_app_config
from blue_horizon.load_data import _repo_root

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path


def get_data_path() -> Path:
    """Return the configured rooms data path.

    Returns:
        Path: Absolute path to the rooms data folder.

    """
    app_config = load_app_config()
    return (_repo_root() / app_config.load_data.rooms_pgsql.data_path).resolve()


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


def get_pgsql_conn_string() -> str:
    """Load the PGSQL connection string from the environment.

    Returns:
        str: The PGSQL connection string stored in PGSQL_DB_URL.

    Raises:
        RuntimeError: If PGSQL_DB_URL is unset.

    """
    return load_app_config().pgsql_db_url


def build_enum_definition(values: Iterable[str]) -> Composed:
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

    values_sql = SQL(", ").join(Literal(value) for value in unique_values)
    return SQL("({values})").format(values=values_sql)


def _normalize_room_id_series(room_ids: pd.Series) -> pd.Series:
    """Convert source room IDs like ``RM101`` into integer IDs.

    Args:
        room_ids: Source room ID series.

    Returns:
        pd.Series: Integer room IDs.

    """
    return room_ids.astype(str).str[2:].astype(int)


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
    df["room_id"] = _normalize_room_id_series(df["room_id"])

    room_types = df["type"].unique().tolist()
    room_bed_types = df["bed_type"].unique().tolist()
    room_statuses = df["status"].unique().tolist()

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
    df["room_id"] = _normalize_room_id_series(df["room_id"])

    availability_statuses = df["status"].unique().tolist()
    df = df.loc[:, ROOM_AVAIL_COLUMNS]

    return df, availability_statuses


def setup_rooms_schema(
    conn: psycopg.Connection,
    room_type_def: Composed,
    bed_type_def: Composed,
    status_def: Composed,
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

        cur.execute(
            SQL("CREATE TYPE room_type AS ENUM {values};").format(values=room_type_def),
        )
        cur.execute(
            SQL("CREATE TYPE room_bed_type AS ENUM {values};").format(
                values=bed_type_def,
            ),
        )
        cur.execute(
            SQL("CREATE TYPE room_status_type AS ENUM {values};").format(
                values=status_def,
            ),
        )

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


def copy_dataframe_into_table(
    conn: psycopg.Connection,
    *,
    table_name: str,
    columns: Sequence[str],
    df: pd.DataFrame,
) -> None:
    """Bulk load a DataFrame into PostgreSQL using ``COPY FROM STDIN``.

    Args:
        conn: Active database connection.
        table_name: Target table name.
        columns: Ordered table columns to populate.
        df: DataFrame containing one column per entry in ``columns``.

    """
    if df.empty:
        logger.warning("No %s records to insert; skipping.", table_name)
        return

    columns_sql = SQL(", ").join(Identifier(column) for column in columns)
    copy_sql = SQL("COPY {table_name} ({columns}) FROM STDIN").format(
        table_name=Identifier(table_name),
        columns=columns_sql,
    )

    with conn.cursor() as cur, cur.copy(copy_sql) as copy:
        for row in df.loc[:, list(columns)].itertuples(index=False, name=None):
            copy.write_row(row)

    logger.info("Inserted %s data.", table_name)


def insert_rooms_data(conn: psycopg.Connection, df_rooms: pd.DataFrame) -> None:
    """Insert normalized rooms data into the database.

    Args:
        conn: Active database connection.
        df_rooms: DataFrame keyed to `ROOMS_COLUMNS`.

    """
    copy_dataframe_into_table(
        conn,
        table_name="rooms",
        columns=ROOMS_COLUMNS,
        df=df_rooms,
    )


def setup_room_availability_schema(
    conn: psycopg.Connection,
    status_def: Composed,
) -> None:
    """Reset availability enums and recreate the `room_availability` table.

    Args:
        conn: Active database connection.
        status_def: Enum definition string for availability_status_type.

    """
    with conn.cursor() as cur:
        cur.execute(SQL("DROP TABLE IF EXISTS room_availability;"))
        cur.execute(SQL("DROP TYPE IF EXISTS availability_status_type CASCADE;"))
        cur.execute(
            SQL("CREATE TYPE availability_status_type AS ENUM {values};").format(
                values=status_def,
            ),
        )

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


def insert_room_availability_data(
    conn: psycopg.Connection,
    df_availability: pd.DataFrame,
) -> None:
    """Insert normalized availability records into `room_availability`.

    Args:
        conn: Active database connection.
        df_availability: DataFrame keyed to `ROOM_AVAIL_COLUMNS`.

    """
    copy_dataframe_into_table(
        conn,
        table_name="room_availability",
        columns=ROOM_AVAIL_COLUMNS,
        df=df_availability,
    )


def reload_sql_tables() -> None:
    """Prepare data and rebuild the room and availability tables.

    The function reads pickled source data, reconstructs the required enums,
    and inserts records into both `rooms` and `room_availability`.
    """
    try:
        conn_string = get_pgsql_conn_string()
        df_rooms, room_type_values, bed_type_values, room_status_values = (
            prepare_rooms_dataframe(get_data_path())
        )
        df_availability, availability_status_values = (
            prepare_room_availability_dataframe(
                get_data_path(),
            )
        )

        room_type_def = build_enum_definition(room_type_values)
        bed_type_def = build_enum_definition(bed_type_values)
        room_status_def = build_enum_definition(room_status_values)
        availability_status_def = build_enum_definition(availability_status_values)

        with psycopg.connect(conn_string) as conn:
            conn.execute("SET search_path TO public;")
            setup_rooms_schema(conn, room_type_def, bed_type_def, room_status_def)
            insert_rooms_data(conn, df_rooms)

            setup_room_availability_schema(conn, availability_status_def)
            insert_room_availability_data(conn, df_availability)
    except Exception:  # pragma: no cover - retries logged
        logger.exception("Failed to rebuild rooms tables")
        raise


if __name__ == "__main__":
    reload_sql_tables()
