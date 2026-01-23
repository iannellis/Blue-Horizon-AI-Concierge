"""Schema-per-case helpers for evaluation DB resets."""

import logging
from enum import Enum
from pathlib import Path
from typing import Any, LiteralString, cast

import pandas as pd
from psycopg import sql
from psycopg_pool import AsyncConnectionPool

from load_data.rooms_pgsql import (
    ROOM_AVAIL_COLUMNS,
    ROOMS_COLUMNS,
    build_enum_definition,
    prepare_room_availability_dataframe,
    prepare_rooms_dataframe,
)

logger = logging.getLogger(__name__)


def _enum_to_scalar(value: Any) -> Any:  # noqa: ANN401
    if isinstance(value, Enum):
        return value.value
    return value


def _normalize_enum_columns(
    df: pd.DataFrame,
    *,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    normalized = df.copy()
    for column in columns:
        normalized[column] = normalized[column].apply(_enum_to_scalar)
    return normalized


async def create_case_schema(
    *,
    pool: AsyncConnectionPool[Any],
    schema: str,
    data_path: str | Path,
) -> None:
    """Prune, create, and seed a schema scoped to a single evaluation case."""
    path = Path(data_path)
    df_rooms, room_type_values, bed_type_values, room_status_values = (
        prepare_rooms_dataframe(
            path,
        )
    )
    df_avail, availability_status_values = prepare_room_availability_dataframe(path)

    room_type_def = build_enum_definition(room_type_values)
    room_bed_type_def = build_enum_definition(bed_type_values)
    room_status_def = build_enum_definition(room_status_values)
    availability_status_def = build_enum_definition(availability_status_values)

    normalized_rooms = _normalize_enum_columns(
        df_rooms,
        columns=("type", "bed_type", "status"),
    )
    normalized_avail = _normalize_enum_columns(
        df_avail,
        columns=("status",),
    )

    rooms_rows = normalized_rooms[ROOMS_COLUMNS].to_numpy().tolist()
    avail_rows = normalized_avail[ROOM_AVAIL_COLUMNS].to_numpy().tolist()

    schema_ident = sql.Identifier(schema)
    schema_created = False

    schema_check = sql.SQL(
        "SELECT 1 FROM information_schema.schemata WHERE schema_name = %s",
    )

    rooms_insert_sql = cast(
        "LiteralString",
        """
        INSERT INTO rooms (
            room_id,
            room_number,
            floor,
            type,
            square_feet,
            basic_amenities,
            additional_amenities,
            max_occupancy,
            bed_type,
            view_type,
            accessibility,
            status,
            last_renovation,
            base_rate,
            max_rate
        ) VALUES (
            %s,
            %s,
            %s,
            %s::room_type,
            %s,
            %s,
            %s,
            %s,
            %s::room_bed_type,
            %s,
            %s,
            %s::room_status_type,
            %s,
            %s,
            %s
        );
        """,
    )

    avail_insert_sql = cast(
        "LiteralString",
        """
        INSERT INTO room_availability (
            room_id,
            room_number,
            date,
            status,
            price,
            max_occupancy
        ) VALUES (
            %s,
            %s,
            %s,
            %s::availability_status_type,
            %s,
            %s
        );
        """,
    )

    try:
        async with pool.connection() as conn, conn.transaction():
                exists = await conn.fetchval(schema_check, (schema,))
                if exists is None:
                    schema_created = True

                await conn.execute(
                    sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(schema_ident),
                )
                await conn.execute(
                    sql.SQL("SET search_path TO {}").format(schema_ident),
                )

                await conn.execute("DROP TABLE IF EXISTS room_availability CASCADE;")
                await conn.execute("DROP TABLE IF EXISTS rooms CASCADE;")
                await conn.execute(
                    "DROP TYPE IF EXISTS availability_status_type CASCADE;",
                )
                await conn.execute("DROP TYPE IF EXISTS room_type CASCADE;")
                await conn.execute("DROP TYPE IF EXISTS room_bed_type CASCADE;")
                await conn.execute("DROP TYPE IF EXISTS room_status_type CASCADE;")

                await conn.execute(
                    cast(
                        "LiteralString",
                        f"CREATE TYPE room_type AS ENUM {room_type_def};",
                    ),
                )
                await conn.execute(
                    cast(
                        "LiteralString",
                        f"CREATE TYPE room_bed_type AS ENUM {room_bed_type_def};",
                    ),
                )
                await conn.execute(
                    cast(
                        "LiteralString",
                        f"CREATE TYPE room_status_type AS ENUM {room_status_def};",
                    ),
                )
                await conn.execute(
                    cast(
                        "LiteralString", (
                        "CREATE TYPE availability_status_type AS ENUM "
                        f"{availability_status_def};",
                        ),
                    ),
                )

                await conn.execute(
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

                await conn.execute(
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
                    """,  # noqa: E501
                )

                if rooms_rows:
                    await conn.executemany(rooms_insert_sql, rooms_rows)
                if avail_rows:
                    await conn.executemany(avail_insert_sql, avail_rows)

    except Exception:
        if schema_created:
            try:
                await drop_case_schema(pool=pool, schema=schema)
            except Exception:
                logger.exception("Failed to drop schema %s during cleanup", schema)
        raise


async def drop_case_schema(
    *,
    pool: AsyncConnectionPool[Any],
    schema: str,
) -> None:
    """Remove a case schema and its contents."""
    schema_ident = sql.Identifier(schema)
    async with pool.connection() as conn:
        await conn.execute(
            sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(schema_ident),
        )
