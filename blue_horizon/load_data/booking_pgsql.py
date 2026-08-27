"""Utilities to rebuild the PostgreSQL booking schema (rooms, availability,and booking tables)."""  # noqa: E501

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, LiteralString, cast

import pandas as pd
import psycopg
from psycopg.sql import SQL, Composed, Identifier, Literal

from blue_horizon.config import load_app_config
from blue_horizon.load_data import _repo_root

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


def get_data_path() -> Path:
    """Return the configured booking data path.

    Returns:
        Path: Absolute path to the booking data folder.

    """
    app_config = load_app_config()
    return (_repo_root() / app_config.load_data.booking_pgsql.data_path).resolve()


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


def read_pickle_dataframe(path: Path) -> pd.DataFrame:
    """Verify, then unpickle a data file expected to hold a DataFrame.

    ``pd.read_pickle`` can return any pickled object, including a bare
    ``Series`` -- this checks the loaded value against what every pickle in
    this pipeline is documented to hold, rather than trusting the source
    file's shape.

    Args:
        path: Pickle file to load.

    Returns:
        pd.DataFrame: The unpickled DataFrame.

    Raises:
        FileNotFoundError: If the target file is missing.
        TypeError: If the pickle did not contain a DataFrame.

    """
    ensure_pickle_path(path)
    df = pd.read_pickle(path)  # noqa: S301
    if not isinstance(df, pd.DataFrame):
        msg = f"Expected a DataFrame in {path}, got {type(df).__name__}"
        raise TypeError(msg)
    return df


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

CUSTOMERS_COLUMNS: Sequence[str] = [
    "customer_id",
    "first_name",
    "last_name",
]

# There is no email/notification functionality in this project, so the
# remaining source columns customers.pkl has (email, phone, address, ...)
# are never loaded, for any customer. How many customers get the dense low
# `customer_id` block reserved for the UI's automated guest assignment /
# eval-harness identities
# is not hard-coded here -- see `AppConfig.load_data.booking_pgsql.
# seeded_customer_count`'s docstring, and `prepare_customers_dataframe`
# below, for why it must stay a single, shared value rather than a constant
# private to this module.


def get_pgsql_conn_string() -> str:
    """Load the Parent branch's root PGSQL connection string from the environment.

    Deliberately reads ``PGSQL_ROOT_PARENT_DB_URL``, not ``PGSQL_ROOT_DB_URL``:
    `reload_sql_tables` may only run against the Parent branch (see its
    docstring), so its connection string needs a name that can't be confused
    with, or accidentally aliased to, whatever branch a `db_integration` test
    run targets.

    Returns:
        str: The PGSQL connection string stored in PGSQL_ROOT_PARENT_DB_URL.

    Raises:
        RuntimeError: If PGSQL_ROOT_PARENT_DB_URL is unset.

    """
    conn_string = load_app_config().pgsql_root_parent_db_url
    if conn_string is None:
        msg = "PGSQL_ROOT_PARENT_DB_URL is unset."
        raise RuntimeError(msg)
    return conn_string


def build_enum_definition(values: Iterable[str]) -> Composed:
    """Generate an ordered SQL enum definition string.

    Args:
        values: Iterable of enum labels that must be preserved in order.

    Returns:
        Parenthesized, quoted list suitable for CREATE TYPE AS ENUM.

    """
    unique_values = list(set(values))
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
    df = read_pickle_dataframe(rooms_path)
    # pandas-stubs' DataFrame.__getitem__(str) overloads also allow a
    # DataFrame back, to cover duplicate-labelled columns -- not a case that
    # applies to this fixed, flat column layout.
    df["room_id"] = _normalize_room_id_series(cast("pd.Series", df["room_id"]))

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
    df = read_pickle_dataframe(availability_path)
    # pandas-stubs' DataFrame.__getitem__(str) overloads also allow a
    # DataFrame back, to cover duplicate-labelled columns -- not a case that
    # applies to this fixed, flat column layout.
    df["room_id"] = _normalize_room_id_series(cast("pd.Series", df["room_id"]))

    availability_statuses = df["status"].unique().tolist()
    df = df.loc[:, ROOM_AVAIL_COLUMNS]

    return df, availability_statuses


def _map_booking_room_ids(
    df_bookings: pd.DataFrame,
    df_rooms: pd.DataFrame,
) -> pd.DataFrame:
    """Map each booking's `room_number` to the `room_id` primary key `rooms` uses.

    Args:
        df_bookings: Bookings with a `room_number` column.
        df_rooms: Prepared rooms data (see `prepare_rooms_dataframe`), with
            `room_number` and `room_id` columns.

    Returns:
        pd.DataFrame: `df_bookings` with an added integer `room_id` column.

    Raises:
        ValueError: If any `room_number` has no matching `room_id`.

    """
    room_id_by_number = dict(
        zip(
            cast("pd.Series", df_rooms["room_number"]),
            cast("pd.Series", df_rooms["room_id"]),
            strict=True,
        ),
    )
    df_bookings = df_bookings.copy()
    df_bookings["room_id"] = cast(
        "pd.Series", df_bookings["room_number"],
    ).map(room_id_by_number)
    if bool(cast("pd.Series", df_bookings["room_id"]).isna().any()):
        msg = "Some pre-existing bookings reference an unknown room_number."
        raise ValueError(msg)
    df_bookings["room_id"] = df_bookings["room_id"].astype(int)
    return df_bookings


def _availability_date_bounds(
    df_availability: pd.DataFrame,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the half-open ``[min_date, max_date)`` range `room_availability` covers.

    Args:
        df_availability: Prepared availability data (see
            `prepare_room_availability_dataframe`), with a `date` column.

    Returns:
        Tuple of the earliest covered date and the exclusive end of the
        covered range (one day past the latest covered date).

    """
    dates = pd.to_datetime(df_availability["date"])
    return dates.min(), dates.max() + pd.Timedelta(days=1)


def _clamp_booking_dates_to_availability(
    df_bookings: pd.DataFrame,
    df_availability: pd.DataFrame,
) -> pd.DataFrame:
    """Drop bookings entirely outside `room_availability`'s date range, clamp the rest.

    The source booking data spans a wider date range than the pre-generated
    `room_availability` table does. Rows are dropped or clamped here, before
    the overlap resolution below, so nothing needs later reconciliation
    against dates `room_availability` was never generated for.

    Args:
        df_bookings: Bookings with `check_in`/`check_out` columns.
        df_availability: Prepared availability data (see
            `prepare_room_availability_dataframe`).

    Returns:
        pd.DataFrame: Bookings within range, with `check_in`/`check_out`
        clamped to `room_availability`'s covered dates.

    """
    min_date, max_date = _availability_date_bounds(df_availability)
    out_of_range = (df_bookings["check_out"] <= min_date) | (
        df_bookings["check_in"] >= max_date
    )
    dropped = int(out_of_range.sum())
    df_bookings = df_bookings.loc[~out_of_range].copy()
    # No pandas-stubs package is installed (pandas ships no bundled stubs for
    # this method either), so type checkers fall back to their own bundled
    # typeshed snapshot of pandas-stubs -- and CLI pyright and Pylance's
    # bundled pyright disagree on that snapshot's version. A scalar
    # Timestamp bound is correct, ordinary pandas usage for a datetime64
    # Series.clip() at runtime; suppressed rather than contorted to satisfy
    # an incomplete stub.
    df_bookings["check_in"] = df_bookings["check_in"].clip(
        lower=min_date,  # pyright: ignore[reportCallIssue, reportArgumentType]
    )
    df_bookings["check_out"] = df_bookings["check_out"].clip(
        upper=max_date, # pyright: ignore[reportCallIssue, reportArgumentType]
    )
    logger.info(
        "Dropped %s pre-existing bookings entirely outside room_availability's "
        "covered range [%s, %s); clamped check_in/check_out on the remaining "
        "%s rows to that range.",
        dropped, min_date.date(), max_date.date(), len(df_bookings),
    )
    return df_bookings


def _select_non_overlapping_bookings(df: pd.DataFrame) -> pd.Series:
    """Keep the maximum non-overlapping subset of bookings, per room.

    Sorts each room's rows by ``check_out`` and greedily keeps a row whenever
    its ``check_in`` is on or after the ``check_out`` of the last row kept for
    that room. Once sorted, kept intervals for a room are automatically
    ordered by both ``check_in`` and ``check_out``, so comparing only against
    the most recently kept row (rather than every row kept so far) is enough
    to catch any overlap. Sorting by ``check_out`` is the classic
    interval-scheduling greedy: it maximizes the number of rows kept, which
    sorting by ``check_in`` -- or leaving rows in source order -- does not
    guarantee.

    Args:
        df: Bookings with ``room_id``, ``check_in``, and ``check_out``
            columns.

    Returns:
        pd.Series: Boolean mask aligned to `df`'s index, True for rows to
        keep.

    """
    sorted_df = df.sort_values(["room_id", "check_out"])
    last_check_out: dict[int, pd.Timestamp] = {}
    keep = []
    for room_id, check_in, check_out in zip(
        sorted_df["room_id"], sorted_df["check_in"], sorted_df["check_out"],
        strict=True,
    ):
        room_last_check_out = last_check_out.get(room_id)
        kept = room_last_check_out is None or check_in >= room_last_check_out
        keep.append(kept)
        if kept:
            last_check_out[room_id] = check_out
    return pd.Series(keep, index=sorted_df.index).reindex(df.index)


def prepare_accepted_bookings(
    data_path: Path,
    df_rooms: pd.DataFrame,
    df_availability: pd.DataFrame,
) -> pd.DataFrame:
    """Load pre-existing bookings and keep the maximal non-overlapping, in-range subset.

    Every row is loaded as a `confirmed` booking regardless of the source
    `booking_status` (including rows the source data marked `Cancelled`),
    since the pre-generated `room_availability` table already marks those
    rooms/dates `Booked` on account of these reservations. `booking_rooms`
    carries the `booking_rooms_no_overlap` GiST exclusion constraint (see
    `setup_schema`'s `schema.sql`), and the source data does have
    overlapping stays for a given room, so `_select_non_overlapping_bookings`
    keeps only the maximum non-overlapping subset per room before anything
    is inserted.

    Args:
        data_path: Directory containing room_bookings.pkl.
        df_rooms: Prepared rooms data (see `prepare_rooms_dataframe`).
        df_availability: Prepared availability data (see
            `prepare_room_availability_dataframe`).

    Returns:
        pd.DataFrame: Accepted bookings with `source_booking_id`,
        `customer_id` (still the source pickle's ids, not yet remapped to the
        loaded `customers` table -- see `prepare_customers_dataframe`),
        `room_id`, `check_in`, `check_out`, and `total_amount` columns.

    """
    bookings_path = data_path / "room_bookings.pkl"
    df_bookings = read_pickle_dataframe(bookings_path)
    # The pickle's index holds source ids like "BK000001". Keep it under its
    # own name -- the DB-generated `bookings.booking_id` is a different value.
    df_bookings = df_bookings.reset_index(names="source_booking_id")
    df_bookings = _map_booking_room_ids(df_bookings, df_rooms)
    df_bookings = _clamp_booking_dates_to_availability(df_bookings, df_availability)

    accepted_mask = _select_non_overlapping_bookings(df_bookings)
    accepted = df_bookings.loc[accepted_mask].reset_index(drop=True)
    logger.info(
        "%s of %s in-range pre-existing bookings kept (%s skipped for "
        "overlapping another kept booking on the same room).",
        len(accepted), len(df_bookings), len(df_bookings) - len(accepted),
    )
    return accepted


def _select_richest_customer_ids(
    df_accepted_bookings: pd.DataFrame,
    count: int,
) -> list[int]:
    """Pick the source `customer_id`s with the most accepted pre-existing bookings.

    Ties are broken by ascending `customer_id`, so the result is reproducible
    across runs against the same source data.

    Args:
        df_accepted_bookings: Accepted bookings (see
            `prepare_accepted_bookings`), still keyed by the source pickle's
            `customer_id`.
        count: Number of customer ids to select.

    Returns:
        list[int]: Up to `count` source customer ids, sorted ascending.

    """
    booking_counts = (
        df_accepted_bookings.groupby("customer_id")
        .size()
        .to_frame("booking_count")
        .reset_index()
        .sort_values(["booking_count", "customer_id"], ascending=[False, True])
    )
    selected = booking_counts["customer_id"].head(count).astype(int).tolist()
    return sorted(selected)


def prepare_customers_dataframe(
    data_path: Path,
    df_accepted_bookings: pd.DataFrame,
    seeded_customer_count: int,
) -> tuple[pd.DataFrame, dict[int, int]]:
    """Load every customer; remap the richest customers to a dense id block.

    Every customer in `customers.pkl` is loaded, not just
    `seeded_customer_count` of them: most accepted pre-existing bookings
    belong to a customer outside any small subset (see
    `_select_richest_customer_ids`), so loading only a subset leaves too many
    rooms marked `Available` once `reconcile_room_availability_status` runs. The
    `seeded_customer_count` customers with the richest booking history are
    still singled out, remapped to a dense id range
    ``[1, seeded_customer_count]`` (assigned in ascending source-id order),
    and loaded first, because `eval/stress/workload.py` and
    `eval/langsmith_target/target.py` both pick a simulated guest identity by
    assuming exactly that contiguous range, and the UI's automated identity
    picker (`write_ops.list_customers`) should offer guests with real
    booking history to demo against. Every other customer is appended after
    them, remapped to a unique id starting at ``seeded_customer_count + 1``
    (also in ascending source-id order, for reproducibility). Every consumer
    of this range must be passed the *same* `seeded_customer_count` -- see
    `AppConfig.load_data.booking_pgsql.seeded_customer_count`'s docstring for
    why this is a config value rather than a constant private to this
    module.

    Args:
        data_path: Directory containing customers.pkl.
        df_accepted_bookings: Accepted bookings (see
            `prepare_accepted_bookings`), used only to rank customers by
            booking count.
        seeded_customer_count: Number of richest customers to remap into the
            dense low id block. Should be
            `AppConfig.load_data.booking_pgsql.seeded_customer_count` in
            every real call, so it stays in sync with every other reader of
            that range.

    Returns:
        A tuple of the prepared customers DataFrame (columns matching
        `CUSTOMERS_COLUMNS`, the richest `seeded_customer_count` first,
        already using the new ids) and the
        ``{source customer_id: new customer_id}`` remap for every loaded
        customer, for `_remap_bookings_to_new_customer_ids` to apply to the
        bookings themselves.

    """
    richest_ids = _select_richest_customer_ids(
        df_accepted_bookings, seeded_customer_count,
    )

    customers_path = data_path / "customers.pkl"
    df = read_pickle_dataframe(customers_path)

    richest_set = set(richest_ids)
    remaining_ids = sorted(
        source_id for source_id in (int(index) for index in df.index)
        if source_id not in richest_set
    )
    ordered_source_ids = richest_ids + remaining_ids
    new_id_by_source_id = {
        source_id: new_id
        for new_id, source_id in enumerate(ordered_source_ids, start=1)
    }

    # Selecting a list of column labels always yields a DataFrame; cast past
    # the same __getitem__ stub imprecision as above.
    selected = cast(
        "pd.DataFrame", df.loc[ordered_source_ids, ["first_name", "last_name"]],
    ).copy()
    selected["customer_id"] = [
        new_id_by_source_id[source_id] for source_id in selected.index
    ]
    selected = selected.loc[:, CUSTOMERS_COLUMNS].reset_index(drop=True)
    return selected, new_id_by_source_id


def _remap_bookings_to_new_customer_ids(
    df_accepted_bookings: pd.DataFrame,
    new_customer_id_by_source_id: dict[int, int],
) -> pd.DataFrame:
    """Replace each booking's source `customer_id` with its loaded, remapped id.

    `prepare_customers_dataframe` now loads every customer, so every source
    `customer_id` here is expected to have an entry in
    `new_customer_id_by_source_id`. Rows with no entry are still dropped, as
    a defensive backstop against that assumption ever silently breaking
    (e.g. `customers.pkl` and `room_bookings.pkl` regenerated out of sync
    with each other), rather than letting an unmapped row reach the
    `bookings.customer_id` foreign key check and fail the whole reload.

    Args:
        df_accepted_bookings: Accepted bookings (see
            `prepare_accepted_bookings`), still keyed by the source pickle's
            `customer_id`.
        new_customer_id_by_source_id: ``{source customer_id: new customer_id}``
            remap from `prepare_customers_dataframe`, covering every loaded
            customer.

    Returns:
        pd.DataFrame: `df_accepted_bookings` with `customer_id` replaced by
        the new id, minus any row whose customer had no match.

    """
    in_scope = df_accepted_bookings["customer_id"].isin(new_customer_id_by_source_id)
    # Series[bool].sum() resolves to a scalar at runtime; cast past the same
    # pandas-stubs overload imprecision noted elsewhere in this module.
    unmapped = int(cast("int", (~in_scope).sum()))
    if unmapped:
        logger.warning(
            "%s accepted pre-existing bookings reference a customer_id with "
            "no entry in customers.pkl; dropping them.",
            unmapped,
        )
    df_bookings = df_accepted_bookings.loc[in_scope].copy()
    df_bookings["customer_id"] = (
        df_bookings["customer_id"].map(new_customer_id_by_source_id).astype(int)
    )
    return df_bookings.reset_index(drop=True)


def setup_schema(
    conn: psycopg.Connection,
    *,
    room_type_values: Iterable[str],
    bed_type_values: Iterable[str],
    room_status_values: Iterable[str],
    availability_status_values: Iterable[str],
) -> None:
    """Rebuild rooms, room_availability, customers, bookings, and booking_rooms.

    Runs `schema.sql` (next to this module) as one statement, with its four
    enum-label placeholders filled in via `build_enum_definition`. Every
    table is dropped and recreated with CASCADE -- see that file for the
    full per-table rationale, including why `booking_rooms_no_overlap`
    needs the `btree_gist` extension `reload_sql_tables` creates before
    calling this.

    Deliberately excludes the maintenance-booking-guard trigger --
    `setup_maintenance_booking_guard` installs that separately, after
    pre-existing bookings are loaded (see that function's docstring for
    why).

    Args:
        conn: Active database connection.
        room_type_values: Room type enum labels.
        bed_type_values: Room bed type enum labels.
        room_status_values: Room status enum labels.
        availability_status_values: Room availability status enum labels.

    """
    sql_path = Path(__file__).with_name("schema.sql")
    sql_text = cast("LiteralString", sql_path.read_text(encoding="utf-8"))
    conn.execute(
        SQL(sql_text).format(
            room_type_values=build_enum_definition(room_type_values),
            bed_type_values=build_enum_definition(bed_type_values),
            room_status_values=build_enum_definition(room_status_values),
            availability_status_values=build_enum_definition(
                availability_status_values,
            ),
        ),
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


def _insert_bookings_rows(
    cur: psycopg.Cursor, customer_ids: Iterable[int],
) -> list[int]:
    """Insert one `bookings` row per pre-existing booking, returning each new id.

    Uses ``executemany(..., returning=True)`` rather than `copy_dataframe_into_table`:
    each row's `booking_id` is generated by the database and is needed
    immediately after to link that booking's `booking_rooms` rows, which a
    plain `COPY` cannot hand back.

    Args:
        cur: Open cursor inside an active connection.
        customer_ids: One customer id per booking to insert, in order.

    Returns:
        list[int]: The new `booking_id` values, in the same order as
        `customer_ids`.

    Raises:
        RuntimeError: If any ``INSERT ... RETURNING`` statement produced no
            row (unreachable in practice: every single-row insert here
            returns exactly one row).

    """
    cur.executemany(
        "INSERT INTO bookings (customer_id, status) VALUES (%s, 'confirmed') "
        "RETURNING booking_id;",
        [(customer_id,) for customer_id in customer_ids],
        returning=True,
    )
    # cur.results() walks the per-statement result sets executemany(returning=True)
    # produced, in submission order.
    booking_ids: list[int] = []
    for result in cur.results():
        row = result.fetchone()
        if row is None:
            msg = "INSERT INTO bookings ... RETURNING booking_id produced no row."
            raise RuntimeError(msg)
        booking_ids.append(row[0])
    return booking_ids


def _set_confirmation_numbers(cur: psycopg.Cursor) -> None:
    """Set a `BH######` confirmation number on every booking that still lacks one.

    Matches `write_ops._confirmation_number`'s format exactly, so a
    pre-existing booking looks indistinguishable from one made through the
    app.

    Args:
        cur: Open cursor inside an active connection.

    """
    cur.execute(
        "UPDATE bookings "
        "SET confirmation_number = 'BH' || lpad(booking_id::text, 6, '0') "
        "WHERE confirmation_number IS NULL;",
    )


def _insert_booking_rooms_rows(
    cur: psycopg.Cursor,
    df_booking_rooms: pd.DataFrame,
) -> None:
    """Insert one `booking_rooms` row per priced, kept room-stay.

    Args:
        cur: Open cursor inside an active connection.
        df_booking_rooms: Rows with `booking_id`, `room_id`, `check_in`,
            `check_out`, `total_amount` columns, in that order.

    """
    cur.executemany(
        "INSERT INTO booking_rooms "
        "(booking_id, room_id, check_in, check_out, total_amount) "
        "VALUES (%s, %s, %s, %s, %s);",
        df_booking_rooms.itertuples(index=False, name=None),
    )


def insert_bookings_data(conn: psycopg.Connection, df_bookings: pd.DataFrame) -> None:
    """Insert pre-existing bookings into `bookings` and `booking_rooms`.

    Args:
        conn: Active database connection.
        df_bookings: Accepted, clamped, customer-remapped bookings with
            `customer_id`, `room_id`, `check_in`, `check_out`, `total_amount`
            columns (see `prepare_accepted_bookings` and
            `_remap_and_filter_bookings_to_loaded_customers`).

    """
    if df_bookings.empty:
        logger.warning("No pre-existing bookings to insert; skipping.")
        return

    with conn.cursor() as cur:
        booking_ids = _insert_bookings_rows(cur, df_bookings["customer_id"].tolist())
        df_bookings = df_bookings.copy()
        df_bookings["booking_id"] = booking_ids
        _set_confirmation_numbers(cur)

        # Selecting a list of column labels always yields a DataFrame; cast
        # past the same __getitem__ stub imprecision noted above.
        df_booking_rooms = cast(
            "pd.DataFrame",
            df_bookings[
                ["booking_id", "room_id", "check_in", "check_out", "total_amount"]
            ],
        ).copy()
        df_booking_rooms["check_in"] = cast(
            "pd.Series", df_booking_rooms["check_in"],
        ).dt.date
        df_booking_rooms["check_out"] = cast(
            "pd.Series", df_booking_rooms["check_out"],
        ).dt.date
        _insert_booking_rooms_rows(cur, df_booking_rooms)

    logger.info("Inserted %s bookings and their booking_rooms.", len(df_bookings))


def _flip_unmatched_booked_rows_to_available(cur: psycopg.Cursor) -> int:
    """Flip `room_availability` rows marked `Booked` with no matching reservation.

    Args:
        cur: Open cursor inside an active connection.

    Returns:
        int: Number of rows flipped to `Available`.

    """
    cur.execute("""
        UPDATE room_availability AS ra
        SET status = 'Available'
        WHERE ra.status = 'Booked'
          AND NOT EXISTS (
            SELECT 1
            FROM booking_rooms AS br
            WHERE br.room_id = ra.room_id
              AND ra.date >= br.check_in
              AND ra.date < br.check_out
          );
        """)
    return cur.rowcount


def _flip_unmatched_available_rows_to_booked(cur: psycopg.Cursor) -> int:
    """Flip `room_availability` rows marked `Available` despite a matching reservation.

    Args:
        cur: Open cursor inside an active connection.

    Returns:
        int: Number of rows flipped to `Booked`.

    """
    cur.execute("""
        UPDATE room_availability AS ra
        SET status = 'Booked'
        WHERE ra.status = 'Available'
          AND EXISTS (
            SELECT 1
            FROM booking_rooms AS br
            WHERE br.room_id = ra.room_id
              AND ra.date >= br.check_in
              AND ra.date < br.check_out
          );
        """)
    return cur.rowcount


def reconcile_room_availability_status(conn: psycopg.Connection) -> None:
    """Reconcile `room_availability.status` against loaded `booking_rooms`, both ways.

    `prepare_accepted_bookings` skips a large share of the source bookings
    (to satisfy `booking_rooms_no_overlap`), so plenty of rows the
    pre-generated `room_availability` table marks `Booked` no longer have a
    matching reservation -- those are flipped to `Available`. `price` is
    already filled in for `Booked` rows in the seed data (see
    `notebooks/eda.ipynb`, which backfills it from the overlapping
    reservations' `total_amount`), so this flip carries a real price forward
    rather than leaving it `NULL`. The reverse direction is checked too: any
    `Available` row that does have a matching `booking_rooms` reservation is
    flipped to `Booked`, with `price` left untouched -- matching how the live
    app books a room (`write_ops.commit_booking` only ever flips `status`, so
    the nightly rate survives a book-then-cancel round trip).

    Args:
        conn: Active database connection.

    """
    with conn.cursor() as cur:
        flipped_to_available = _flip_unmatched_booked_rows_to_available(cur)
        flipped_to_booked = _flip_unmatched_available_rows_to_booked(cur)
    logger.info(
        "Reconciled room_availability against booking_rooms: %s rows set to "
        "Available, %s rows set to Booked.",
        flipped_to_available, flipped_to_booked,
    )


def setup_maintenance_booking_guard(conn: psycopg.Connection) -> None:
    """Create a trigger refusing `booking_rooms` writes that cover a Maintenance night.

    Runs `maintenance_booking_guard.sql` (next to this module) as one
    statement -- see that file for the full rationale, including why
    `reload_sql_tables` installs this only after pre-existing bookings are
    already loaded, rather than as part of `setup_schema`.

    Args:
        conn: Active database connection.

    """
    sql_path = Path(__file__).with_name("maintenance_booking_guard.sql")
    sql_text = cast("LiteralString", sql_path.read_text(encoding="utf-8"))
    conn.execute(SQL(sql_text))


def regrant_booking_agent_role(conn: psycopg.Connection) -> None:
    """Reapply the `bh_agent_ro`/`bh_agent_rw` grants dropped by table rebuilds.

    `setup_schema` does `DROP TABLE ... CASCADE` on every table it rebuilds,
    which drops any grants that were on it. Without this step, a reload
    silently leaves both booking-agent roles with zero
    privileges on every table until someone notices and reruns the SQL file by
    hand — which is exactly what caused a real `InsufficientPrivilege` failure
    across the whole `db_integration` test suite after a reload with no
    follow-up regrant.

    The exact grants applied are defined once, in
    `regrant_booking_agent_role.sql` next to this module (see that file's
    header for the full per-role rationale); this just executes it so the
    grants can never drift out of sync with a reload again.

    Args:
        conn: Active database connection, authenticated as a role with
            GRANT/REVOKE privileges on the affected tables (i.e. the same
            root connection `reload_sql_tables` already uses).

    Raises:
        FileNotFoundError: If `regrant_booking_agent_role.sql` is missing.

    """
    sql_path = Path(__file__).with_name("regrant_booking_agent_role.sql")
    # SQL() requires a LiteralString because it does no escaping -- a
    # deliberate guard against composing untrusted/dynamic SQL text. This
    # text is a repo-committed script read verbatim, not untrusted input, so
    # the cast documents that this is exactly the case the guard is fine with.
    sql_text = cast("LiteralString", sql_path.read_text(encoding="utf-8"))
    conn.execute(SQL(sql_text))


def reload_sql_tables() -> None:
    """Prepare data and rebuild the room, availability, customer, and booking tables.

    The function reads pickled source data, reconstructs the required enums,
    and inserts records into `rooms` and `room_availability`. It also
    (re)creates `customers`, `bookings`, and `booking_rooms`, and loads all
    three: `customers` gets every source customer (see
    `prepare_customers_dataframe`), with the
    `AppConfig.load_data.booking_pgsql.seeded_customer_count` richest by
    accepted pre-existing booking history remapped to a dense low id range
    and loaded first, so the UI's automated identity picker and the
    eval/stress harnesses' simulated-guest selection keep working unchanged;
    `bookings` and `booking_rooms` get pre-existing
    reservations for every loaded customer (see `prepare_accepted_bookings`),
    clamped to the date range `room_availability` covers and filtered down to
    the maximum non-overlapping subset per room, since `booking_rooms`
    carries a GiST exclusion constraint (see `setup_schema`'s `schema.sql`)
    that makes a double-booking impossible to insert, so this also creates
    the `btree_gist` extension that constraint depends on.

    Loading real bookings can leave `room_availability` rows marked `Booked`
    with no matching `booking_rooms` reservation, or `Available` despite one
    existing; `reconcile_room_availability_status` fixes both directions.
    `setup_maintenance_booking_guard` then adds a trigger refusing any
    `booking_rooms` write that would cover a night `room_availability` marks
    `Maintenance`, as a schema-level backstop alongside the GiST constraint --
    installed only now, after the load above, so it never rejects one of
    these pre-existing bookings itself (see that function's docstring).

    Finally, it reapplies the `bh_agent_ro`/`bh_agent_rw` grants that the
    table rebuilds dropped, via `regrant_booking_agent_role`.

    This function drops and recreates every table it touches, so it must be
    run against a branch's parent, never a branch that gets reset in place —
    otherwise the next reset restores an empty `customers` table and every
    booking fails on a foreign-key violation.
    """
    try:
        conn_string = get_pgsql_conn_string()
        seeded_customer_count = (
            load_app_config().load_data.booking_pgsql.seeded_customer_count
        )
        df_rooms, room_type_values, bed_type_values, room_status_values = (
            prepare_rooms_dataframe(get_data_path())
        )
        df_availability, availability_status_values = (
            prepare_room_availability_dataframe(
                get_data_path(),
            )
        )
        df_accepted_bookings = prepare_accepted_bookings(
            get_data_path(), df_rooms, df_availability,
        )
        df_customers, new_customer_id_by_source_id = prepare_customers_dataframe(
            get_data_path(), df_accepted_bookings, seeded_customer_count,
        )
        df_bookings = _remap_bookings_to_new_customer_ids(
            df_accepted_bookings, new_customer_id_by_source_id,
        )

        with psycopg.connect(conn_string) as conn:
            conn.execute("SET search_path TO public;")
            conn.execute("CREATE EXTENSION IF NOT EXISTS btree_gist;")
            setup_schema(
                conn,
                room_type_values=room_type_values,
                bed_type_values=bed_type_values,
                room_status_values=room_status_values,
                availability_status_values=availability_status_values,
            )

            copy_dataframe_into_table(
                conn, table_name="rooms", columns=ROOMS_COLUMNS, df=df_rooms,
            )
            copy_dataframe_into_table(
                conn,
                table_name="room_availability",
                columns=ROOM_AVAIL_COLUMNS,
                df=df_availability,
            )
            copy_dataframe_into_table(
                conn,
                table_name="customers",
                columns=CUSTOMERS_COLUMNS,
                df=df_customers,
            )
            insert_bookings_data(conn, df_bookings)
            reconcile_room_availability_status(conn)
            setup_maintenance_booking_guard(conn)

            regrant_booking_agent_role(conn)
    except Exception:  # pragma: no cover - retries logged
        logger.exception("Failed to rebuild booking tables")
        raise


if __name__ == "__main__":
    reload_sql_tables()
