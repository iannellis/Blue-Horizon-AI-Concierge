"""Typed room search: the only way the booking model reads the rooms tables.

The model never writes SQL. It fills in a typed argument model whose allowed
values come from the database at startup, and this module runs one of two
fixed, parameterized queries with those values. Every result is bounded: the
model may ask for a number of rooms, but no more than a fixed maximum come
back, each with a fixed set of fields, plus a `matching_count` so "how many"
questions need no second query.

Nothing here imports LangChain, so a transport other than the booking agent's
tool-calling loop (an MCP server, for example) can wrap the same functions.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Literal

from psycopg import sql
from psycopg.rows import dict_row
from pydantic import BaseModel, Field, create_model, model_validator

from blue_horizon.agents.booking.write_ops import fmt_money

if TYPE_CHECKING:
    from collections.abc import Sequence

    from psycopg import AsyncConnection

SortBy = Literal["price", "floor_desc", "square_feet_desc", "room_number"]


@dataclass(frozen=True)
class RoomsMetadata:
    """Search vocabulary and bounds read from the database at startup.

    Attributes:
        enum_values: Mapping of Postgres enum type name to its values.
        amenities: Sorted union of basic and additional amenity values.
        view_types: Sorted distinct view type values.
        min_floor: Lowest floor in the hotel.
        max_floor: Highest (top) floor in the hotel.
        first_night: Earliest night with availability data.
        last_night: Latest night with availability data.

    """

    enum_values: dict[str, list[str]]
    amenities: tuple[str, ...]
    view_types: tuple[str, ...]
    min_floor: int
    max_floor: int
    first_night: dt.date
    last_night: dt.date

    @property
    def last_check_out(self) -> dt.date:
        """Return the latest valid check-out date.

        Returns:
            The day after `last_night`, since check-out is exclusive.

        """
        return self.last_night + dt.timedelta(days=1)


# Room columns every search returns. Internal identifiers, the room-level
# status, rate columns, and renovation dates are deliberately absent.
_ROOM_COLUMNS: Final = sql.SQL("""
    r.room_number,
    r.floor,
    r.type::text AS type,
    r.bed_type::text AS bed_type,
    r.max_occupancy,
    r.square_feet,
    r.view_type AS view_types,
    array_remove(
        COALESCE(r.basic_amenities, '{}') || COALESCE(r.additional_amenities, '{}'),
        NULL
    ) AS amenities,
    r.accessibility AS accessible,
    COUNT(*) OVER () AS matching_count""")

# Each optional filter is a no-op when its parameter is NULL.
_ROOM_FILTERS: Final = sql.SQL("""
    (%(room_numbers)s::int[] IS NULL
        OR r.room_number = ANY(%(room_numbers)s::int[]))
    AND (%(room_types)s::text[] IS NULL OR r.type::text = ANY(%(room_types)s::text[]))
    AND (%(bed_types)s::text[] IS NULL
        OR r.bed_type::text = ANY(%(bed_types)s::text[]))
    AND (%(amenities)s::text[] IS NULL
        OR COALESCE(r.basic_amenities, '{}') || COALESCE(r.additional_amenities, '{}')
            @> %(amenities)s::text[])
    AND (%(view_types)s::text[] IS NULL OR r.view_type && %(view_types)s::text[])
    AND (%(min_floor)s::int IS NULL OR r.floor >= %(min_floor)s::int)
    AND (%(max_floor)s::int IS NULL OR r.floor <= %(max_floor)s::int)
    AND (%(min_occupancy)s::int IS NULL OR r.max_occupancy >= %(min_occupancy)s::int)
    AND (%(min_square_feet)s::int IS NULL
        OR r.square_feet >= %(min_square_feet)s::int)
    AND (%(accessible)s::boolean IS NULL
        OR r.accessibility = %(accessible)s::boolean)""")

# A room qualifies for a stay only when every night in [check_in, check_out)
# is Available and priced. UNIQUE(room_id, date) makes COUNT(*) the night count.
# MATERIALIZED stops the planner inlining the CTE into a nested loop that
# re-aggregates room_availability once per candidate room: 630 ms against
# 22 ms for a three-night top-floor search.
_DATED_SEARCH_SQL: Final = sql.SQL("""
WITH stay AS MATERIALIZED (
    SELECT
        ra.room_id,
        COUNT(*) AS nights,
        SUM(ra.price) AS total_price,
        MAX(ra.price) AS max_price
    FROM room_availability AS ra
    WHERE ra.date >= %(check_in)s::date
        AND ra.date < %(check_out)s::date
        AND ra.status = 'Available'
        AND ra.price IS NOT NULL
    GROUP BY ra.room_id
    HAVING COUNT(*) = %(check_out)s::date - %(check_in)s::date
)
SELECT {columns}, s.nights, s.total_price
FROM rooms AS r
JOIN stay AS s ON s.room_id = r.room_id
WHERE {filters}
    AND (%(max_nightly_price)s::numeric IS NULL
        OR s.max_price <= %(max_nightly_price)s::numeric)
ORDER BY {order_by}
LIMIT %(limit)s""")

_UNDATED_SEARCH_SQL: Final = sql.SQL("""
SELECT {columns}
FROM rooms AS r
WHERE {filters}
ORDER BY {order_by}
LIMIT %(limit)s""")

# Price ordering needs a stay; without dates it falls back to room number.
_ORDER_BY: Final[dict[tuple[SortBy, bool], sql.SQL]] = {
    ("price", True): sql.SQL("s.total_price, r.room_number"),
    ("price", False): sql.SQL("r.room_number"),
    ("floor_desc", True): sql.SQL("r.floor DESC, s.total_price, r.room_number"),
    ("floor_desc", False): sql.SQL("r.floor DESC, r.room_number"),
    ("square_feet_desc", True): sql.SQL(
        "r.square_feet DESC NULLS LAST, s.total_price, r.room_number",
    ),
    ("square_feet_desc", False): sql.SQL(
        "r.square_feet DESC NULLS LAST, r.room_number",
    ),
    ("room_number", True): sql.SQL("r.room_number"),
    ("room_number", False): sql.SQL("r.room_number"),
}


def build_search_args_model(
    meta: RoomsMetadata,
    *,
    max_room_numbers: int,
    default_results: int,
    max_results: int,
) -> type[BaseModel]:
    """Build the `search_rooms` argument model from database metadata.

    Categorical fields are `Literal` types over the values the database
    actually holds, so the JSON schema the model sees lists every legal value
    and an unknown one is rejected before any SQL runs.

    `limit` has no upper bound here. A request above `max_results` is clamped
    by `run_room_search`, which says so in its result, rather than rejected,
    so a guest asking for "every room" still gets rooms back.

    Args:
        meta: Search vocabulary and bounds loaded at startup.
        max_room_numbers: Maximum length of the `room_numbers` list.
        default_results: Rooms returned when the model does not set `limit`.
        max_results: Most rooms one search returns, quoted in `limit`'s
            description.

    Returns:
        A Pydantic model class whose instances are valid search arguments.

    """
    window = (
        f"Nights {meta.first_night} through {meta.last_night} can be searched; "
        f"the latest check-out is {meta.last_check_out}."
    )
    room_type = _literal(meta.enum_values.get("room_type", []))
    bed_type = _literal(meta.enum_values.get("room_bed_type", []))
    amenity = _literal(meta.amenities)
    view_type = _literal(meta.view_types)
    floor_bounds: dict[str, Any] = {"ge": meta.min_floor, "le": meta.max_floor}

    return create_model(
        "SearchRoomsArgs",
        __doc__="Filters for a room search. Omit a filter to leave it unrestricted.",
        __validators__={"_check_consistency": _consistency_validator(meta)},
        check_in=(
            dt.date | None,
            Field(
                default=None,
                description=(
                    f"First night of the stay (ISO date). {window} "
                    "Stays may cross into the next year. Requires check_out."
                ),
            ),
        ),
        check_out=(
            dt.date | None,
            Field(
                default=None,
                description=(
                    "Check-out date (ISO date), exclusive: the guest does not stay "
                    "that night. Requires check_in. Omit both dates for questions "
                    "about rooms in general."
                ),
            ),
        ),
        room_numbers=(
            list[int] | None,
            Field(
                default=None,
                max_length=max_room_numbers,
                description="Only these room numbers.",
            ),
        ),
        room_types=(
            list[room_type] | None,
            Field(default=None, description="Any of these room types."),
        ),
        bed_types=(
            list[bed_type] | None,
            Field(default=None, description="Any of these bed types."),
        ),
        amenities=(
            list[amenity] | None,
            Field(default=None, description="The room must have all of these."),
        ),
        view_types=(
            list[view_type] | None,
            Field(default=None, description="The room must have any of these."),
        ),
        min_floor=(
            int | None,
            Field(
                default=None,
                **floor_bounds,
                description=(
                    f"Lowest acceptable floor. Floors run {meta.min_floor} to "
                    f"{meta.max_floor}."
                ),
            ),
        ),
        max_floor=(
            int | None,
            Field(
                default=None,
                **floor_bounds,
                description=(
                    f"Highest acceptable floor. The top floor is {meta.max_floor}; "
                    "set min_floor and max_floor to it for 'top floor'."
                ),
            ),
        ),
        min_occupancy=(
            int | None,
            Field(default=None, ge=1, description="Minimum guests the room sleeps."),
        ),
        min_square_feet=(
            int | None,
            Field(default=None, ge=1, description="Minimum room size in square feet."),
        ),
        accessible=(
            bool | None,
            Field(default=None, description="Require (true) or exclude (false)."),
        ),
        max_nightly_price=(
            float | None,
            Field(
                default=None,
                gt=0,
                description="Every night must cost at most this. Requires dates.",
            ),
        ),
        limit=(
            int,
            Field(
                default=default_results,
                ge=1,
                description=(
                    f"How many rooms to return, {default_results} by default. At "
                    f"most {max_results} come back; a larger value is reduced to "
                    f"{max_results}. matching_count always counts every match."
                ),
            ),
        ),
        sort_by=(
            SortBy,
            Field(
                default="price",
                description=(
                    "Result order. 'price' is cheapest stay first (room number "
                    "without dates); the others are highest floor, largest room, "
                    "and room number."
                ),
            ),
        ),
    )


def _literal(values: Sequence[str]) -> Any:  # noqa: ANN401
    """Build a `Literal` type over values known only at runtime.

    Args:
        values: Allowed string values; must be non-empty.

    Returns:
        The `Literal[...]` type admitting exactly `values`.

    Raises:
        ValueError: If `values` is empty, since `Literal[()]` is not a type.

    """
    if not values:
        msg = "Cannot build a search vocabulary from an empty value list"
        raise ValueError(msg)
    return Literal[tuple(values)]  # pyright: ignore[reportInvalidTypeArguments]


def _consistency_validator(meta: RoomsMetadata) -> Any:  # noqa: ANN401
    """Build the cross-field validator for the search argument model.

    Args:
        meta: Supplies the bookable date window quoted in error messages.

    Returns:
        A Pydantic `model_validator(mode="after")` for `create_model`.

    """
    window = (
        f"Availability covers nights {meta.first_night} through {meta.last_night} "
        f"(latest check-out {meta.last_check_out})."
    )

    def check(args: Any) -> Any:  # noqa: ANN401
        """Reject argument combinations no query can honor.

        Args:
            args: The validated search arguments.

        Returns:
            `args`, unchanged.

        Raises:
            ValueError: If the dates, price, or floor range are inconsistent.

        """
        check_in: dt.date | None = args.check_in
        check_out: dt.date | None = args.check_out
        if (check_in is None) != (check_out is None):
            msg = "Give both check_in and check_out, or neither."
            raise ValueError(msg)
        if check_in is not None and check_out is not None:
            if check_out <= check_in:
                msg = "check_out must be after check_in."
                raise ValueError(msg)
            if check_in < meta.first_night or check_out > meta.last_check_out:
                msg = f"Dates {check_in} to {check_out} are out of range. {window}"
                raise ValueError(msg)
        elif args.max_nightly_price is not None:
            msg = "max_nightly_price needs check_in and check_out."
            raise ValueError(msg)
        if (
            args.min_floor is not None
            and args.max_floor is not None
            and args.min_floor > args.max_floor
        ):
            msg = "min_floor cannot be above max_floor."
            raise ValueError(msg)
        return args

    return model_validator(mode="after")(check)


async def run_room_search(
    conn: AsyncConnection[Any],
    args: BaseModel,
    *,
    max_results: int,
) -> dict[str, Any]:
    """Run a room search on an open connection.

    Args:
        conn: Connection to run the query on. The caller owns its lifetime,
            transaction, and retry policy.
        args: An instance of the model from `build_search_args_model`.
        max_results: Most rooms to return, whatever `args.limit` asks for.

    Returns:
        `{"status": "ok", "matching_count": n, "rooms": [...]}`, where `n`
        counts every matching room and `rooms` holds at most
        `min(args.limit, max_results)` of them. Dated searches add `nights`
        and `total_price` to each room. When `args.limit` exceeds
        `max_results`, a `limit_note` says the result was cut to
        `max_results`.

    """
    params = _query_params(args, max_results=max_results)
    dated = params["check_in"] is not None
    query = select_search_query(sort_by=params["sort_by"], dated=dated)
    async with conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(query, params)
        rows = await cur.fetchall()

    matching_count = int(rows[0]["matching_count"]) if rows else 0
    result: dict[str, Any] = {
        "status": "ok",
        "matching_count": matching_count,
        "rooms": [_room_result(row) for row in rows],
    }
    requested: int = args.limit  # pyright: ignore[reportAttributeAccessIssue]
    if requested > max_results:
        result["limit_note"] = _limit_note(requested, max_results)
    return result


def _query_params(args: BaseModel, *, max_results: int) -> dict[str, Any]:
    """Turn search arguments into query parameters.

    An empty list means "no restriction", the same as an omitted filter.

    Args:
        args: Validated search arguments.
        max_results: Ceiling applied to the requested `limit`.

    Returns:
        Mapping of every named placeholder in the search SQL to its value.

    """
    params = args.model_dump()
    for key, value in params.items():
        if isinstance(value, list) and not value:
            params[key] = None
    params["limit"] = min(params["limit"], max_results)
    return params


def select_search_query(*, sort_by: SortBy, dated: bool) -> sql.Composed:
    """Compose the fixed search query for an ordering and date mode.

    Args:
        sort_by: Requested result order.
        dated: Whether the search has a check-in and check-out.

    Returns:
        The complete parameterized query.

    """
    template = _DATED_SEARCH_SQL if dated else _UNDATED_SEARCH_SQL
    return template.format(
        columns=_ROOM_COLUMNS,
        filters=_ROOM_FILTERS,
        order_by=_ORDER_BY[sort_by, dated],
    )


def _room_result(row: dict[str, Any]) -> dict[str, Any]:
    """Shape one result row for the model.

    Args:
        row: A row from the search query.

    Returns:
        JSON-safe room fields, with `nights` and `total_price` on dated rows.

    """
    room = {k: v for k, v in row.items() if k not in {"matching_count", "total_price"}}
    if "total_price" in row:
        room["total_price"] = fmt_money(row["total_price"])
    return room


def _limit_note(requested: int, max_results: int) -> str:
    """Explain that a search returned fewer rooms than were asked for.

    The instruction to pass this on lives here, in the result, rather than in
    a system prompt, because an MCP client brings its own prompt.

    Args:
        requested: The `limit` the caller asked for.
        max_results: The most rooms one search returns.

    Returns:
        A note the calling model should relay to the user.

    """
    return (
        f"{requested} rooms were requested, but one search returns at most "
        f"{max_results}, so no more than {max_results} are listed. Tell the user "
        "that the list was cut to this maximum."
    )
