"""Room search against a real database, as the read-only agent role.

Checks what the unit tests in `test_search.py` cannot: that the fixed SQL runs
under `bh_agent_ro`'s grants, returns what the filters ask for, stays bounded,
and that the loaded data has the availability window and floors the prompt
and argument model are built from.

Needs `PGSQL_RO_DB_URL`. Run with the `_EVAL` variables set so
`tests/conftest.py` points it at the Development branch, never Production.
"""

# ruff: noqa: S101

from __future__ import annotations

import asyncio
import datetime as dt
import os
import platform
from typing import TYPE_CHECKING, Any, LiteralString

import psycopg
import pytest

from blue_horizon.agents.booking.db_utils import fetch_rooms_metadata
from blue_horizon.agents.booking.search import (
    RoomsMetadata,
    build_search_args_model,
    run_room_search,
)

if TYPE_CHECKING:
    from pydantic import BaseModel

pytestmark = pytest.mark.db_integration

if platform.system() == "Windows":
    # psycopg's async mode cannot run on Windows's default ProactorEventLoop
    # (the loop `asyncio.run()` otherwise selects). Matches the same fix in
    # `tests/booking/test_write_ops.py`.
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

_DEFAULT_RESULTS = 4
_MAX_RESULTS = 15
_ROOM_FIELDS = frozenset({
    "room_number",
    "floor",
    "type",
    "bed_type",
    "max_occupancy",
    "square_feet",
    "view_types",
    "amenities",
    "accessible",
})
_STAY_FIELDS = frozenset({"nights", "total_price"})


def _ro_url() -> str:
    """Read the read-only URL, skipping when it is not configured.

    Returns:
        The `PGSQL_RO_DB_URL` value.

    """
    url = os.environ.get("PGSQL_RO_DB_URL")
    if not url:
        pytest.skip("PGSQL_RO_DB_URL not set; skipping db_integration test.")
    return url


@pytest.fixture(scope="module")
def meta() -> RoomsMetadata:
    """Load the real rooms metadata once for this module.

    Returns:
        Metadata read through the read-only role.

    """
    return asyncio.run(fetch_rooms_metadata(_ro_url()))


def _search(meta: RoomsMetadata, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
    """Validate arguments and run one search as the read-only role.

    Args:
        meta: Metadata the argument model is built from.
        **kwargs: Raw search arguments.

    Returns:
        The search result.

    """
    model: type[BaseModel] = build_search_args_model(
        meta,
        max_room_numbers=10,
        default_results=_DEFAULT_RESULTS,
        max_results=_MAX_RESULTS,
    )
    args = model.model_validate(kwargs)

    async def _run() -> dict[str, Any]:
        async with await psycopg.AsyncConnection.connect(_ro_url()) as conn:
            return await run_room_search(conn, args, max_results=_MAX_RESULTS)

    return asyncio.run(_run())


def _scalar(sql: LiteralString) -> Any:  # noqa: ANN401
    """Run a one-value query as the read-only role.

    Args:
        sql: A query returning one row with one column.

    Returns:
        That value.

    """
    with psycopg.connect(_ro_url()) as conn:
        row = conn.execute(sql).fetchone()
    assert row is not None
    return row[0]


class TestRoomsMetadata:
    """The loaded data defines the window and floors the tool is built from."""

    def test_window_and_floors(self, meta: RoomsMetadata) -> None:
        """Nights run 2025-01-04 to 2026-01-03 and floors 1 to 20."""
        assert meta.first_night == dt.date(2025, 1, 4)
        assert meta.last_night == dt.date(2026, 1, 3)
        assert meta.last_check_out == dt.date(2026, 1, 4)
        assert (meta.min_floor, meta.max_floor) == (1, 20)

    def test_vocabularies_have_no_nulls(self, meta: RoomsMetadata) -> None:
        """Amenity and view vocabularies are non-empty and NULL-free."""
        assert meta.amenities
        assert "Ocean View" in meta.view_types
        assert all(isinstance(v, str) for v in (*meta.amenities, *meta.view_types))


class TestSearch:
    """Searches return what their filters ask for, and no more."""

    def test_presidential_suite_on_the_top_floor(self, meta: RoomsMetadata) -> None:
        """'Presidential suite on the top floor, Mar 3 to 5' finds only those."""
        result = _search(
            meta,
            check_in="2025-03-03",
            check_out="2025-03-05",
            room_types=["Presidential Suite"],
            min_floor=meta.max_floor,
            max_floor=meta.max_floor,
        )
        assert result["status"] == "ok"
        assert result["rooms"]
        for room in result["rooms"]:
            assert room["type"] == "Presidential Suite"
            assert room["floor"] == meta.max_floor
            assert room["nights"] == 2  # noqa: PLR2004

    def test_matching_count_counts_every_match(self, meta: RoomsMetadata) -> None:
        """An undated count matches a direct count, while rooms stay bounded."""
        result = _search(meta, view_types=["Ocean View"])
        expected = _scalar(
            "SELECT COUNT(*) FROM rooms WHERE view_type && ARRAY['Ocean View']",
        )
        assert result["matching_count"] == expected
        assert len(result["rooms"]) == min(expected, _DEFAULT_RESULTS)

    def test_new_year_stay(self, meta: RoomsMetadata) -> None:
        """A stay ending on the last check-out date runs and counts five nights."""
        result = _search(meta, check_in="2025-12-30", check_out="2026-01-04")
        assert result["status"] == "ok"
        assert all(room["nights"] == 5 for room in result["rooms"])  # noqa: PLR2004

    def test_unfiltered_search_is_bounded(self, meta: RoomsMetadata) -> None:
        """Without a limit, a search returns the default number of rooms."""
        result = _search(meta, check_in="2025-06-01", check_out="2025-06-08")
        assert len(result["rooms"]) == _DEFAULT_RESULTS
        assert result["matching_count"] >= len(result["rooms"])
        assert "limit_note" not in result

    def test_limit_within_maximum(self, meta: RoomsMetadata) -> None:
        """A limit up to the maximum returns that many rooms, with no note."""
        result = _search(
            meta, check_in="2025-06-01", check_out="2025-06-08", limit=_MAX_RESULTS,
        )
        assert len(result["rooms"]) == _MAX_RESULTS
        assert "limit_note" not in result

    def test_limit_above_maximum_is_clamped_and_noted(
        self, meta: RoomsMetadata,
    ) -> None:
        """'Show me every room' returns the maximum and says it was cut."""
        result = _search(
            meta, check_in="2025-06-01", check_out="2025-06-08", limit=500,
        )
        assert len(result["rooms"]) == _MAX_RESULTS
        assert "500" in result["limit_note"]

    def test_amenities_must_all_be_present(self, meta: RoomsMetadata) -> None:
        """Every returned room has every requested amenity."""
        wanted = list(meta.amenities[:2])
        result = _search(meta, amenities=wanted)
        for room in result["rooms"]:
            assert set(wanted) <= set(room["amenities"])

    def test_only_guest_facing_fields(self, meta: RoomsMetadata) -> None:
        """Rooms carry the fixed field set, with stay fields only when dated."""
        undated = _search(meta)
        dated = _search(meta, check_in="2025-03-03", check_out="2025-03-05")
        assert all(set(room) == _ROOM_FIELDS for room in undated["rooms"])
        assert all(
            set(room) == _ROOM_FIELDS | _STAY_FIELDS for room in dated["rooms"]
        )
