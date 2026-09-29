"""Unit tests for the `search_rooms` argument model and query selection.

No database and no searches. The metadata is built by hand in
`tests/booking/_search_fixtures.py`, so the edge dates below are that
fixture's window, not values read from anywhere. These tests only check
which arguments the model accepts and rejects, and which fixed query a
given ordering selects. Running real searches is `test_search_db.py`'s job.
"""

# ruff: noqa: S101

from __future__ import annotations

import datetime as dt
from typing import Any, get_args

import pytest
from pydantic import BaseModel, ValidationError

from blue_horizon.agents.booking.search import (
    SortBy,
    _limit_note,
    _query_params,
    build_search_args_model,
    select_search_query,
)
from tests.booking._search_fixtures import (
    DEFAULT_RESULTS,
    FIRST_NIGHT,
    LAST_NIGHT,
    MAX_RESULTS,
    MAX_ROOM_NUMBERS,
    make_rooms_metadata,
)

_ONE_DAY = dt.timedelta(days=1)


@pytest.fixture(scope="module")
def args_model() -> type[BaseModel]:
    """Build the argument model from hand-made metadata.

    Returns:
        The `search_rooms` argument model.

    """
    return build_search_args_model(
        make_rooms_metadata(),
        max_room_numbers=MAX_ROOM_NUMBERS,
        default_results=DEFAULT_RESULTS,
        max_results=MAX_RESULTS,
    )


def _validate(model: type[BaseModel], **kwargs: Any) -> BaseModel:  # noqa: ANN401
    """Validate keyword arguments the way the tool layer does.

    Args:
        model: The argument model.
        **kwargs: Raw arguments, as a model would send them.

    Returns:
        The validated arguments.

    """
    return model.model_validate(kwargs)


class TestDateWindow:
    """Dates must fall inside the availability window, which crosses New Year."""

    def test_first_night_accepted(self, args_model: type[BaseModel]) -> None:
        """A stay starting on the first night is valid."""
        _validate(
            args_model, check_in=FIRST_NIGHT, check_out=FIRST_NIGHT + _ONE_DAY,
        )

    def test_night_before_window_rejected(self, args_model: type[BaseModel]) -> None:
        """A stay starting the day before the window is rejected, naming it."""
        with pytest.raises(ValidationError, match="2025-01-04"):
            _validate(
                args_model,
                check_in=FIRST_NIGHT - _ONE_DAY,
                check_out=FIRST_NIGHT + _ONE_DAY,
            )

    def test_last_check_out_accepted(self, args_model: type[BaseModel]) -> None:
        """Checking out the day after the last night is valid."""
        _validate(args_model, check_in=LAST_NIGHT, check_out=LAST_NIGHT + _ONE_DAY)

    def test_check_out_past_window_rejected(self, args_model: type[BaseModel]) -> None:
        """Checking out two days after the last night is rejected."""
        with pytest.raises(ValidationError, match="2026-01-04"):
            _validate(
                args_model,
                check_in=LAST_NIGHT,
                check_out=LAST_NIGHT + 2 * _ONE_DAY,
            )

    def test_new_year_crossing_accepted(self, args_model: type[BaseModel]) -> None:
        """A stay from Dec 30 to Jan 2 spans two years and is valid."""
        args = _validate(args_model, check_in="2025-12-30", check_out="2026-01-02")
        assert getattr(args, "check_out") == dt.date(2026, 1, 2)  # noqa: B009

    def test_check_out_before_check_in_rejected(
        self, args_model: type[BaseModel],
    ) -> None:
        """A zero-night stay is rejected."""
        with pytest.raises(ValidationError, match="after check_in"):
            _validate(args_model, check_in="2025-03-05", check_out="2025-03-05")

    @pytest.mark.parametrize(
        "dates",
        [{"check_in": "2025-03-03"}, {"check_out": "2025-03-05"}],
    )
    def test_one_date_alone_rejected(
        self, args_model: type[BaseModel], dates: dict[str, str],
    ) -> None:
        """Dates come in pairs or not at all."""
        with pytest.raises(ValidationError, match="both"):
            _validate(args_model, **dates)

    def test_price_needs_dates(self, args_model: type[BaseModel]) -> None:
        """A nightly price cap means nothing without a stay."""
        with pytest.raises(ValidationError, match="max_nightly_price"):
            _validate(args_model, max_nightly_price=300)

    def test_price_with_dates_accepted(self, args_model: type[BaseModel]) -> None:
        """A nightly price cap with a stay is valid."""
        _validate(
            args_model,
            check_in="2025-03-03",
            check_out="2025-03-05",
            max_nightly_price=300,
        )


class TestVocabulary:
    """Categorical filters accept only values from the metadata."""

    def test_known_values_accepted(self, args_model: type[BaseModel]) -> None:
        """Values present in the metadata validate."""
        _validate(
            args_model,
            room_types=["Presidential Suite"],
            bed_types=["King"],
            amenities=["Balcony", "Wi-Fi"],
            view_types=["Ocean View"],
        )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"amenities": ["Jacuzzi"]},
            {"view_types": ["Mountain View"]},
            {"room_types": ["Penthouse"]},
            {"bed_types": ["Twin"]},
        ],
    )
    def test_unknown_value_rejected(
        self, args_model: type[BaseModel], kwargs: dict[str, list[str]],
    ) -> None:
        """A value the database does not hold is rejected before any SQL."""
        with pytest.raises(ValidationError):
            _validate(args_model, **kwargs)

    def test_schema_lists_allowed_values(self, args_model: type[BaseModel]) -> None:
        """The model-facing schema enumerates the vocabulary."""
        schema = str(args_model.model_json_schema())
        assert "Ocean View" in schema
        assert "Mini Bar" in schema

    def test_too_many_room_numbers_rejected(
        self, args_model: type[BaseModel],
    ) -> None:
        """A search may name only a bounded number of rooms."""
        with pytest.raises(ValidationError):
            _validate(args_model, room_numbers=list(range(MAX_ROOM_NUMBERS + 1)))


class TestFloors:
    """Floor filters are bounded by the hotel's floors and consistent."""

    def test_top_floor_accepted(self, args_model: type[BaseModel]) -> None:
        """'Top floor' is min_floor and max_floor both at the top."""
        _validate(args_model, min_floor=20, max_floor=20)

    @pytest.mark.parametrize("kwargs", [{"max_floor": 21}, {"min_floor": 0}])
    def test_out_of_range_floor_rejected(
        self, args_model: type[BaseModel], kwargs: dict[str, int],
    ) -> None:
        """A floor the hotel does not have is rejected."""
        with pytest.raises(ValidationError):
            _validate(args_model, **kwargs)

    def test_inverted_range_rejected(self, args_model: type[BaseModel]) -> None:
        """min_floor above max_floor is rejected."""
        with pytest.raises(ValidationError, match="min_floor"):
            _validate(args_model, min_floor=15, max_floor=10)

    def test_schema_names_the_top_floor(self, args_model: type[BaseModel]) -> None:
        """The max_floor description tells the model which floor is the top."""
        description = args_model.model_json_schema()["properties"]["max_floor"][
            "description"
        ]
        assert "top floor is 20" in description


class TestQuerySelection:
    """Each ordering selects a fixed query; only values vary."""

    @pytest.mark.parametrize("sort_by", get_args(SortBy))
    @pytest.mark.parametrize("dated", [True, False])
    def test_every_ordering_is_bounded(self, *, sort_by: SortBy, dated: bool) -> None:
        """Every sort_by has a limited query, priced only when dated."""
        query = select_search_query(sort_by=sort_by, dated=dated).as_string()
        assert query.rstrip().endswith("LIMIT %(limit)s")
        assert ("s.total_price" in query) is dated

    @pytest.mark.parametrize(
        ("sort_by", "dated", "order_by"),
        [
            ("price", True, "ORDER BY s.total_price, r.room_number"),
            ("price", False, "ORDER BY r.room_number"),
            ("floor_desc", False, "ORDER BY r.floor DESC, r.room_number"),
            (
                "square_feet_desc",
                False,
                "ORDER BY r.square_feet DESC NULLS LAST, r.room_number",
            ),
        ],
    )
    def test_ordering_clause(
        self, *, sort_by: SortBy, dated: bool, order_by: str,
    ) -> None:
        """Each sort_by maps to its fixed ORDER BY clause."""
        assert order_by in select_search_query(sort_by=sort_by, dated=dated).as_string()

    def test_no_internal_columns_selected(self) -> None:
        """Identifiers, room status, rates, and renovation dates never leave."""
        query = select_search_query(sort_by="price", dated=True).as_string()
        select_list = query[query.rindex("SELECT") : query.index("FROM rooms")]
        for column in ("room_id", "r.status", "base_rate", "max_rate", "renovation"):
            assert column not in select_list

    def test_empty_list_means_unrestricted(self, args_model: type[BaseModel]) -> None:
        """An empty filter list is passed as NULL, the same as omitting it."""
        params = _query_params(
            _validate(args_model, amenities=[]), max_results=MAX_RESULTS,
        )
        assert params["amenities"] is None


class TestLimit:
    """The model may ask for a number of rooms, clamped to the maximum."""

    def test_default_limit(self, args_model: type[BaseModel]) -> None:
        """Without a limit, a search asks for the default number of rooms."""
        params = _query_params(_validate(args_model), max_results=MAX_RESULTS)
        assert params["limit"] == DEFAULT_RESULTS

    def test_limit_within_maximum_is_kept(self, args_model: type[BaseModel]) -> None:
        """A limit up to the maximum is used as given."""
        params = _query_params(
            _validate(args_model, limit=MAX_RESULTS), max_results=MAX_RESULTS,
        )
        assert params["limit"] == MAX_RESULTS

    def test_limit_above_maximum_is_clamped(
        self, args_model: type[BaseModel],
    ) -> None:
        """A larger limit is accepted, then clamped rather than rejected."""
        params = _query_params(
            _validate(args_model, limit=MAX_RESULTS + 10), max_results=MAX_RESULTS,
        )
        assert params["limit"] == MAX_RESULTS

    def test_limit_below_one_is_rejected(self, args_model: type[BaseModel]) -> None:
        """A limit of zero is a mistake the model can fix."""
        with pytest.raises(ValidationError):
            _validate(args_model, limit=0)

    def test_description_states_default_and_maximum(
        self, args_model: type[BaseModel],
    ) -> None:
        """Any client's model learns the default and maximum from the schema."""
        description = args_model.model_json_schema()["properties"]["limit"][
            "description"
        ]
        assert f"{DEFAULT_RESULTS} by default" in description
        assert f"At most {MAX_RESULTS}" in description

    def test_limit_note_tells_the_user(self) -> None:
        """The clamp note carries its own instruction, since MCP skips our prompt."""
        note = _limit_note(MAX_RESULTS + 5, MAX_RESULTS)
        assert str(MAX_RESULTS + 5) in note
        assert str(MAX_RESULTS) in note
        assert "Tell the user" in note
