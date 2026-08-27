"""Tests for eval/evaluators/_info_filters.py.

This evaluator had zero direct test coverage before this file (only its
metric key *names* appeared in `tests/eval/test_result_utils.py`'s
aggregation fixtures). Covers the pure scoring helpers directly, and
eval_info_expected_filters() end-to-end via minimal mock Run/Example
objects (no LangSmith, no network).
"""

# ruff: noqa: S101

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

from eval.evaluators._info_filters import (
    _build_info_expected_filters_failure,
    _canon_expected_filters,
    _extract_filters_for_tool,
    _FilterObservation,
    _fold_checks,
    _has_filters,
    _score_expected_filters,
    eval_info_expected_filters,
)
from eval.models import ExampleTurn, ToolSummaryEntry, TurnOutput

if TYPE_CHECKING:
    from langsmith.schemas import Example, Run

    from eval.config import EvalConfig

_MAX_PRICE = 150


def _make_run(turn_outputs: list[dict[str, Any]]) -> Run:
    """Build a minimal run-like object for testing.

    Args:
        turn_outputs: List of per-turn output dicts.

    Returns:
        Run: A `SimpleNamespace` with an `outputs` attribute, cast to `Run`
        since `eval_info_expected_filters` only ever reads that attribute.

    """
    return cast("Run", SimpleNamespace(outputs={"turn_outputs": turn_outputs}))


def _make_example(turns: list[dict[str, Any]]) -> Example:
    """Build a minimal example-like object for testing.

    Args:
        turns: List of turn dicts, each optionally containing expected_filters.

    Returns:
        Example: A `SimpleNamespace` with an `inputs` attribute, cast to
        `Example` since `eval_info_expected_filters` only ever reads that
        attribute.

    """
    return cast("Example", SimpleNamespace(inputs={"turns": turns}))


def _make_cfg(
    *,
    info_filter_failures_max: int = 50,
    json_value_max: int = 10_000,
) -> EvalConfig:
    """Build a minimal cfg-like object for testing.

    Args:
        info_filter_failures_max: Maximum failure entries to keep.
        json_value_max: Maximum JSON string length.

    Returns:
        EvalConfig: A `SimpleNamespace` with an `evaluator_limits` attribute,
        cast to `EvalConfig` since `eval_info_expected_filters` only ever
        reads that attribute.

    """
    limits = SimpleNamespace(
        info_filter_failures_max=info_filter_failures_max,
        json_value_max=json_value_max,
    )
    return cast("EvalConfig", SimpleNamespace(evaluator_limits=limits))


def _amenities_entry(filters_norm: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a query_amenities tool_summary entry dict with normalized filters.

    Args:
        filters_norm: Normalized filters the entry carries, if any.

    Returns:
        A tool_summary entry dict for query_amenities.

    """
    entry: dict[str, Any] = {"tool": "query_amenities", "status": "ok"}
    if filters_norm is not None:
        entry["filters_norm"] = filters_norm
    return entry


# ---------------------------------------------------------------------------
# _has_filters
# ---------------------------------------------------------------------------


class TestHasFilters:
    """_has_filters distinguishes a non-empty filters dict from None/empty."""

    def test_none_is_not_filters(self) -> None:
        """None is not filters."""
        assert _has_filters(None) is False

    def test_empty_dict_is_not_filters(self) -> None:
        """An empty dict is not filters."""
        assert _has_filters({}) is False

    def test_non_empty_dict_is_filters(self) -> None:
        """A non-empty dict is filters."""
        assert _has_filters({"max_price": 100}) is True


# ---------------------------------------------------------------------------
# _canon_expected_filters
# ---------------------------------------------------------------------------


class TestCanonExpectedFilters:
    """_canon_expected_filters drops Nones and coerces canonical keys."""

    def test_drops_null_values(self) -> None:
        """Keys present with a null value are dropped."""
        result = _canon_expected_filters({"min_price": None, "max_price": 150})
        assert "min_price" not in result
        assert result["max_price"] == _MAX_PRICE

    def test_coerces_booking_required_to_bool(self) -> None:
        """booking_required coerces to bool."""
        result = _canon_expected_filters({"booking_required": 1})
        assert result["booking_required"] is True

    def test_coerces_price_fields_to_float(self) -> None:
        """min_price/max_price coerce to float."""
        result = _canon_expected_filters({"max_price": "150"})
        assert result["max_price"] == _MAX_PRICE
        assert isinstance(result["max_price"], float)

    def test_coerces_other_fields_to_int(self) -> None:
        """Non-price, non-bool canonical fields coerce to int."""
        result = _canon_expected_filters({"max_notice_hours": "4"})
        assert result["max_notice_hours"] == 4  # noqa: PLR2004

    def test_ignores_non_canonical_keys(self) -> None:
        """Keys outside the canonical set are dropped entirely."""
        result = _canon_expected_filters({"not_a_real_filter": "x"})
        assert result == {}

    def test_empty_input_returns_empty(self) -> None:
        """An empty dict returns an empty dict."""
        assert _canon_expected_filters({}) == {}


# ---------------------------------------------------------------------------
# _extract_filters_for_tool
# ---------------------------------------------------------------------------


class TestExtractFiltersForTool:
    """_extract_filters_for_tool finds the last matching entry."""

    def test_no_matching_tool_returns_none(self) -> None:
        """A tool_summary with no matching tool name returns (None, [])."""
        entries = [ToolSummaryEntry(tool="query_faq")]
        filters, unknown = _extract_filters_for_tool(entries, "query_amenities")
        assert filters is None
        assert unknown == []

    def test_matching_entry_returns_its_filters(self) -> None:
        """The matching entry's filters_norm and filters_unknown_keys are returned."""
        entries = [
            ToolSummaryEntry(
                tool="query_amenities",
                filters_norm={"max_price": 100},
                filters_unknown_keys=["bogus"],
            ),
        ]
        filters, unknown = _extract_filters_for_tool(entries, "query_amenities")
        assert filters == {"max_price": 100}
        assert unknown == ["bogus"]

    def test_last_matching_entry_wins(self) -> None:
        """When a tool is called twice in one turn, the last call wins."""
        entries = [
            ToolSummaryEntry(tool="query_amenities", filters_norm={"max_price": 50}),
            ToolSummaryEntry(tool="query_amenities", filters_norm={"max_price": 100}),
        ]
        filters, _unknown = _extract_filters_for_tool(entries, "query_amenities")
        assert filters == {"max_price": 100}

    def test_empty_tool_summary_returns_none(self) -> None:
        """An empty tool_summary returns (None, [])."""
        filters, unknown = _extract_filters_for_tool([], "query_amenities")
        assert filters is None
        assert unknown == []


# ---------------------------------------------------------------------------
# _fold_checks / _score_expected_filters
# ---------------------------------------------------------------------------


class TestFoldChecks:
    """_fold_checks tallies named pass/fail checks into a _FilterScore."""

    def test_all_pass(self) -> None:
        """All-passing checks produce zero failures."""
        score = _fold_checks([("a", True), ("b", True)])
        assert score.total_checks == 2  # noqa: PLR2004
        assert score.passed_checks == 2  # noqa: PLR2004
        assert score.failed_checks == []

    def test_some_fail_preserves_order(self) -> None:
        """Failed check labels are recorded in check order."""
        score = _fold_checks([("a", True), ("b", False), ("c", False)])
        assert score.passed_checks == 1
        assert score.failed_checks == ["b", "c"]

    def test_divergence_fields_pass_through(self) -> None:
        """divergence_checks/divergence_failures are carried through unchanged."""
        score = _fold_checks([("a", True)], divergence_checks=1, divergence_failures=1)
        assert score.divergence_checks == 1
        assert score.divergence_failures == 1


class TestScoreExpectedFilters:
    """_score_expected_filters dispatches on whether filters are expected."""

    def test_expected_present_and_matched(self) -> None:
        """Both tools called with matching filters -> everything passes."""
        expected = {"max_price": 100}
        obs = _FilterObservation(
            expected=expected,
            amenity_filters=expected,
            service_filters=expected,
            amenity_unknown=[],
            service_unknown=[],
        )
        score = _score_expected_filters(obs)
        assert score.failed_checks == []
        assert score.divergence_checks == 1
        assert score.divergence_failures == 0

    def test_expected_present_but_amenity_missing(self) -> None:
        """query_amenities never called -> amenities_filters_present fails."""
        expected = {"max_price": 100}
        obs = _FilterObservation(
            expected=expected,
            amenity_filters=None,
            service_filters=expected,
            amenity_unknown=[],
            service_unknown=[],
        )
        score = _score_expected_filters(obs)
        assert "amenities_filters_present" in score.failed_checks
        assert "amenities_filters_match_expected" in score.failed_checks
        # Only one tool called this turn -> no divergence check applies.
        assert score.divergence_checks == 0

    def test_filters_diverge_between_tools(self) -> None:
        """Both tools called with different filters -> divergence failure."""
        expected = {"max_price": 100}
        obs = _FilterObservation(
            expected=expected,
            amenity_filters={"max_price": 100},
            service_filters={"max_price": 200},
            amenity_unknown=[],
            service_unknown=[],
        )
        score = _score_expected_filters(obs)
        assert "filters_diverge_between_tools" in score.failed_checks
        assert score.divergence_checks == 1
        assert score.divergence_failures == 1

    def test_unknown_filter_keys_fail_the_check(self) -> None:
        """An unknown key on either tool fails unknown_filter_keys_used."""
        expected = {"max_price": 100}
        obs = _FilterObservation(
            expected=expected,
            amenity_filters=expected,
            service_filters=expected,
            amenity_unknown=["bogus_key"],
            service_unknown=[],
        )
        score = _score_expected_filters(obs)
        assert "unknown_filter_keys_used" in score.failed_checks

    def test_expected_empty_and_no_filters_called(self) -> None:
        """No filters expected, none passed -> everything passes."""
        obs = _FilterObservation(
            expected={},
            amenity_filters=None,
            service_filters=None,
            amenity_unknown=[],
            service_unknown=[],
        )
        score = _score_expected_filters(obs)
        assert score.failed_checks == []

    def test_expected_empty_but_filters_called_anyway(self) -> None:
        """No filters expected, but a tool was called with filters -> fails."""
        obs = _FilterObservation(
            expected={},
            amenity_filters={"max_price": 100},
            service_filters=None,
            amenity_unknown=[],
            service_unknown=[],
        )
        score = _score_expected_filters(obs)
        assert "amenities_no_filters" in score.failed_checks


# ---------------------------------------------------------------------------
# _build_info_expected_filters_failure
# ---------------------------------------------------------------------------


class TestBuildFailure:
    """_build_info_expected_filters_failure assembles the failure record."""

    def test_fields_populated_from_observation(self) -> None:
        """Every observation field lands in the failure dict."""
        obs = _FilterObservation(
            expected={"max_price": 100},
            amenity_filters=None,
            service_filters=None,
            amenity_unknown=[],
            service_unknown=[],
        )
        failure = _build_info_expected_filters_failure(
            idx=2,
            example_turn=ExampleTurn(user="spa please"),
            turn_output=TurnOutput(route_pred="info"),
            observation=obs,
            failed_checks=["amenities_filters_present"],
        )
        assert failure["turn_index"] == 2  # noqa: PLR2004
        assert failure["user_snippet"] == "spa please"
        assert failure["route_pred"] == "info"
        assert failure["expected"] == {"max_price": 100}
        assert failure["failed_checks"] == ["amenities_filters_present"]


# ---------------------------------------------------------------------------
# eval_info_expected_filters (end-to-end with mocks)
# ---------------------------------------------------------------------------


class TestEvalInfoExpectedFilters:
    """eval_info_expected_filters() returns correct LangSmith feedback dicts."""

    def test_no_labeled_turns_returns_skipped(self) -> None:
        """No turn has expected_filters -> the skipped result."""
        run = _make_run([{"route_pred": "info"}])
        example = _make_example([{"user": "what time is checkout?"}])
        results = eval_info_expected_filters(run, example, cfg=_make_cfg())
        keys = [r["key"] for r in results]
        assert "info_expected_filters_skipped" in keys

    def test_matching_filters_pass_rate_one(self) -> None:
        """Filters called correctly on both tools -> pass_rate 1.0."""
        expected = {"max_price": 100}
        run = _make_run(
            [
                {
                    "route_pred": "info",
                    "tool_summary": [
                        _amenities_entry(expected),
                        {"tool": "query_services", "filters_norm": expected},
                    ],
                },
            ],
        )
        example = _make_example(
            [{"user": "spa under $100", "expected_filters": expected}],
        )
        results = eval_info_expected_filters(run, example, cfg=_make_cfg())
        scores = {r["key"]: r.get("score") for r in results}
        assert scores["info_expected_filters_pass_rate"] == 1.0

    def test_missing_filters_lowers_pass_rate_and_records_failure(self) -> None:
        """Filters expected but never sent -> pass_rate < 1.0, failure recorded."""
        expected = {"max_price": 100}
        run = _make_run([{"route_pred": "info", "tool_summary": []}])
        example = _make_example(
            [{"user": "spa under $100", "expected_filters": expected}],
        )
        results = eval_info_expected_filters(run, example, cfg=_make_cfg())
        scores = {r["key"]: r.get("score") for r in results}
        pass_rate = scores["info_expected_filters_pass_rate"]
        assert pass_rate is not None
        assert pass_rate < 1.0
        failures_entry = next(
            r for r in results if r["key"] == "info_expected_filters_failures"
        )
        assert failures_entry["value"] != "[]"

    def test_no_filters_expected_and_none_sent_passes(self) -> None:
        """A turn labeled with empty expected_filters and no filters sent passes."""
        run = _make_run(
            [{"route_pred": "info", "tool_summary": [{"tool": "query_faq"}]}],
        )
        example = _make_example([{"user": "check-in time?", "expected_filters": {}}])
        results = eval_info_expected_filters(run, example, cfg=_make_cfg())
        scores = {r["key"]: r.get("score") for r in results}
        assert scores["info_expected_filters_pass_rate"] == 1.0

    def test_labeled_turns_count(self) -> None:
        """info_expected_filters_turns reflects the number of labeled turns."""
        expected = {"max_price": 100}
        run = _make_run(
            [
                {"route_pred": "info", "tool_summary": [_amenities_entry(expected)]},
                {"route_pred": "info", "tool_summary": []},
            ],
        )
        example = _make_example(
            [
                {"user": "spa?", "expected_filters": expected},
                {"user": "check-in time?"},
            ],
        )
        results = eval_info_expected_filters(run, example, cfg=_make_cfg())
        scores = {r["key"]: r.get("score") for r in results}
        assert scores["info_expected_filters_turns"] == 1.0

    def test_failures_max_caps_recorded_failures(self) -> None:
        """info_filter_failures_max limits how many failures are recorded."""
        expected = {"max_price": 100}
        run = _make_run(
            [{"route_pred": "info", "tool_summary": []} for _ in range(3)],
        )
        example = _make_example(
            [{"user": "spa?", "expected_filters": expected} for _ in range(3)],
        )
        cfg = _make_cfg(info_filter_failures_max=1)
        results = eval_info_expected_filters(run, example, cfg=cfg)
        failures_entry = next(
            r for r in results if r["key"] == "info_expected_filters_failures"
        )
        assert len(json.loads(failures_entry["value"])) == 1
