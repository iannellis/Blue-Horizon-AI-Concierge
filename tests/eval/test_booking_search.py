"""Tests for eval/evaluators/_booking_search.py.

Covers the per-key matching rules of `_mismatches` (absent keys, null,
set-equal lists, dates, numbers, and ``limit`` as a minimum) and runs
`eval_booking_expected_search` end-to-end on minimal mock Run/Example
objects: alternatives, several calls in one turn, turns with no search,
unlabeled cases, and the choice of the closest call in a failure record.
"""

# ruff: noqa: S101
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest

from eval.evaluators._booking_search import _mismatches, eval_booking_expected_search

if TYPE_CHECKING:
    from langsmith.schemas import Example, Run

    from eval.config import EvalConfig

_SPEC: dict[str, Any] = {
    "room_types": ["Presidential Suite"],
    "min_floor": 20,
    "check_in": "2025-03-03",
    "check_out": "2025-03-05",
}


def _make_run(turn_outputs: list[dict[str, Any]]) -> Run:
    """Build a minimal run-like object for testing.

    Args:
        turn_outputs: List of per-turn output dicts.

    Returns:
        Run: A `SimpleNamespace` with an `outputs` attribute, cast to `Run`
        since the evaluator only reads that attribute.

    """
    return cast("Run", SimpleNamespace(outputs={"turn_outputs": turn_outputs}))


def _make_example(turns: list[dict[str, Any]]) -> Example:
    """Build a minimal example-like object for testing.

    Args:
        turns: List of turn dicts, each optionally containing expected_search.

    Returns:
        Example: A `SimpleNamespace` with an `inputs` attribute, cast to
        `Example` since the evaluator only reads that attribute.

    """
    return cast("Example", SimpleNamespace(inputs={"turns": turns}))


def _make_cfg(*, search_filter_failures_max: int = 50) -> EvalConfig:
    """Build a minimal cfg-like object for testing.

    Args:
        search_filter_failures_max: Maximum failure entries to keep.

    Returns:
        EvalConfig: A `SimpleNamespace` with an `evaluator_limits` attribute,
        cast to `EvalConfig` since the evaluator only reads that attribute.

    """
    limits = SimpleNamespace(
        search_filter_failures_max=search_filter_failures_max,
        json_value_max=10_000,
    )
    return cast("EvalConfig", SimpleNamespace(evaluator_limits=limits))


def _search_turn(*calls: dict[str, Any]) -> dict[str, Any]:
    """Build one turn output containing the given search_rooms calls.

    Args:
        *calls: The model-supplied arguments of each search_rooms call.

    Returns:
        A turn output dict whose tool_summary holds one entry per call.

    """
    return {
        "tool_summary": [
            {"tool": "search_rooms", "status": "ok", "search_args": args}
            for args in calls
        ],
    }


def _run_eval(
    turn_outputs: list[dict[str, Any]],
    turns: list[dict[str, Any]],
    *,
    cfg: EvalConfig | None = None,
) -> dict[str, dict[str, Any]]:
    """Run the evaluator and index its feedback by key.

    Args:
        turn_outputs: Per-turn run outputs.
        turns: Dataset turns.
        cfg: Evaluator config; a default one when omitted.

    Returns:
        Mapping from feedback key to feedback dict.

    """
    results = eval_booking_expected_search(
        _make_run(turn_outputs),
        _make_example(turns),
        cfg=cfg or _make_cfg(),
    )
    return {r["key"]: r for r in results}


# ---------------------------------------------------------------------------
# _mismatches
# ---------------------------------------------------------------------------


class TestMismatches:
    """_mismatches applies the per-key matching rules."""

    def test_exact_match(self) -> None:
        """A call with exactly the spec's values matches."""
        assert _mismatches(_SPEC, dict(_SPEC)) == []

    def test_absent_key_is_unconstrained(self) -> None:
        """Arguments the spec does not mention do not matter."""
        args = {**_SPEC, "sort_by": "price", "accessible": True}
        assert _mismatches(_SPEC, args) == []

    def test_null_requires_unset(self) -> None:
        """A null spec value fails when the model set the argument."""
        spec = {"check_in": None}
        assert _mismatches(spec, {}) == []
        assert _mismatches(spec, {"check_in": "2025-03-03"}) == ["check_in"]

    def test_lists_compare_as_sets(self) -> None:
        """List order is ignored, but the members must be the same."""
        spec = {"view_types": ["Ocean View", "Panoramic Ocean View"]}
        reordered = {"view_types": ["Panoramic Ocean View", "Ocean View"]}
        assert _mismatches(spec, reordered) == []
        assert _mismatches(spec, {"view_types": ["Ocean View"]}) == ["view_types"]
        assert _mismatches(spec, {"view_types": "Ocean View"}) == ["view_types"]

    def test_dates_compare_as_strings(self) -> None:
        """A date in the wrong year fails."""
        args = {**_SPEC, "check_out": "2024-03-05"}
        assert _mismatches(_SPEC, args) == ["check_out"]

    def test_numbers_compare_numerically(self) -> None:
        """An int spec matches the same float value, not a different one."""
        assert _mismatches({"min_floor": 20}, {"min_floor": 20.0}) == []
        assert _mismatches({"min_floor": 20}, {"min_floor": 19}) == ["min_floor"]
        assert _mismatches({"min_floor": 20}, {"min_floor": True}) == ["min_floor"]

    @pytest.mark.parametrize(
        ("actual", "expected_mismatches"),
        [(15, []), (50, []), (10, ["limit"]), (None, ["limit"])],
    )
    def test_limit_is_a_minimum(
        self,
        actual: int | None,
        expected_mismatches: list[str],
    ) -> None:
        """The model's limit passes at or above the expected value.

        Args:
            actual: Limit the model sent, or None when omitted.
            expected_mismatches: Expected result of `_mismatches`.

        """
        args = {} if actual is None else {"limit": actual}
        assert _mismatches({"limit": 15}, args) == expected_mismatches

    def test_reports_every_failing_key(self) -> None:
        """All failing keys are returned, sorted."""
        args = {"room_types": ["Suite"], "min_floor": 20}
        assert _mismatches(_SPEC, args) == ["check_in", "check_out", "room_types"]


# ---------------------------------------------------------------------------
# eval_booking_expected_search (end-to-end with mocks)
# ---------------------------------------------------------------------------


class TestEvalBookingExpectedSearch:
    """eval_booking_expected_search returns correct LangSmith feedback dicts."""

    def test_unlabeled_case_is_skipped(self) -> None:
        """A case without expected_search emits only the skipped key."""
        results = _run_eval([_search_turn({})], [{"user": "hi"}])
        assert list(results) == ["booking_expected_search_skipped"]
        assert results["booking_expected_search_skipped"]["score"] == 1.0

    def test_matching_turn_passes(self) -> None:
        """A turn whose only call matches the spec scores 1.0."""
        results = _run_eval(
            [_search_turn(dict(_SPEC))],
            [{"user": "suite", "expected_search": _SPEC}],
        )
        assert results["booking_expected_search_turns"]["score"] == 1.0
        assert results["booking_expected_search_pass_rate"]["score"] == 1.0
        failures = json.loads(results["booking_expected_search_failures"]["value"])
        assert failures == []

    def test_any_alternative_may_match(self) -> None:
        """A list of specs passes when any one of them matches."""
        alternatives = [
            {"view_types": ["Ocean View"]},
            {"view_types": ["Ocean View", "Panoramic Ocean View"]},
        ]
        results = _run_eval(
            [_search_turn({"view_types": ["Panoramic Ocean View", "Ocean View"]})],
            [{"user": "ocean", "expected_search": alternatives}],
        )
        assert results["booking_expected_search_pass_rate"]["score"] == 1.0

    def test_any_call_may_match(self) -> None:
        """A turn passes when a later call matches after an earlier miss."""
        results = _run_eval(
            [_search_turn({"room_types": ["Suite"]}, dict(_SPEC))],
            [{"user": "suite", "expected_search": _SPEC}],
        )
        assert results["booking_expected_search_pass_rate"]["score"] == 1.0

    def test_turn_without_search_fails(self) -> None:
        """A labeled turn with no search_rooms call fails."""
        results = _run_eval(
            [{"tool_summary": []}],
            [{"user": "suite", "expected_search": _SPEC}],
        )
        assert results["booking_expected_search_pass_rate"]["score"] == 0.0
        failures = json.loads(results["booking_expected_search_failures"]["value"])
        assert failures[0]["failed_check"] == "search_called"
        assert failures[0]["calls"] == []

    def test_failure_reports_closest_call(self) -> None:
        """The failure record names the mismatches of the closest call."""
        far = {"room_types": ["Suite"]}
        near = {**_SPEC, "check_out": "2024-03-05"}
        results = _run_eval(
            [_search_turn(far, near)],
            [{"user": "suite", "expected_search": _SPEC}],
        )
        failures = json.loads(results["booking_expected_search_failures"]["value"])
        assert failures[0]["failed_check"] == "search_args_match"
        assert failures[0]["closest_mismatches"] == ["check_out"]
        assert failures[0]["calls"] == [far, near]

    def test_pass_rate_and_per_turn_over_labeled_turns(self) -> None:
        """Only labeled turns count, and each gets a per-turn record."""
        results = _run_eval(
            [_search_turn(dict(_SPEC)), _search_turn({}), _search_turn({})],
            [
                {"user": "a", "expected_search": _SPEC},
                {"user": "b"},
                {"user": "c", "expected_search": _SPEC},
            ],
        )
        labeled_turns = 2.0
        pass_rate = 0.5
        assert results["booking_expected_search_turns"]["score"] == labeled_turns
        assert results["booking_expected_search_pass_rate"]["score"] == pass_rate
        per_turn = json.loads(results["booking_expected_search_per_turn"]["value"])
        assert per_turn == [
            {"turn_index": 0, "pass_rate": 1.0},
            {"turn_index": 2, "pass_rate": 0.0},
        ]

    def test_failures_are_capped(self) -> None:
        """search_filter_failures_max limits how many failures are recorded."""
        turns = [{"user": str(i), "expected_search": _SPEC} for i in range(3)]
        results = _run_eval(
            [_search_turn({}) for _ in turns],
            turns,
            cfg=_make_cfg(search_filter_failures_max=1),
        )
        failures = json.loads(results["booking_expected_search_failures"]["value"])
        assert len(failures) == 1
