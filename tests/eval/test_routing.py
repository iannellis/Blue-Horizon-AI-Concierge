"""Tests for eval/evaluators/_routing.py.

eval_routing_accuracy() had no prior test coverage. Covers it end-to-end via
minimal mock Run/Example objects (no LangSmith, no network).
"""

# ruff: noqa: S101

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

from eval.evaluators._routing import eval_routing_accuracy

if TYPE_CHECKING:
    from langsmith.schemas import Example, Run


def _make_run(turn_outputs: list[dict[str, Any]]) -> Run:
    """Build a minimal run-like object for testing.

    Args:
        turn_outputs: List of per-turn output dicts.

    Returns:
        Run: A `SimpleNamespace` with an `outputs` attribute, cast to `Run`.

    """
    return cast("Run", SimpleNamespace(outputs={"turn_outputs": turn_outputs}))


def _make_example(turns: list[dict[str, Any]]) -> Example:
    """Build a minimal example-like object for testing.

    Args:
        turns: List of turn dicts, each optionally containing expected_route.

    Returns:
        Example: A `SimpleNamespace` with an `inputs` attribute, cast to `Example`.

    """
    return cast("Example", SimpleNamespace(inputs={"turns": turns}))


class TestEvalRoutingAccuracy:
    """eval_routing_accuracy() scores per-turn route predictions."""

    def test_no_turns_returns_zero_score(self) -> None:
        """An empty run/example produces a 0.0 accuracy, not a crash."""
        results = eval_routing_accuracy(_make_run([]), _make_example([]))
        scores = {r["key"]: r.get("score") for r in results}
        assert scores["route_accuracy"] == 0.0

    def test_all_correct_scores_one(self) -> None:
        """Every prediction matching its expected route scores 1.0."""
        run = _make_run(
            [{"route_pred": "info"}, {"route_pred": "booking"}],
        )
        example = _make_example(
            [{"expected_route": "info"}, {"expected_route": "booking"}],
        )
        results = eval_routing_accuracy(run, example)
        scores = {r["key"]: r.get("score") for r in results}
        assert scores["route_accuracy"] == 1.0

    def test_mismatch_lowers_accuracy(self) -> None:
        """One mismatched turn out of two halves the accuracy."""
        run = _make_run([{"route_pred": "info"}, {"route_pred": "info"}])
        example = _make_example(
            [{"expected_route": "info"}, {"expected_route": "booking"}],
        )
        results = eval_routing_accuracy(run, example)
        scores = {r["key"]: r.get("score") for r in results}
        assert scores["route_accuracy"] == 0.5  # noqa: PLR2004

    def test_missing_output_counts_as_missing_not_a_match(self) -> None:
        """A turn with no run output at all is scored as a miss, not skipped."""
        run = _make_run([{"route_pred": "info"}])
        example = _make_example(
            [{"expected_route": "info"}, {"expected_route": "booking"}],
        )
        results = eval_routing_accuracy(run, example)
        scores = {r["key"]: r.get("score") for r in results}
        assert scores["route_turns"] == 2.0  # noqa: PLR2004
        assert scores["route_accuracy"] == 0.5  # noqa: PLR2004

    def test_confusions_recorded(self) -> None:
        """route_confusions records the expected->predicted pairing."""
        run = _make_run([{"route_pred": "booking"}])
        example = _make_example([{"expected_route": "info"}])
        results = eval_routing_accuracy(run, example)
        confusions_entry = next(r for r in results if r["key"] == "route_confusions")
        assert "info->booking" in confusions_entry["value"]
