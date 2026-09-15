"""Tests for eval/evaluators/_latency.py.

eval_turn_latency() once treated each parsed ``TurnOutput`` as a dict, so its
``"latency_ms" in t`` filter was always false and every eval summary silently
lost its latency quantiles. Covers it via minimal mock Run objects.
"""

# ruff: noqa: S101

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

from eval.evaluators._latency import eval_turn_latency

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


def _records(turn_outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run the evaluator and decode its ``latency_per_turn`` records.

    Args:
        turn_outputs: List of per-turn output dicts.

    Returns:
        The decoded list of ``{route, latency_ms}`` records.

    """
    results = eval_turn_latency(
        _make_run(turn_outputs),
        cast("Example", SimpleNamespace(inputs={})),
    )
    assert [r["key"] for r in results] == ["latency_per_turn"]
    return json.loads(results[0]["value"])


class TestEvalTurnLatency:
    """eval_turn_latency() collects per-turn latency records."""

    def test_records_every_timed_turn(self) -> None:
        """Each turn carrying ``latency_ms`` yields one route-tagged record."""
        records = _records(
            [
                {"route_pred": "info", "latency_ms": 812.5},
                {"route_pred": "booking", "latency_ms": 2400.0},
            ],
        )
        assert records == [
            {"route": "info", "latency_ms": 812.5},
            {"route": "booking", "latency_ms": 2400.0},
        ]

    def test_skips_turns_without_latency(self) -> None:
        """A turn with no ``latency_ms`` is omitted rather than recorded as null."""
        records = _records([{"route_pred": "info"}])
        assert records == []
