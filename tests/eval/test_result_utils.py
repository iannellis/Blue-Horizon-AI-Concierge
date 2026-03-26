"""Tests for eval/_result_utils.py normalization and summary helpers."""

# ruff: noqa: S101

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from eval._result_utils import (
    _normalize_feedback,
    _summarize_results,
    compute_latency_summary,
)


def _feedback_item(
    key: str,
    *,
    score: object | None = None,
    value: object | None = None,
    comment: str | None = None,
) -> list[tuple[str, object]]:
    """Build a persisted LangSmith feedback item as a list of pairs.

    Args:
        key: Metric key name.
        score: Optional score value.
        value: Optional value payload.
        comment: Optional comment.

    Returns:
        Feedback item encoded as the list-of-pairs structure stored in
        ``results.jsonl``.

    """
    item: list[tuple[str, object]] = [("key", key)]
    item.append(("score", score))
    item.append(("value", value))
    item.append(("comment", comment))
    return item


def _persisted_evaluators(
    *feedback_items: list[tuple[str, object]],
) -> dict[str, object]:
    """Build a persisted ``evaluators`` payload for a result row.

    Args:
        *feedback_items: Persisted feedback items to include.

    Returns:
        Evaluator payload matching the on-disk ``results.jsonl`` shape.

    """
    return {"results": {"value": list(feedback_items)}}


class TestNormalizeFeedback:
    """_normalize_feedback() flattens persisted evaluator payloads."""

    def test_nested_results_payload_becomes_flat_metric_map(self) -> None:
        """Nested persisted feedback is normalized into metric-keyed entries."""
        feedback = _persisted_evaluators(
            _feedback_item("route_accuracy", score=1.0, comment="all good"),
            _feedback_item(
                "latency_per_turn",
                value='[{"route": "rooms", "latency_ms": 12.5}]',
            ),
        )

        normalized = _normalize_feedback(feedback)

        assert normalized["route_accuracy"]["score"] == 1.0
        assert normalized["route_accuracy"]["comment"] == "all good"
        assert normalized["latency_per_turn"]["value"] == (
            '[{"route": "rooms", "latency_ms": 12.5}]'
        )


class TestSummarizeResults:
    """_summarize_results() aggregates case-level metric means."""

    def test_case_level_scores_are_aggregated_into_mean_metrics(self) -> None:
        """Per-case scores become ``mean_*`` summary fields."""
        expected_route_accuracy_mean = 0.75
        expected_consumer_quality_mean = 3.0
        expected_rag_mean = 0.75
        expected_filter_pass_rate_mean = 0.5
        rows = [
            {
                "evaluators": _persisted_evaluators(
                    _feedback_item("route_accuracy", score=1.0),
                    _feedback_item("judge_consumer_quality", score=4.0),
                    _feedback_item("rag_faithfulness_mean", score=0.50),
                    _feedback_item("info_expected_filters_pass_rate", score=0.25),
                ),
                "error": None,
            },
            {
                "evaluators": _persisted_evaluators(
                    _feedback_item("route_accuracy", score=0.5),
                    _feedback_item("judge_consumer_quality", score=2.0),
                    _feedback_item("rag_faithfulness_mean", score=1.0),
                    _feedback_item("info_expected_filters_pass_rate", score=0.75),
                ),
                "error": None,
            },
        ]
        context = SimpleNamespace(
            experiment_name="demo",
            dataset_name="dataset",
            max_concurrency=4,
            started_at=datetime(2026, 3, 1, tzinfo=UTC),
            finished_at=datetime(2026, 3, 1, 0, 5, tzinfo=UTC),
            upload_results=False,
        )

        summary = _summarize_results(rows, context)

        assert summary["mean_route_accuracy"] == expected_route_accuracy_mean
        assert summary["mean_consumer_quality"] == expected_consumer_quality_mean
        assert summary["mean_rag_faithfulness_mean"] == expected_rag_mean
        assert (
            summary["mean_info_expected_filters_pass_rate"]
            == expected_filter_pass_rate_mean
        )


class TestComputeLatencySummary:
    """compute_latency_summary() reads normalized persisted feedback."""

    def test_latency_summary_parses_nested_feedback_items(
        self,
        tmp_path: Path,
    ) -> None:
        """Latency quantiles are computed from persisted ``latency_per_turn`` data."""
        expected_rooms_p50 = 15.0
        expected_info_p95 = 5.0
        results_path = tmp_path / "results.jsonl"
        row = {
            "evaluators": _persisted_evaluators(
                _feedback_item(
                    "latency_per_turn",
                    value=json.dumps(
                        [
                            {"route": "rooms", "latency_ms": 10.0},
                            {"route": "rooms", "latency_ms": 20.0},
                            {"route": "info", "latency_ms": 5.0},
                        ],
                    ),
                ),
            ),
        }
        results_path.write_text(f"{json.dumps(row)}\n", encoding="utf-8")

        summary = compute_latency_summary(results_path)

        assert summary["latency_quantiles_ms"]["rooms"]["p50_ms"] == expected_rooms_p50
        assert summary["latency_quantiles_ms"]["info"]["p95_ms"] == expected_info_p95
