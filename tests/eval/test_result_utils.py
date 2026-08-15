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


def _as_dict(value: object) -> dict[str, object]:
    """Narrow a summary value expected to be a nested dict.

    `_summarize_results()` and `compute_latency_summary()` return
    structurally heterogeneous ``dict[str, object]`` summaries, so their
    nested values are only known to be dicts by the tests' own knowledge of
    what they put in, not by the return type.

    Args:
        value: Value pulled from a summary dict.

    Returns:
        dict[str, object]: The same value, narrowed for indexing.

    Raises:
        TypeError: If `value` is not actually a dict.

    """
    if not isinstance(value, dict):
        msg = f"Expected a dict, got {type(value).__name__}"
        raise TypeError(msg)
    return value


class TestNormalizeFeedback:
    """_normalize_feedback() flattens persisted evaluator payloads."""

    def test_nested_results_payload_becomes_flat_metric_map(self) -> None:
        """Nested persisted feedback is normalized into metric-keyed entries."""
        feedback = _persisted_evaluators(
            _feedback_item("route_accuracy", score=1.0, comment="all good"),
            _feedback_item(
                "latency_per_turn",
                value='[{"route": "booking", "latency_ms": 12.5}]',
            ),
        )

        normalized = _normalize_feedback(feedback)

        assert normalized["route_accuracy"]["score"] == 1.0
        assert normalized["route_accuracy"]["comment"] == "all good"
        assert normalized["latency_per_turn"]["value"] == (
            '[{"route": "booking", "latency_ms": 12.5}]'
        )


class TestSummarizeResults:
    """_summarize_results() groups case-based and turn-based summaries."""

    def test_summary_groups_case_and_turn_based_metrics(self) -> None:
        """Case-based and turn-based aggregates are reported separately."""
        expected_case_route_accuracy = 0.75
        expected_case_consumer_quality = 3.0
        expected_case_rag_faithfulness = 0.75
        expected_case_filter_pass_rate = 0.5
        expected_turn_route_accuracy = (1.0 * 2.0 + 0.5 * 4.0) / 6.0
        expected_turn_reference_pass_rate = (0.5 * 2.0 + 1.0 * 1.0) / 3.0
        expected_turn_booking_pass_rate = (0.0 + 1.0 + 1.0) / 3.0
        expected_turn_filter_pass_rate = (0.0 + 0.5 + 1.0) / 3.0
        expected_turn_rag_faithfulness = (0.4 + 0.6 + 1.0) / 3.0
        expected_turn_rag_precision = (0.5 + 0.25) / 2.0
        rows = [
            {
                "evaluators": _persisted_evaluators(
                    _feedback_item("route_accuracy", score=1.0),
                    _feedback_item("route_turns", score=2.0),
                    _feedback_item("booking_no_unexpected_failure_rate", score=0.5),
                    _feedback_item(
                        "booking_outcome_per_turn",
                        value=json.dumps(
                            [
                                {"turn_index": 0, "pass_rate": 0.0},
                                {"turn_index": 1, "pass_rate": 1.0},
                            ],
                        ),
                    ),
                    _feedback_item("judge_consumer_quality", score=4.0),
                    _feedback_item("rag_faithfulness_mean", score=0.50),
                    _feedback_item(
                        "rag_per_turn",
                        value=json.dumps(
                            [
                                {
                                    "turn_index": 0,
                                    "faithfulness": 0.4,
                                    "answer_relevancy": 0.6,
                                    "context_precision": None,
                                    "context_recall": None,
                                },
                                {
                                    "turn_index": 1,
                                    "faithfulness": 0.6,
                                    "answer_relevancy": 0.8,
                                    "context_precision": 0.5,
                                    "context_recall": 1.0,
                                },
                            ],
                        ),
                    ),
                    _feedback_item("info_reference_subset_pass_rate", score=0.5),
                    _feedback_item("info_reference_subset_turns", score=2.0),
                    _feedback_item("info_expected_filters_pass_rate", score=0.25),
                    _feedback_item(
                        "info_expected_filters_per_turn",
                        value=json.dumps(
                            [
                                {"turn_index": 0, "pass_rate": 0.0},
                                {"turn_index": 1, "pass_rate": 0.5},
                            ],
                        ),
                    ),
                ),
                "error": None,
            },
            {
                "evaluators": _persisted_evaluators(
                    _feedback_item("route_accuracy", score=0.5),
                    _feedback_item("route_turns", score=4.0),
                    _feedback_item("booking_no_unexpected_failure_rate", score=1.0),
                    _feedback_item(
                        "booking_outcome_per_turn",
                        value=json.dumps(
                            [{"turn_index": 0, "pass_rate": 1.0}],
                        ),
                    ),
                    _feedback_item("judge_consumer_quality", score=2.0),
                    _feedback_item("rag_faithfulness_mean", score=1.0),
                    _feedback_item(
                        "rag_per_turn",
                        value=json.dumps(
                            [
                                {
                                    "turn_index": 0,
                                    "faithfulness": 1.0,
                                    "answer_relevancy": 0.4,
                                    "context_precision": 0.25,
                                    "context_recall": 0.75,
                                },
                            ],
                        ),
                    ),
                    _feedback_item("info_reference_subset_pass_rate", score=1.0),
                    _feedback_item("info_reference_subset_turns", score=1.0),
                    _feedback_item("info_expected_filters_pass_rate", score=0.75),
                    _feedback_item(
                        "info_expected_filters_per_turn",
                        value=json.dumps(
                            [{"turn_index": 0, "pass_rate": 1.0}],
                        ),
                    ),
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
        case_summary = _as_dict(summary["case_based_summary"])
        turn_summary = _as_dict(summary["turn_based_summary"])

        assert case_summary["route_accuracy"] == expected_case_route_accuracy
        assert case_summary["consumer_quality"] == expected_case_consumer_quality
        assert case_summary["rag_faithfulness_mean"] == expected_case_rag_faithfulness
        assert (
            case_summary["info_expected_filters_pass_rate"]
            == expected_case_filter_pass_rate
        )
        assert turn_summary["route_accuracy"] == expected_turn_route_accuracy
        assert turn_summary["info_reference_subset_pass_rate"] == (
            expected_turn_reference_pass_rate
        )
        assert turn_summary["booking_no_unexpected_failure_rate"] == (
            expected_turn_booking_pass_rate
        )
        assert turn_summary["info_expected_filters_pass_rate"] == (
            expected_turn_filter_pass_rate
        )
        assert turn_summary["rag_faithfulness_mean"] == expected_turn_rag_faithfulness
        assert turn_summary["rag_context_precision_mean"] == (
            expected_turn_rag_precision
        )
        assert "consumer_quality" not in turn_summary


class TestComputeLatencySummary:
    """compute_latency_summary() reads normalized persisted feedback."""

    def test_latency_summary_parses_nested_feedback_items(
        self,
        tmp_path: Path,
    ) -> None:
        """Latency quantiles are computed from persisted ``latency_per_turn`` data."""
        expected_booking_p50 = 15.0
        expected_info_p95 = 5.0
        results_path = tmp_path / "results.jsonl"
        row = {
            "evaluators": _persisted_evaluators(
                _feedback_item(
                    "latency_per_turn",
                    value=json.dumps(
                        [
                            {"route": "booking", "latency_ms": 10.0},
                            {"route": "booking", "latency_ms": 20.0},
                            {"route": "info", "latency_ms": 5.0},
                        ],
                    ),
                ),
            ),
        }
        results_path.write_text(f"{json.dumps(row)}\n", encoding="utf-8")

        summary = compute_latency_summary(results_path)
        quantiles = _as_dict(summary["latency_quantiles_ms"])
        booking_quantiles = _as_dict(quantiles["booking"])
        info_quantiles = _as_dict(quantiles["info"])

        assert booking_quantiles["p50_ms"] == expected_booking_p50
        assert info_quantiles["p95_ms"] == expected_info_p95
