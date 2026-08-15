"""Result extraction and aggregation utilities for evaluation runs.

Provides Protocol stubs for LangSmith result objects, helpers for extracting
per-result data (case/example IDs, outputs, feedback), and functions for
aggregating summary statistics across all results.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

import pandas as pd
from pydantic import TypeAdapter, ValidationError

from eval._utils import json_safe

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Structural protocols for LangSmith result objects
# ---------------------------------------------------------------------------

_FEEDBACK_PAIR_LIST_ADAPTER = TypeAdapter(list[tuple[str, Any]])

_CASE_SUMMARY_SCORE_KEY_MAP: dict[str, str] = {
    "route_accuracy": "route_accuracy",
    "routing_accuracy": "route_accuracy",
    "booking_no_unexpected_failure_rate": "booking_no_unexpected_failure_rate",
    "judge_consumer_quality": "consumer_quality",
    "judge_grounding_faithfulness": "grounding",
    "judge_injection_resistance": "injection_resistance",
    "rag_faithfulness_mean": "rag_faithfulness_mean",
    "rag_answer_relevancy_mean": "rag_answer_relevancy_mean",
    "rag_context_precision_mean": "rag_context_precision_mean",
    "rag_context_recall_mean": "rag_context_recall_mean",
    "info_reference_subset_pass_rate": "info_reference_subset_pass_rate",
    "info_expected_filters_pass_rate": "info_expected_filters_pass_rate",
}

_TURN_WEIGHTED_SCORE_KEY_MAP: dict[str, tuple[str, str]] = {
    "route_accuracy": ("route_accuracy", "route_turns"),
    "info_reference_subset_pass_rate": (
        "info_reference_subset_pass_rate",
        "info_reference_subset_turns",
    ),
}

_PER_TURN_SCORE_VALUE_KEY_MAP: dict[str, tuple[str, str]] = {
    "booking_outcome_per_turn": ("booking_no_unexpected_failure_rate", "pass_rate"),
    "info_expected_filters_per_turn": (
        "info_expected_filters_pass_rate",
        "pass_rate",
    ),
}

_RAG_PER_TURN_SCORE_KEY_MAP: dict[str, str] = {
    "faithfulness": "rag_faithfulness_mean",
    "answer_relevancy": "rag_answer_relevancy_mean",
    "context_precision": "rag_context_precision_mean",
    "context_recall": "rag_context_recall_mean",
}


class SupportsAttrs(Protocol):
    """Structural protocol for objects exposing attributes."""

    def __getattr__(self, name: str) -> object:
        """Return an attribute by name."""


class ExampleLike(SupportsAttrs, Protocol):
    """Protocol for dataset example objects used by LangSmith."""

    id: object | None
    example_id: object | None
    case_id: object | None
    inputs: Mapping[str, object] | SupportsAttrs | None
    tags: Iterable[object] | None


class RunLike(SupportsAttrs, Protocol):
    """Protocol for LangSmith run objects."""

    id: object | None
    run_id: object | None
    outputs: Mapping[str, object] | None
    tags: Iterable[object] | None


class FeedbackItemLike(SupportsAttrs, Protocol):
    """Protocol for evaluation feedback items."""

    key: object | None
    name: object | None
    score: object | None
    value: object | None
    comment: object | None


class ResultLike(SupportsAttrs, Protocol):
    """Protocol for LangSmith evaluation result items."""

    example: ExampleLike | Mapping[str, object] | None
    dataset_item: ExampleLike | Mapping[str, object] | None
    run: RunLike | Mapping[str, object] | None
    parent_run: RunLike | Mapping[str, object] | None
    tags: Iterable[object] | None
    tag: Iterable[object] | None
    feedback: Mapping[str, object] | Iterable[FeedbackItemLike] | None
    evaluation_results: Mapping[str, object] | Iterable[FeedbackItemLike] | None
    evaluations: Mapping[str, object] | Iterable[FeedbackItemLike] | None
    error: object | None
    exception: object | None


@dataclass
class TurnMetricAggregation:
    """Accumulates intermediate values for turn-based summary metrics.

    Attributes:
        weighted_score_totals: Sum of ``score * weight`` for weighted metrics.
        weight_totals: Sum of weights for weighted metrics.
        per_turn_scores: Raw per-turn score lists for metrics derived from
            per-turn payloads.

    """

    weighted_score_totals: dict[str, float]
    weight_totals: dict[str, float]
    per_turn_scores: dict[str, list[float]]


# ---------------------------------------------------------------------------
# Top-level result row builder
# ---------------------------------------------------------------------------


def _build_results_row(result: ResultLike | Mapping[str, object]) -> dict[str, object]:
    """Build a JSON row containing outputs and evaluator feedback.

    Args:
        result: Evaluation result item from LangSmith.

    Returns:
        Serialized row for experiment results.

    """
    outputs = _extract_run_outputs(result)
    return {
        "case_id": _extract_case_id(result),
        "example_id": _extract_example_id(result),
        "outputs_summary": _summarize_outputs(outputs),
        "evaluators": _collect_feedback(result),
        "error": _extract_error(result),
    }


def _extract_case_id(result: ResultLike | Mapping[str, object]) -> str | None:
    """Extract a case identifier from an evaluation result.

    Args:
        result: Evaluation result item from LangSmith.

    Returns:
        Case identifier string if available.

    """
    example = _as_attr_source(
        _get_attr(result, "example") or _get_attr(result, "dataset_item"),
    )
    if example is None:
        return None
    value = _get_attr(_as_attr_source(_get_attr(example, "inputs")), "case_id")
    if isinstance(value, str):
        return value
    if value is not None:
        return str(value)
    value = _get_attr(example, "case_id")
    if isinstance(value, str):
        return value
    if value is not None:
        return str(value)
    return None


def _extract_example_id(result: ResultLike | Mapping[str, object]) -> str | None:
    """Extract the LangSmith example id from an evaluation result.

    Args:
        result: Evaluation result item from LangSmith.

    Returns:
        Example identifier string if available.

    """
    example = _as_attr_source(
        _get_attr(result, "example") or _get_attr(result, "dataset_item"),
    )
    if example is None:
        return None
    for value in (_get_attr(example, "id"), _get_attr(example, "example_id")):
        if isinstance(value, str):
            return value
        if value is not None:
            return str(value)
    return None


def _summarize_outputs(outputs: Mapping[str, object]) -> dict[str, object]:
    """Summarize outputs without including full transcripts.

    Args:
        outputs: Output payload from the target run.

    Returns:
        Summary containing turn counts and final routes.

    """
    turns: list[object] | None = None
    for key in ("turn_outputs", "turns", "messages"):
        candidate = outputs.get(key)
        if isinstance(candidate, list):
            turns = candidate
            break
    final_routes: list[object] = []
    if turns:
        for turn in turns:
            if isinstance(turn, Mapping):
                route = (
                    turn.get("route_pred")
                    or turn.get("final_route")
                    or turn.get("route")
                    or turn.get("routed_to")
                )
                response = turn.get("response")
                if route is None and isinstance(response, Mapping):
                    route = response.get("route")
                final_routes.append(route)
            else:
                final_routes.append(None)
    return {
        "num_turns": len(turns) if turns is not None else None,
        "final_routes": final_routes or None,
    }


def _collect_feedback(
    result: ResultLike | Mapping[str, object],
) -> dict[str, dict[str, object]]:
    """Collect evaluator feedback keyed by evaluator name.

    Args:
        result: Evaluation result item from LangSmith.

    Returns:
        Mapping of evaluator key to feedback details.

    """
    feedback: dict[str, dict[str, object]] = {}
    raw_feedback = (
        _get_attr(result, "feedback")
        or _get_attr(result, "evaluation_results")
        or _get_attr(result, "evaluations")
    )
    if not raw_feedback:
        return feedback
    if isinstance(raw_feedback, Mapping):
        for key, payload in raw_feedback.items():
            if isinstance(payload, Mapping):
                # json_safe() always returns a dict for a Mapping input; its
                # signature is deliberately `object` to cover every branch of
                # its recursive serialization, so that guarantee isn't visible
                # to the type checker from the call alone.
                feedback[key] = cast("dict[str, object]", json_safe(payload))
            else:
                feedback[key] = {"value": json_safe(payload)}
        return feedback
    if isinstance(raw_feedback, Iterable):
        feedback.update(_normalize_feedback(raw_feedback))
    return feedback


def _normalize_feedback(feedback: object) -> dict[str, dict[str, object]]:
    """Normalize raw or persisted evaluator feedback into a flat metric map.

    Args:
        feedback: Raw LangSmith feedback object, persisted evaluator mapping, or
            a nested ``{"results": {"value": ...}}`` structure from ``results.jsonl``.

    Returns:
        Mapping of metric key to a payload containing ``score``, ``value``, and
        ``comment`` fields when present.

    """
    if isinstance(feedback, Mapping):
        nested_results = feedback.get("results")
        if nested_results is not None:
            return _normalize_feedback(nested_results)

        raw_value = feedback.get("value")
        if (
            raw_value is not None
            and isinstance(raw_value, Iterable)
            and not isinstance(raw_value, (str, bytes, Mapping))
        ):
            return _normalize_feedback_from_sequence(raw_value)

        return _normalize_feedback_from_mapping(feedback)

    if isinstance(feedback, Iterable) and not isinstance(feedback, (str, bytes)):
        return _normalize_feedback_from_sequence(feedback)

    return {}


def _normalize_feedback_from_mapping(
    feedback: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    """Normalize a flat metric-keyed feedback mapping.

    Args:
        feedback: Mapping keyed by metric name.

    Returns:
        Flat metric map with normalized payloads.

    """
    normalized: dict[str, dict[str, object]] = {}

    for key, payload in feedback.items():
        if not isinstance(payload, Mapping):
            continue
        normalized[str(key)] = _build_feedback_payload(payload)

    return normalized


def _build_feedback_payload(payload: Mapping[str, object]) -> dict[str, object]:
    """Build a normalized feedback payload from a mapping.

    Args:
        payload: Raw payload mapping.

    Returns:
        Payload containing only ``score``, ``value``, and ``comment`` when present.

    """
    normalized = {
        "score": json_safe(payload.get("score")),
        "value": json_safe(payload.get("value")),
        "comment": json_safe(payload.get("comment")),
    }
    return {key: value for key, value in normalized.items() if value is not None}


def _normalize_feedback_from_sequence(
    feedback: Iterable[object],
) -> dict[str, dict[str, object]]:
    """Normalize a sequence of LangSmith feedback entries.

    Args:
        feedback: Iterable of feedback entries, each represented as an object,
            mapping, or persisted list-of-pairs structure.

    Returns:
        Flat metric map with normalized payloads.

    """
    normalized: dict[str, dict[str, object]] = {}

    for item in feedback:
        payload = _coerce_feedback_entry(item)
        if payload is None:
            continue

        key = payload.get("key") or payload.get("name")
        if not key:
            continue

        normalized[str(key)] = _build_feedback_payload(payload)

    return normalized


def _coerce_feedback_entry(item: object) -> dict[str, object] | None:
    """Coerce a feedback item into a plain dictionary.

    Args:
        item: Feedback entry represented as an object, mapping, or list of
            ``(key, value)`` pairs.

    Returns:
        Plain dictionary representation of the feedback item, or ``None`` when
        the entry cannot be interpreted.

    """
    if item is None:
        return None

    if isinstance(item, Mapping):
        return dict(item)

    try:
        pairs = _FEEDBACK_PAIR_LIST_ADAPTER.validate_python(item)
    except ValidationError:
        pairs = None

    if pairs is not None:
        return {str(key): value for key, value in pairs}

    attr_source = _as_attr_source(item)
    if attr_source is None:
        return None

    key = _get_attr(attr_source, "key") or _get_attr(attr_source, "name")
    if key is None:
        return None

    return {
        "key": key,
        "score": _get_attr(attr_source, "score"),
        "value": _get_attr(attr_source, "value"),
        "comment": _get_attr(attr_source, "comment"),
    }


def _extract_run_outputs(
    result: ResultLike | Mapping[str, object],
) -> dict[str, object]:
    """Extract outputs produced by the target run.

    Args:
        result: Evaluation result item from LangSmith.

    Returns:
        Output dictionary for the run (may be empty).

    """
    run = _get_attr(result, "run") or _get_attr(result, "parent_run")
    outputs = _get_attr(_as_attr_source(run), "outputs")
    if isinstance(outputs, Mapping):
        return dict(outputs)
    return {}


def _extract_error(result: ResultLike | Mapping[str, object]) -> str | None:
    """Extract error information from a result if present.

    Args:
        result: Evaluation result item from LangSmith.

    Returns:
        Error message string if present.

    """
    error = _get_attr(result, "error") or _get_attr(result, "exception")
    if error:
        return str(error)
    return None


def _get_attr(
    mapping_or_obj: Mapping[str, object] | SupportsAttrs | None,
    key: str,
) -> object | None:
    """Read a key from a mapping or attribute from an object.

    Args:
        mapping_or_obj: Mapping or object to read from.
        key: Key or attribute name.

    Returns:
        The retrieved value or None.

    """
    if mapping_or_obj is None:
        return None
    if isinstance(mapping_or_obj, Mapping):
        return mapping_or_obj.get(key)
    return getattr(mapping_or_obj, key, None)


def _as_attr_source(
    value: object | None,
) -> Mapping[str, object] | SupportsAttrs | None:
    """Narrow a value to a supported attribute source if possible.

    Args:
        value: Value to narrow.

    Returns:
        The value if it supports mapping or attribute access; otherwise None.

    """
    if value is None:
        return None
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "__getattr__"):
        return value  # type: ignore[return-value]
    return None


# ---------------------------------------------------------------------------
# Summary aggregation
# ---------------------------------------------------------------------------


def _load_feedback_value_records(raw_value: object) -> list[dict[str, object]]:
    """Load a feedback ``value`` payload into a list of record dicts.

    Args:
        raw_value: Feedback payload that may already be a list of mappings or a
            JSON-encoded string containing such a list.

    Returns:
        List of dictionary records extracted from the payload.

    """
    if isinstance(raw_value, list):
        return [dict(item) for item in raw_value if isinstance(item, Mapping)]

    if not raw_value:
        return []

    with suppress(json.JSONDecodeError, TypeError):
        parsed = json.loads(str(raw_value))
        if isinstance(parsed, list):
            return [dict(item) for item in parsed if isinstance(item, Mapping)]

    return []


def compute_latency_summary(results_path: Path) -> dict[str, object]:
    """Compute p50/p95/p99 per-route latency quantiles from results.jsonl.

    Reads ``latency_per_turn`` JSON value fields from every row in the file,
    flattens all ``{route, latency_ms}`` records, and computes quantiles
    grouped by orchestration route.

    Args:
        results_path: Path to the ``results.jsonl`` file written by
            ``run_experiment.py``.

    Returns:
        Dict with a single ``"latency_quantiles_ms"`` key whose value is a
        mapping of route name → ``{p50_ms, p95_ms, p99_ms}``.  Returns an
        empty dict when no latency data is present.

    """
    records: list[dict[str, object]] = []
    with results_path.open(encoding="utf-8") as fh:
        for raw_line in fh:
            stripped = raw_line.strip()
            if not stripped:
                continue
            case = json.loads(stripped)
            feedback = _normalize_feedback(case.get("evaluators", {}))
            latency_payload = feedback.get("latency_per_turn", {})
            if isinstance(latency_payload, Mapping):
                records.extend(
                    _load_feedback_value_records(latency_payload.get("value")),
                )

    if not records:
        return {}

    df = pd.DataFrame(records)
    df["route"] = df["route"].fillna("unknown")
    grp = df.groupby("route")["latency_ms"]
    # Series.quantile() is stubbed to also return a Series, to cover its
    # list-of-quantiles overload -- not the case for the single-float q used
    # here, which always returns one scalar.
    result: dict[str, object] = {
        str(route): {
            "p50_ms": round(float(cast("float", series.quantile(0.50))), 1),
            "p95_ms": round(float(cast("float", series.quantile(0.95))), 1),
            "p99_ms": round(float(cast("float", series.quantile(0.99))), 1),
        }
        for route, series in grp
    }
    return {"latency_quantiles_ms": result}


def format_latency_table(latency: dict[str, object]) -> str:
    """Format per-route p50/p95/p99 latency quantiles as a plain-text table.

    Args:
        latency: Dict returned by ``compute_latency_summary``, containing
            a ``"latency_quantiles_ms"`` key with per-route quantile data.

    Returns:
        Multi-line string with a header and one row per route, or an empty
        string when ``latency`` contains no quantile data.

    """
    quantiles = latency.get("latency_quantiles_ms")
    if not isinstance(quantiles, dict) or not quantiles:
        return ""
    header = f"  {'Route':<12} {'p50 ms':>8} {'p95 ms':>8} {'p99 ms':>8}"
    lines = ["\nPer-route latency (ms):", header, "  " + "-" * (len(header) - 2)]
    for route, stats in sorted(quantiles.items()):
        if isinstance(stats, dict):
            lines.append(
                f"  {route:<12} {stats['p50_ms']:>8.0f}"
                f" {stats['p95_ms']:>8.0f} {stats['p99_ms']:>8.0f}",
            )
    return "\n".join(lines)


def _summarize_results(
    result_rows: Iterable[Mapping[str, object]],
    context: Any,  # noqa: ANN401
) -> dict[str, object]:
    """Aggregate summary statistics from evaluation rows.

    Args:
        result_rows: Iterable of evaluation result rows.
        context: ``SummaryContext`` metadata for the summary output.

    Returns:
        Summary dictionary with aggregate metrics.

    """
    case_metric_values = _init_case_metric_values()
    turn_metric_values = _init_turn_metric_values()
    num_examples = 0
    num_failed_runs = 0

    for row in result_rows:
        num_examples += 1
        if row.get("error"):
            num_failed_runs += 1
        feedback = row.get("evaluators", {})
        if isinstance(feedback, Mapping):
            normalized_feedback = _normalize_feedback(feedback)
            _update_case_metric_values(case_metric_values, normalized_feedback)
            _update_turn_metric_values(turn_metric_values, normalized_feedback)

    summary = _build_summary_base(context, num_examples, num_failed_runs)
    case_based_summary = _mean_metrics(case_metric_values)
    turn_based_summary = _build_turn_based_summary(turn_metric_values)

    if case_based_summary:
        summary["case_based_summary"] = case_based_summary
    if turn_based_summary:
        summary["turn_based_summary"] = turn_based_summary

    return summary


def _init_case_metric_values() -> dict[str, list[float]]:
    """Initialize storage for case-weighted summary aggregation.

    Returns:
        Mapping of metric names to collected float values.

    """
    return {
        "route_accuracy": [],
        "booking_no_unexpected_failure_rate": [],
        "consumer_quality": [],
        "grounding": [],
        "injection_resistance": [],
        "rag_faithfulness_mean": [],
        "rag_answer_relevancy_mean": [],
        "rag_context_precision_mean": [],
        "rag_context_recall_mean": [],
        "info_reference_subset_pass_rate": [],
        "info_expected_filters_pass_rate": [],
    }


def _update_case_metric_values(
    metric_values: dict[str, list[float]],
    feedback: Mapping[str, dict[str, object]],
) -> None:
    """Update case-weighted metric values from evaluator feedback.

    Args:
        metric_values: Aggregated metric values to update.
        feedback: Normalized evaluator feedback mapping.

    """
    for key, payload in feedback.items():
        summary_key = _CASE_SUMMARY_SCORE_KEY_MAP.get(key)
        if summary_key is None:
            continue

        score = _extract_numeric(payload.get("score"))
        if score is None:
            continue

        metric_values[summary_key].append(score)


def _init_turn_metric_values() -> TurnMetricAggregation:
    """Initialize storage for turn-based summary aggregation.

    Returns:
        Aggregation container for weighted and per-turn turn metrics.

    """
    weighted_score_totals = dict.fromkeys(_TURN_WEIGHTED_SCORE_KEY_MAP, 0.0)
    weight_totals = dict.fromkeys(_TURN_WEIGHTED_SCORE_KEY_MAP, 0.0)
    per_turn_scores = {
        "booking_no_unexpected_failure_rate": [],
        "info_expected_filters_pass_rate": [],
        "rag_faithfulness_mean": [],
        "rag_answer_relevancy_mean": [],
        "rag_context_precision_mean": [],
        "rag_context_recall_mean": [],
    }
    return TurnMetricAggregation(
        weighted_score_totals=weighted_score_totals,
        weight_totals=weight_totals,
        per_turn_scores=per_turn_scores,
    )


def _update_turn_metric_values(
    metric_values: TurnMetricAggregation,
    feedback: Mapping[str, dict[str, object]],
) -> None:
    """Update turn-based metric aggregates from normalized evaluator feedback.

    Args:
        metric_values: Aggregation container updated in place.
        feedback: Normalized evaluator feedback mapping.

    """
    for summary_key, (score_key, weight_key) in _TURN_WEIGHTED_SCORE_KEY_MAP.items():
        _accumulate_weighted_turn_metric(
            metric_values,
            feedback,
            summary_key=summary_key,
            score_key=score_key,
            weight_key=weight_key,
        )

    for feedback_key, (summary_key, score_field) in (
        _PER_TURN_SCORE_VALUE_KEY_MAP.items()
    ):
        payload = feedback.get(feedback_key)
        if not isinstance(payload, Mapping):
            continue
        _extend_turn_scores_from_payload(
            metric_values.per_turn_scores[summary_key],
            payload,
            score_field=score_field,
        )

    rag_payload = feedback.get("rag_per_turn")
    if isinstance(rag_payload, Mapping):
        _extend_rag_turn_scores(metric_values.per_turn_scores, rag_payload)


def _accumulate_weighted_turn_metric(
    metric_values: TurnMetricAggregation,
    feedback: Mapping[str, dict[str, object]],
    *,
    summary_key: str,
    score_key: str,
    weight_key: str,
) -> None:
    """Accumulate a turn-weighted metric using case score and turn count.

    Args:
        metric_values: Aggregation container updated in place.
        feedback: Normalized evaluator feedback mapping.
        summary_key: Destination summary metric name.
        score_key: Feedback key containing the case-level score.
        weight_key: Feedback key containing the number of scored turns.

    """
    score_payload = feedback.get(score_key)
    weight_payload = feedback.get(weight_key)
    if (
        not isinstance(score_payload, Mapping)
        or not isinstance(weight_payload, Mapping)
    ):
        return

    score = _extract_numeric(score_payload.get("score"))
    weight = _extract_numeric(weight_payload.get("score"))
    if score is None or weight is None or weight <= 0:
        return

    metric_values.weighted_score_totals[summary_key] += score * weight
    metric_values.weight_totals[summary_key] += weight


def _extend_turn_scores_from_payload(
    scores: list[float],
    payload: Mapping[str, object],
    *,
    score_field: str,
) -> None:
    """Extend a metric score list from per-turn payload records.

    Args:
        scores: Score list updated in place.
        payload: Feedback payload containing a ``value`` field.
        score_field: Record field containing the per-turn numeric score.

    """
    for record in _load_feedback_value_records(payload.get("value")):
        score = _extract_numeric(record.get(score_field))
        if score is not None:
            scores.append(score)


def _extend_rag_turn_scores(
    per_turn_scores: dict[str, list[float]],
    payload: Mapping[str, object],
) -> None:
    """Extend turn-based RAG score lists from ``rag_per_turn`` payloads.

    Args:
        per_turn_scores: Per-metric score lists updated in place.
        payload: Normalized ``rag_per_turn`` feedback payload.

    """
    for record in _load_feedback_value_records(payload.get("value")):
        for record_key, summary_key in _RAG_PER_TURN_SCORE_KEY_MAP.items():
            score = _extract_numeric(record.get(record_key))
            if score is not None:
                per_turn_scores[summary_key].append(score)


def _build_turn_based_summary(
    metric_values: TurnMetricAggregation,
) -> dict[str, object]:
    """Build the turn-based summary from accumulated intermediate values.

    Args:
        metric_values: Aggregation container populated across all cases.

    Returns:
        Mapping of metric names to turn-based summary values.

    """
    summary: dict[str, object] = {}

    for metric_name, weighted_total in metric_values.weighted_score_totals.items():
        total_weight = metric_values.weight_totals[metric_name]
        if total_weight <= 0:
            continue
        summary[metric_name] = weighted_total / total_weight

    summary.update(_mean_metrics(metric_values.per_turn_scores))
    return summary


def _extract_numeric(value: object) -> float | None:
    """Extract a float value from an arbitrary object.

    Args:
        value: Value to parse.

    Returns:
        Float value when possible, otherwise None.

    """
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _build_summary_base(
    context: Any,  # noqa: ANN401
    num_examples: int,
    num_failed_runs: int,
) -> dict[str, object]:
    """Build the base summary metadata.

    Args:
        context: ``SummaryContext`` carrying experiment metadata.
        num_examples: Number of evaluated examples.
        num_failed_runs: Number of failed runs.

    Returns:
        Summary payload with base metadata.

    """
    return {
        "experiment_name": context.experiment_name,
        "dataset_name": context.dataset_name,
        "max_concurrency": context.max_concurrency,
        "started_at": context.started_at.isoformat(),
        "finished_at": context.finished_at.isoformat(),
        "upload_results": context.upload_results,
        "num_examples": num_examples,
        "num_failed_runs": num_failed_runs,
    }


def _mean_metrics(metric_values: Mapping[str, list[float]]) -> dict[str, object]:
    """Compute arithmetic means for metrics with collected values.

    Args:
        metric_values: Mapping of metric names to collected float values.

    Returns:
        Mapping of metric names to mean values, omitting keys with no values.

    """
    means: dict[str, object] = {}
    for key, values in metric_values.items():
        if not values:
            continue
        means[key] = statistics.mean(values)
    return means


def _build_error_summary(
    context: Any,  # noqa: ANN401
    error: Exception,
) -> dict[str, object]:
    """Build a minimal summary when the experiment fails.

    Args:
        context: ``SummaryContext`` carrying experiment metadata.
        error: Exception that caused the failure.

    Returns:
        Summary payload containing the failure details.

    """
    return {
        "experiment_name": context.experiment_name,
        "dataset_name": context.dataset_name,
        "max_concurrency": context.max_concurrency,
        "started_at": context.started_at.isoformat(),
        "finished_at": context.finished_at.isoformat(),
        "upload_results": context.upload_results,
        "error": str(error),
    }
