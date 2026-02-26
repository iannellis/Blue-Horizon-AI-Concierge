"""Result extraction and aggregation utilities for evaluation runs.

Provides Protocol stubs for LangSmith result objects, helpers for extracting
per-result data (case/example IDs, outputs, feedback), and functions for
aggregating summary statistics across all results.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Protocol, cast

from eval._utils import json_safe

# ---------------------------------------------------------------------------
# Structural protocols for LangSmith result objects
# ---------------------------------------------------------------------------


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
                feedback[key] = cast("dict[str, object]", json_safe(payload))
            else:
                feedback[key] = {"value": json_safe(payload)}
        return feedback
    if isinstance(raw_feedback, Iterable):
        for item in raw_feedback:
            if item is None:
                continue
            key = _get_attr(item, "key") or _get_attr(item, "name")
            if not key:
                continue
            payload = {
                "score": json_safe(_get_attr(item, "score")),
                "value": json_safe(_get_attr(item, "value")),
                "comment": json_safe(_get_attr(item, "comment")),
            }
            feedback[str(key)] = {k: v for k, v in payload.items() if v is not None}
    return feedback


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
    metric_values = _init_metric_values()
    num_examples = 0
    num_failed_runs = 0

    for row in result_rows:
        num_examples += 1
        if row.get("error"):
            num_failed_runs += 1
        feedback = row.get("evaluators", {})
        if isinstance(feedback, Mapping):
            _update_metric_values(metric_values, feedback)

    summary = _build_summary_base(context, num_examples, num_failed_runs)
    return summary | _mean_metrics(metric_values)


def _init_metric_values() -> dict[str, list[float]]:
    """Initialize metric storage for summary aggregation.

    Returns:
        Mapping of metric names to collected float values.

    """
    return {
        "route_accuracy": [],
        "rooms_no_unexpected_failure_rate": [],
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


def _update_metric_values(
    metric_values: dict[str, list[float]],
    feedback: Mapping[str, object],
) -> None:
    """Update aggregated metric values from evaluator feedback.

    Args:
        metric_values: Aggregated metric values to update.
        feedback: Evaluator feedback mapping.

    """
    for key, payload in feedback.items():
        if not isinstance(payload, Mapping):
            continue
        score = _extract_numeric(payload.get("score"))
        value = _extract_numeric(payload.get("value"))
        for metric, values in metric_values.items():
            metric_value = _extract_numeric(payload.get(metric))
            if metric_value is not None:
                values.append(metric_value)
        if key in {"route_accuracy", "routing_accuracy"} and score is not None:
            metric_values["route_accuracy"].append(score)
        if key == "rooms_no_unexpected_failure_rate" and score is not None:
            metric_values["rooms_no_unexpected_failure_rate"].append(score)
        if key == "info_reference_subset_pass_rate" and score is not None:
            metric_values["info_reference_subset_pass_rate"].append(score)
        if "expected_filters" in key and value is not None:
            metric_values["info_expected_filters_pass_rate"].append(value)


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
    """Compute mean metrics, skipping missing values.

    Args:
        metric_values: Mapping of metrics to collected values.

    Returns:
        Mapping of mean metric names to values.

    """
    mean_values = {
        "mean_route_accuracy": _mean(metric_values["route_accuracy"]),
        "mean_rooms_no_unexpected_failure_rate": _mean(
            metric_values["rooms_no_unexpected_failure_rate"],
        ),
        "mean_consumer_quality": _mean(metric_values["consumer_quality"]),
        "mean_grounding": _mean(metric_values["grounding"]),
        "mean_injection_resistance": _mean(metric_values["injection_resistance"]),
        "mean_rag_faithfulness_mean": _mean(
            metric_values["rag_faithfulness_mean"],
        ),
        "mean_rag_answer_relevancy_mean": _mean(
            metric_values["rag_answer_relevancy_mean"],
        ),
        "mean_rag_context_precision_mean": _mean(
            metric_values["rag_context_precision_mean"],
        ),
        "mean_rag_context_recall_mean": _mean(
            metric_values["rag_context_recall_mean"],
        ),
        "mean_info_reference_subset_pass_rate": _mean(
            metric_values["info_reference_subset_pass_rate"],
        ),
        "mean_info_expected_filters_pass_rate": _mean(
            metric_values["info_expected_filters_pass_rate"],
        ),
    }
    return {key: value for key, value in mean_values.items() if value is not None}


def _mean(values: list[float]) -> float | None:
    """Compute the arithmetic mean for a list of values.

    Args:
        values: List of numeric values.

    Returns:
        The mean value, or None if the list is empty.

    """
    return sum(values) / len(values) if values else None


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
