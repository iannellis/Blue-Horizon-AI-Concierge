"""Run LangSmith evaluations for the hotel agent dataset and persist artifacts."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Iterable, Mapping, MutableMapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol

from langsmith.evaluation import aevaluate

from eval.config import MetadataConfig, load_eval_config
from eval.evaluators import (
    eval_injection_tripwires,
    eval_llm_rubrics,
    eval_rooms_outcome_and_invariants,
    eval_routing_accuracy,
)
from eval.langsmith_target import run_example

if TYPE_CHECKING:
    from pathlib import Path

@dataclass(frozen=True)
class RunArtifacts:
    """Stores filesystem paths for evaluation artifacts.

    Attributes:
        output_dir: Directory containing the artifact files for this run.
        turn_results_path: JSONL file with per-example turn outputs.
        eval_results_path: JSONL file with per-example evaluator results.
        summary_path: JSON file with aggregate evaluation statistics.

    """

    output_dir: Path
    turn_results_path: Path
    eval_results_path: Path
    summary_path: Path


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


async def main() -> None:
    """Run the LangSmith experiment and persist local artifacts."""
    cfg = load_eval_config()
    dataset_name = cfg.experiment.dataset_name
    experiment_name = cfg.experiment.experiment_name
    max_concurrency = cfg.experiment.max_concurrency
    output_root = cfg.experiment.output_dir
    artifacts = _build_output_paths(output_root)

    # Setting LANGSMITH_TEST_CACHE enables caching of example runs to reduce cost/time.
    metadata = _build_metadata(cfg.metadata)

    evaluators = [
        eval_routing_accuracy,
        eval_injection_tripwires,
        eval_rooms_outcome_and_invariants,
        eval_llm_rubrics,
    ]

    aevaluate_kwargs: MutableMapping[str, Any] = {
        "evaluators": evaluators,
        "max_concurrency": max_concurrency,
        "upload_results": True,
        "metadata": metadata,
    }
    signature = inspect.signature(aevaluate)
    if "experiment_name" in signature.parameters:
        aevaluate_kwargs["experiment_name"] = experiment_name
    elif "experiment_prefix" in signature.parameters:
        aevaluate_kwargs["experiment_prefix"] = experiment_name

    results = await aevaluate(run_example, dataset_name, **aevaluate_kwargs)
    results_list: list[Any] = []
    if results is not None:
        results_list.extend([item async for item in results])

    turn_rows = [_build_turn_row(result) for result in results_list]
    eval_rows = [_build_eval_row(result) for result in results_list]

    _write_jsonl(artifacts.turn_results_path, turn_rows)
    _write_jsonl(artifacts.eval_results_path, eval_rows)
    summary = _summarize_results(eval_rows)
    with artifacts.summary_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, ensure_ascii=True, indent=2))


def _build_output_paths(base_dir: Path) -> RunArtifacts:
    """Create a timestamped output directory and artifact paths.

    Args:
        base_dir: Base output directory.

    Returns:
        RunArtifacts containing paths for local JSON artifacts.

    """
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_dir = base_dir / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return RunArtifacts(
        output_dir=output_dir,
        turn_results_path=output_dir / "turn_results.jsonl",
        eval_results_path=output_dir / "eval_results.jsonl",
        summary_path=output_dir / "summary.json",
    )


def _build_metadata(metadata_cfg: MetadataConfig) -> dict[str, object]:
    """Build a metadata dictionary for the LangSmith experiment.

    Args:
        metadata_cfg: Metadata configuration for optional labels.

    Returns:
        Metadata payload for LangSmith experiment tracking.

    """
    metadata: dict[str, object] = {
        "git_sha": metadata_cfg.git_sha,
        "router_model": metadata_cfg.router_model,
        "judge_model": metadata_cfg.judge_model,
        "schema_version": metadata_cfg.schema_version,
    }
    return {key: value for key, value in metadata.items() if value}


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
    for value in (
        _get_attr(example, "id"),
        _get_attr(example, "example_id"),
        _get_attr(example, "case_id"),
        _get_attr(_as_attr_source(_get_attr(example, "inputs")), "case_id"),
    ):
        if isinstance(value, str):
            return value
        if value is not None:
            return str(value)
    return None


def _extract_tags(result: ResultLike | Mapping[str, object]) -> list[str] | None:
    """Extract tags from an evaluation result.

    Args:
        result: Evaluation result item from LangSmith.

    Returns:
        List of tag strings if available.

    """
    for source in ("tags", "tag"):
        tags = _get_attr(result, source)
        if isinstance(tags, Iterable) and not isinstance(tags, (str, bytes)):
            return [str(tag) for tag in tags]
    example = _get_attr(result, "example") or _get_attr(result, "dataset_item")
    tags = _get_attr(_as_attr_source(example), "tags")
    if isinstance(tags, Iterable) and not isinstance(tags, (str, bytes)):
        return [str(tag) for tag in tags]
    run = _get_attr(result, "run")
    tags = _get_attr(_as_attr_source(run), "tags")
    if isinstance(tags, Iterable) and not isinstance(tags, (str, bytes)):
        return [str(tag) for tag in tags]
    return None


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


def _extract_run_id(result: ResultLike | Mapping[str, object]) -> str | None:
    """Extract LangSmith run id from the evaluation result.

    Args:
        result: Evaluation result item from LangSmith.

    Returns:
        Run identifier if available.

    """
    run = _as_attr_source(_get_attr(result, "run"))
    for value in (_get_attr(run, "id"), _get_attr(run, "run_id")):
        if isinstance(value, str):
            return value
        if value is not None:
            return str(value)
    return None


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
                feedback[key] = dict(payload)
            else:
                feedback[key] = {"value": payload}
        return feedback
    if isinstance(raw_feedback, Iterable):
        for item in raw_feedback:
            if item is None:
                continue
            key = _get_attr(item, "key") or _get_attr(item, "name")
            if not key:
                continue
            payload = {
                "score": _get_attr(item, "score"),
                "value": _get_attr(item, "value"),
                "comment": _get_attr(item, "comment"),
            }
            feedback[str(key)] = {k: v for k, v in payload.items() if v is not None}
    return feedback


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    """Write an iterable of JSON-serializable rows to a JSONL file.

    Args:
        path: Output file path.
        rows: Iterable of mapping rows to serialize.

    """
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _is_failure(value: object) -> bool:
    """Determine whether a value indicates a failure.

    Args:
        value: Value from evaluator feedback.

    Returns:
        True if the value indicates failure.

    """
    if value is None:
        return False
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return value == 0
    if isinstance(value, str):
        return value.strip().lower() in {"fail", "failed", "false", "0"}
    return False


def _summarize_results(
    eval_rows: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    """Aggregate summary statistics from evaluation rows.

    Args:
        eval_rows: Iterable of evaluation result rows.

    Returns:
        Summary dictionary with aggregate metrics.

    """
    routing_scores: list[float] = []
    judge_scores: list[float] = []
    tripwire_fail_count = 0
    invariant_fail_count = 0

    for row in eval_rows:
        feedback = row.get("evaluators", {})
        if not isinstance(feedback, Mapping):
            continue
        for key, payload in feedback.items():
            score = None
            value = None
            if isinstance(payload, Mapping):
                score = payload.get("score")
                value = payload.get("value")
            if key in {"route_accuracy", "routing_accuracy"} and score is not None:
                routing_scores.append(float(score))
            if "tripwire" in key and _is_failure(value if value is not None else score):
                tripwire_fail_count += 1
            if "invariant" in key and _is_failure(
                value if value is not None else score,
            ):
                invariant_fail_count += 1
            if (
                "rubric" in key or "judge" in key or "llm" in key
            ) and score is not None:
                judge_scores.append(float(score))

    def _mean(values: list[float]) -> float | None:
        """Compute the arithmetic mean for a list of values.

        Args:
            values: List of numeric values.

        Returns:
            The mean value, or None if the list is empty.

        """
        return sum(values) / len(values) if values else None

    return {
        "mean_route_accuracy": _mean(routing_scores),
        "tripwire_fail_count": tripwire_fail_count,
        "db_invariant_fail_count": invariant_fail_count,
        "mean_judge_score": _mean(judge_scores),
        "examples_count": len(list(eval_rows)),
    }


def _build_turn_row(result: ResultLike | Mapping[str, object]) -> dict[str, object]:
    """Build a JSON row containing turn outputs for an example.

    Args:
        result: Evaluation result item from LangSmith.

    Returns:
        Serialized row for turn-level outputs.

    """
    outputs = _extract_run_outputs(result)
    return {
        "case_id": _extract_case_id(result),
        "tags": _extract_tags(result),
        "per_turn_outputs": outputs.get("turn_outputs")
        or outputs.get("turns")
        or outputs.get("messages"),
        "final_db_schema": outputs.get("final_db_schema") or outputs.get("db_schema"),
        "error": _extract_error(result),
    }


def _build_eval_row(result: ResultLike | Mapping[str, object]) -> dict[str, object]:
    """Build a JSON row containing evaluator results for an example.

    Args:
        result: Evaluation result item from LangSmith.

    Returns:
        Serialized row for evaluation feedback.

    """
    return {
        "case_id": _extract_case_id(result),
        "run_id": _extract_run_id(result),
        "evaluators": _collect_feedback(result),
        "error": _extract_error(result),
    }


if __name__ == "__main__":
    asyncio.run(main())
