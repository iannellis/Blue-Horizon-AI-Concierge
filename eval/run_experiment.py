"""Run LangSmith evaluations for the hotel agent dataset and persist artifacts."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import platform
import re
from collections.abc import Iterable, Mapping, MutableMapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol, cast

from dotenv import load_dotenv
from langsmith import Client
from langsmith.evaluation import aevaluate

from blue_horizon.agents.rooms import set_eval_schema
from blue_horizon.config import load_app_config
from eval._utils import json_safe
from eval.config import MetadataConfig, load_eval_config
from eval.evaluators import (
    eval_info_expected_filters,
    eval_info_reference_subset,
    eval_injection_tripwires,
    eval_llm_rubrics,
    eval_rag_metrics_info_turns,
    eval_rooms_outcome_and_invariants,
    eval_routing_accuracy,
)
from eval.langsmith_target import run_example
from eval.rooms_schema_manager import (
    create_case_schema,
    generate_schema_name,
    open_schema_pool,
    teardown_schema,
)

if TYPE_CHECKING:
    from contextvars import Token
    from pathlib import Path

    from langsmith.schemas import Example
    from psycopg_pool import AsyncConnectionPool

    from eval.config import EvalConfig

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunArtifacts:
    """Stores filesystem paths for evaluation artifacts.

    Attributes:
        output_dir: Directory containing the artifact files for this run.
        results_path: JSONL file with per-example results and evaluator feedback.
        summary_path: JSON file with aggregate evaluation statistics.

    """

    output_dir: Path
    results_path: Path
    summary_path: Path


@dataclass(frozen=True)
class SummaryContext:
    """Stores experiment metadata for summary generation.

    Attributes:
        dataset_name: Name of the evaluated dataset.
        experiment_name: Name of the experiment run.
        max_concurrency: Max concurrency used during evaluation.
        started_at: Experiment start timestamp.
        finished_at: Experiment completion timestamp.
        upload_results: Whether LangSmith uploads were enabled.

    """

    dataset_name: str
    experiment_name: str
    max_concurrency: int
    started_at: datetime
    finished_at: datetime
    upload_results: bool


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


def _configure_logging(experiment_name: str, log_dir: Path) -> None:
    """Configure file-based logging for the evaluation run.

    Writes logs to ``<log_dir>/<experiment_name>.log`` using a file handler.
    Preserves any existing console handlers for color-coded terminal output.

    Args:
        experiment_name: Name of the current experiment, used as the log
            filename stem.
        log_dir: Directory where log files should be written.

    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{experiment_name}.log"

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ),
    )

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(file_handler)


def _sanitize_notes(notes: str) -> str:
    """Sanitize notes into a filesystem-friendly token.

    Args:
        notes: Raw run notes.

    Returns:
        Sanitized notes token with only letters, digits, and underscores.

    """
    replaced = notes.strip().replace(" ", "_")
    token = re.sub(r"[^A-Za-z0-9_]", "", replaced)
    return token[:40]


def _format_timestamp(timestamp: datetime) -> str:
    """Format a timestamp for experiment naming.

    Args:
        timestamp: Timestamp to format.

    Returns:
        Timestamp string formatted as YYYYMMDD_HHMMSS.

    """
    return timestamp.strftime("%Y%m%d_%H%M%S")


def _build_experiment_name(
    prefix: str,
    run_notes: str | None,
    started_at: datetime,
) -> str:
    """Build the experiment name using the prefix, timestamp, and notes.

    Args:
        prefix: Base prefix for the experiment name.
        run_notes: Optional run notes to append.
        started_at: Timestamp for the experiment start.

    Returns:
        The formatted experiment name.

    """
    timestamp = _format_timestamp(started_at)
    name = f"{prefix}_{timestamp}"
    if run_notes:
        notes_token = _sanitize_notes(run_notes)
        if notes_token:
            name = f"{name}_{notes_token}"
    return name


def _build_output_paths(base_dir: Path, experiment_name: str) -> RunArtifacts:
    """Create an experiment output directory and artifact paths.

    Args:
        base_dir: Base output directory.
        experiment_name: Name of the experiment run.

    Returns:
        RunArtifacts containing paths for local JSON artifacts.

    """
    output_dir = base_dir / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return RunArtifacts(
        output_dir=output_dir,
        results_path=output_dir / "results.jsonl",
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


def _load_dataset_examples(
    dataset_name: str,
    limit: int | None,
) -> list[Example]:
    """Load examples from a LangSmith dataset with an optional limit.

    Args:
        dataset_name: Name of the dataset in LangSmith.
        limit: Maximum number of examples to load, or ``None`` for all.

    Returns:
        List of dataset examples in stable order.

    """
    client = Client()
    examples = client.list_examples(dataset_name=dataset_name)
    loaded: list[Example] = []
    for example in examples:
        loaded.append(example)
        if limit is not None and len(loaded) >= limit:
            break
    return loaded


_ROOMS_TAGS: frozenset[str] = frozenset({"rooms", "mixed"})


def _has_rooms_cases(examples: Iterable[Example]) -> bool:
    """Return ``True`` if any example carries a rooms-path tag.

    Args:
        examples: Loaded LangSmith examples.

    Returns:
        Whether at least one example has a ``"rooms"`` or ``"mixed"`` tag.

    """
    for ex in examples:
        inputs = getattr(ex, "inputs", None) or {}
        tags = inputs.get("tags") or []
        if _ROOMS_TAGS.intersection(tags):
            return True
    return False


async def _setup_eval_schema(
    cfg: EvalConfig,
) -> tuple[AsyncConnectionPool[Any], str, Token[str | None]]:
    """Create a dedicated schema for the evaluation run.

    Opens a connection pool, generates a unique schema name, loads baseline
    room data into the new schema, and sets the ``EVAL_SCHEMA`` ContextVar so
    that the rooms agent routes SQL there.

    Args:
        cfg: The loaded evaluation configuration.

    Returns:
        A tuple of ``(pool, schema_name, token)`` for later teardown.

    """
    schema = generate_schema_name("eval")
    logger.info("Setting up eval schema: %s", schema)
    pool = await open_schema_pool(
        load_app_config().pgsql_db_url,
        max_size=cfg.reset_pool.max_size,
    )
    await create_case_schema(
        pool=pool,
        schema=schema,
        data_path=str(cfg.rooms.data_path),
    )
    token = set_eval_schema(schema)
    logger.info("Eval schema ready: %s", schema)
    return pool, schema, token


def _write_summary(path: Path, summary: Mapping[str, object]) -> None:
    """Write a summary JSON file to disk.

    Args:
        path: Output file path.
        summary: Summary payload to serialize.

    """
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, ensure_ascii=True, indent=2))


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    """Write an iterable of JSON-serializable rows to a JSONL file.

    Args:
        path: Output file path.
        rows: Iterable of mapping rows to serialize.

    """
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=True, default=json_safe) + "\n",
            )


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
                    turn.get("final_route")
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


def _init_metric_values() -> dict[str, list[float]]:
    """Initialize metric storage for summary aggregation.

    Returns:
        Mapping of metric names to collected float values.

    """
    return {
        "route_accuracy": [],
        "rooms_outcome_match_rate": [],
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
        if key == "rooms_outcome_match_rate" and score is not None:
            metric_values["rooms_outcome_match_rate"].append(score)
        if key == "info_reference_subset_pass_rate" and score is not None:
            metric_values["info_reference_subset_pass_rate"].append(score)
        if "expected_filters" in key and value is not None:
            metric_values["info_expected_filters_pass_rate"].append(value)


def _build_summary_base(
    context: SummaryContext,
    num_examples: int,
    num_failed_runs: int,
) -> dict[str, object]:
    """Build the base summary metadata.

    Args:
        context: Summary metadata for the experiment.
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


def _mean(values: list[float]) -> float | None:
    """Compute the arithmetic mean for a list of values.

    Args:
        values: List of numeric values.

    Returns:
        The mean value, or None if the list is empty.

    """
    return sum(values) / len(values) if values else None


def _mean_metrics(metric_values: Mapping[str, list[float]]) -> dict[str, object]:
    """Compute mean metrics, skipping missing values.

    Args:
        metric_values: Mapping of metrics to collected values.

    Returns:
        Mapping of mean metric names to values.

    """
    mean_values = {
        "mean_route_accuracy": _mean(metric_values["route_accuracy"]),
        "mean_rooms_outcome_match_rate": _mean(
            metric_values["rooms_outcome_match_rate"],
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


def _summarize_results(
    result_rows: Iterable[Mapping[str, object]],
    context: SummaryContext,
) -> dict[str, object]:
    """Aggregate summary statistics from evaluation rows.

    Args:
        result_rows: Iterable of evaluation result rows.
        context: Metadata for the summary output.

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


def _build_error_summary(
    context: SummaryContext,
    error: Exception,
) -> dict[str, object]:
    """Build a minimal summary when the experiment fails.

    Args:
        context: Summary metadata for the experiment.
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


async def main() -> None:
    """Run the LangSmith experiment and persist local artifacts."""
    cfg = load_eval_config()
    started_at = datetime.now(UTC)
    dataset_name = cfg.experiment.dataset_name
    experiment_prefix = cfg.experiment.experiment_prefix
    run_notes = cfg.experiment.run_notes
    experiment_name = _build_experiment_name(
        experiment_prefix,
        run_notes,
        started_at,
    )
    max_concurrency = cfg.experiment.max_concurrency
    output_root = cfg.experiment.output_dir
    log_dir = cfg.experiment.log_dir
    artifacts = _build_output_paths(output_root, experiment_name)
    upload_results = cfg.experiment.upload_results
    limit = cfg.experiment.limit

    _configure_logging(experiment_name, log_dir)

    # Setting LANGSMITH_TEST_CACHE enables caching of example runs to reduce cost/time.
    metadata = _build_metadata(cfg.metadata)

    evaluators = [
        eval_routing_accuracy,
        eval_injection_tripwires,
        eval_rooms_outcome_and_invariants,
        eval_llm_rubrics,
        eval_rag_metrics_info_turns,
        eval_info_reference_subset,
        eval_info_expected_filters,
    ]

    aevaluate_kwargs: MutableMapping[str, Any] = {
        "evaluators": evaluators,
        "max_concurrency": max_concurrency,
        "upload_results": upload_results,
        "metadata": metadata,
    }
    signature = inspect.signature(aevaluate)
    if "experiment_name" in signature.parameters:
        aevaluate_kwargs["experiment_name"] = experiment_name
    elif "experiment_prefix" in signature.parameters:
        aevaluate_kwargs["experiment_prefix"] = experiment_name

    examples = _load_dataset_examples(dataset_name, limit)
    has_rooms = _has_rooms_cases(examples)
    schema_resources = await _setup_eval_schema(cfg) if has_rooms else None

    try:
        results = await aevaluate(run_example, examples, **aevaluate_kwargs)
        results_list: list[Any] = []
        if results is not None:
            results_list.extend([item async for item in results])

        result_rows = [_build_results_row(result) for result in results_list]
        _write_jsonl(artifacts.results_path, result_rows)
        finished_at = datetime.now(UTC)
        context = SummaryContext(
            dataset_name=dataset_name,
            experiment_name=experiment_name,
            max_concurrency=max_concurrency,
            started_at=started_at,
            finished_at=finished_at,
            upload_results=upload_results,
        )
        summary = _summarize_results(result_rows, context)
        _write_summary(artifacts.summary_path, summary)
    except Exception as exc:
        finished_at = datetime.now(UTC)
        context = SummaryContext(
            dataset_name=dataset_name,
            experiment_name=experiment_name,
            max_concurrency=max_concurrency,
            started_at=started_at,
            finished_at=finished_at,
            upload_results=upload_results,
        )
        summary = _build_error_summary(context, exc)
        _write_summary(artifacts.summary_path, summary)
        raise
    finally:
        if schema_resources is not None:
            pool, schema, token = schema_resources
            drop_err = await teardown_schema(token, pool, schema)
            if drop_err:
                logger.warning("Schema teardown error: %s", drop_err)


if __name__ == "__main__":
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
