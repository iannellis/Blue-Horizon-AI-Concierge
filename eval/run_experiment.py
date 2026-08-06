"""Run LangSmith evaluations for the hotel agent dataset and persist artifacts."""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import logging
import os
import platform
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv
from langsmith import Client
from langsmith.evaluation import aevaluate

from blue_horizon.config import load_app_config
from eval._result_utils import (
    _build_error_summary,
    _build_results_row,
    _summarize_results,
    compute_latency_summary,
    format_latency_table,
)
from eval._utils import configure_logging, json_safe
from eval.booking_db_manager import reset_neon_branch
from eval.config import MetadataConfig, load_eval_config
from eval.evaluators import (
    eval_booking_outcome_and_invariants,
    eval_info_expected_filters,
    eval_info_reference_subset,
    eval_injection_tripwires,
    eval_llm_rubrics,
    eval_rag_metrics_info_turns,
    eval_routing_accuracy,
    eval_turn_latency,
)
from eval.langsmith_target import run_example

if TYPE_CHECKING:
    from collections.abc import Iterable, MutableMapping
    from pathlib import Path

    from langsmith.schemas import Example

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


def _format_timestamp(timestamp: datetime) -> str:
    """Format a timestamp for experiment naming.

    Args:
        timestamp: Timestamp to format.

    Returns:
        Timestamp string formatted as YYYYMMDD_HHMMSS.

    """
    return timestamp.strftime("%Y%m%d_%H%M%S")


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


_BOOKING_TAGS: frozenset[str] = frozenset({"booking", "mixed"})


def _has_booking_cases(examples: Iterable[Example]) -> bool:
    """Return ``True`` if any example carries a booking-path tag.

    Args:
        examples: Loaded LangSmith examples.

    Returns:
        Whether at least one example has a ``"booking"`` or ``"mixed"`` tag.

    """
    for ex in examples:
        inputs = getattr(ex, "inputs", None) or {}
        tags = inputs.get("tags") or []
        if _BOOKING_TAGS.intersection(tags):
            return True
        turns = inputs.get("turns") or []
        if not isinstance(turns, list):
            continue
        for turn in turns:
            if not isinstance(turn, Mapping):
                continue
            route = turn.get("expected_route")
            if isinstance(route, str) and route == "booking":
                return True
    return False


async def _prepare_eval_database(cfg: EvalConfig) -> None:
    """Redirect the DB URL to the eval database and reset the Neon branch.

    Applies any ``PGSQL_EVAL_DB_URL`` override so the booking agent connects to
    the eval database, then calls the Neon management API to restore the
    configured branch to its parent baseline.

    Args:
        cfg: The loaded evaluation configuration.

    Raises:
        RuntimeError: If the Neon branch reset fails.

    """
    _override_eval_db_url(cfg)
    logger.info(
        "Resetting Neon branch %r in project %r.",
        cfg.neon.branch_name,
        cfg.neon.project_id,
    )
    await reset_neon_branch(cfg.neon, api_key=cfg.neon_api_key)
    logger.info("Eval database ready (Neon branch reset complete).")


def _override_eval_db_url(cfg: EvalConfig) -> None:
    """Override ``PGSQL_DB_URL`` with ``PGSQL_EVAL_DB_URL`` when the latter is set.

    If ``PGSQL_EVAL_DB_URL`` is present in *cfg*, this function writes its
    value to ``PGSQL_DB_URL`` and clears the ``load_app_config`` LRU cache so
    the next ``load_app_config()`` call picks up the overridden URL.  This
    ensures the booking agent inside ``OrchestrationManager`` connects to the
    correct Neon branch database rather than the default application database.

    If ``PGSQL_EVAL_DB_URL`` is not set, this function is a no-op.

    Args:
        cfg: The loaded evaluation configuration.

    """
    eval_db_url = cfg.pgsql_eval_db_url
    if not eval_db_url:
        return
    os.environ["PGSQL_DB_URL"] = eval_db_url
    load_app_config.cache_clear()
    logger.info("PGSQL_EVAL_DB_URL override applied: PGSQL_DB_URL updated.")


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


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the evaluation run.

    Returns:
        Parsed argument namespace.  The ``config`` attribute holds an optional
        path string to a TOML configuration file, or ``None`` when omitted.

    """
    parser = argparse.ArgumentParser(
        description="Run LangSmith evaluations for the hotel agent dataset.",
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        default=None,
        help=(
            "Path to a TOML configuration file. "
            "Defaults to eval/eval_config.toml when omitted."
        ),
    )
    return parser.parse_args()


def _log_latency_table(latency: dict[str, object]) -> None:
    """Log per-route p50/p95/p99 latency quantiles as an INFO table.

    Args:
        latency: Dict returned by ``compute_latency_summary``, containing
            a ``"latency_quantiles_ms"`` key with per-route quantile data.

    """
    table = format_latency_table(latency)
    if table:
        logger.info(table)


def _build_aevaluate_kwargs(
    cfg: EvalConfig,
    experiment_name: str,
    metadata: dict[str, object],
    evaluators: list[Any],
) -> MutableMapping[str, Any]:
    """Build the keyword-argument dict for ``aevaluate``.

    Inspects the ``aevaluate`` signature to choose between the
    ``experiment_name`` and ``experiment_prefix`` parameter depending on the
    installed LangSmith version.

    Args:
        cfg: Loaded evaluation configuration.
        experiment_name: Formatted experiment name string.
        metadata: Metadata dict for LangSmith experiment tracking.
        evaluators: List of evaluator callables.

    Returns:
        A mapping of keyword arguments ready to unpack into ``aevaluate``.

    """
    kwargs: MutableMapping[str, Any] = {
        "evaluators": evaluators,
        "max_concurrency": cfg.experiment.max_concurrency,
        "upload_results": cfg.experiment.upload_results,
        "metadata": metadata,
    }
    signature = inspect.signature(aevaluate)
    if "experiment_name" in signature.parameters:
        kwargs["experiment_name"] = experiment_name
    elif "experiment_prefix" in signature.parameters:
        kwargs["experiment_prefix"] = experiment_name
    return kwargs


async def _run_and_write_results(
    cfg: EvalConfig,
    artifacts: RunArtifacts,
    examples: list[Any],
    aevaluate_kwargs: MutableMapping[str, Any],
    started_at: datetime,
) -> None:
    """Execute evaluation, write results JSONL, and write the summary JSON.

    Constructs ``SummaryContext`` with the actual ``finished_at`` timestamp in
    a ``finally`` block so the timestamp is always accurate regardless of
    whether the run succeeded or failed.  The experiment name is read back
    from ``aevaluate_kwargs`` (checking both the ``experiment_name`` and
    ``experiment_prefix`` keys for LangSmith version compatibility).

    Args:
        cfg: Loaded evaluation configuration (passed to ``run_example``).
        artifacts: Filesystem paths for the output files.
        examples: Dataset examples to evaluate.
        aevaluate_kwargs: Keyword arguments forwarded to ``aevaluate``.
        started_at: Timestamp when the experiment was started.

    Raises:
        Exception: Re-raises any exception thrown during evaluation or
            result collection after writing an error summary to disk.

    """
    result_rows: list[Any] = []
    caught_exc: BaseException | None = None
    try:
        target = partial(run_example, cfg=cfg)
        results = await aevaluate(target, examples, **aevaluate_kwargs)
        if results is not None:
            result_rows.extend([item async for item in results])
        result_rows = [_build_results_row(result) for result in result_rows]
        _write_jsonl(artifacts.results_path, result_rows)
    except Exception as exc:
        caught_exc = exc
        raise
    finally:
        exp_name = str(
            aevaluate_kwargs.get("experiment_name")
            or aevaluate_kwargs.get("experiment_prefix")
            or "",
        )
        context = SummaryContext(
            dataset_name=cfg.experiment.dataset_name,
            experiment_name=exp_name,
            max_concurrency=cfg.experiment.max_concurrency,
            started_at=started_at,
            finished_at=datetime.now(UTC),
            upload_results=cfg.experiment.upload_results,
        )
        if caught_exc is not None:
            summary: Mapping[str, object] = _build_error_summary(context, caught_exc)
        else:
            summary = _summarize_results(result_rows, context)
            latency = compute_latency_summary(artifacts.results_path)
            if latency:
                summary = {**summary, **latency}
                _log_latency_table(latency)
        _write_summary(artifacts.summary_path, summary)


async def main() -> None:
    """Run the LangSmith experiment and persist local artifacts."""
    args = _parse_args()
    cfg = load_eval_config(path=args.config)
    started_at = datetime.now(UTC)
    dataset_name = cfg.experiment.dataset_name
    experiment_name = _build_experiment_name(
        cfg.experiment.experiment_prefix,
        cfg.experiment.run_notes,
        started_at,
    )
    artifacts = _build_output_paths(cfg.experiment.output_dir, experiment_name)

    configure_logging(experiment_name, cfg.experiment.log_dir)

    # Setting LANGSMITH_TEST_CACHE enables caching of example runs to reduce cost/time.
    metadata = _build_metadata(cfg.metadata)

    evaluators = [
        eval_routing_accuracy,
        partial(eval_injection_tripwires, cfg=cfg),
        partial(eval_booking_outcome_and_invariants, cfg=cfg),
        partial(eval_llm_rubrics, cfg=cfg),
        partial(eval_rag_metrics_info_turns, cfg=cfg),
        partial(eval_info_reference_subset, cfg=cfg),
        partial(eval_info_expected_filters, cfg=cfg),
        eval_turn_latency,
    ]
    aevaluate_kwargs = _build_aevaluate_kwargs(
        cfg, experiment_name, metadata, evaluators,
    )

    examples = _load_dataset_examples(dataset_name, cfg.experiment.limit)
    if _has_booking_cases(examples):
        await _prepare_eval_database(cfg)

    await _run_and_write_results(cfg, artifacts, examples, aevaluate_kwargs, started_at)


if __name__ == "__main__":
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
