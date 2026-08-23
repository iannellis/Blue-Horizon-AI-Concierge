"""Run LangSmith evaluations for the hotel agent dataset and persist artifacts."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
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
from typing import TYPE_CHECKING, Any, cast

from dotenv import load_dotenv
from langsmith import Client
from langsmith import env as ls_env
from langsmith.evaluation import aevaluate

from blue_horizon.agents.prompt_utils import load_packaged_text
from blue_horizon.config import app_config_source_text, load_app_config
from eval._result_utils import (
    _build_error_summary,
    _build_results_row,
    _summarize_results,
    compute_latency_summary,
    format_latency_table,
)
from eval._utils import configure_logging, json_safe
from eval.booking_db_manager import reset_neon_branch
from eval.config import eval_config_source_text, load_eval_config
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

    from langsmith.schemas import Example, Run

    from blue_horizon.config import AppConfig
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
        metadata: The same metadata attached to the LangSmith experiment
            (git commit, models, config fingerprints), mirrored into
            ``summary.json`` so it is available locally even when
            ``upload_results`` is ``False``.

    """

    dataset_name: str
    experiment_name: str
    max_concurrency: int
    started_at: datetime
    finished_at: datetime
    upload_results: bool
    metadata: Mapping[str, object]


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


def _resolve_run_notes(
    configured_notes: str | None,
    case_ids: list[str] | None,
) -> str | None:
    """Replace the configured run notes with a subset label when case-filtered.

    The configured ``run_notes`` (e.g. "Full dataset with 206 cases") describes
    the full dataset and becomes misleading in the experiment name and output
    directory once ``--case-id`` restricts the run to a handful of examples.
    When a case-id filter is active, this builds a notes string derived from the
    requested case IDs instead, so downstream naming makes the subset explicit
    rather than claiming to be the full run.

    Args:
        configured_notes: The ``run_notes`` value from the TOML config.
        case_ids: Requested case_id filter, or ``None`` to run the full dataset.

    Returns:
        ``configured_notes`` unchanged when no case-id filter is active,
        otherwise a "subset_..." label built from ``case_ids``.

    """
    if case_ids is None:
        return configured_notes
    if len(case_ids) == 1:
        return f"subset_{case_ids[0]}"
    shown = "_".join(case_ids[:2])
    return f"subset_{len(case_ids)}cases_{shown}"


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


def _build_metadata(
    cfg: EvalConfig,
    app_cfg: AppConfig,
    eval_config_path: str | None,
    *,
    upload_results: bool,
) -> dict[str, object]:
    """Build the metadata dictionary attached to the LangSmith experiment.

    Combines everything needed to answer "what exactly produced these
    scores": the commit and dirty-tree state, the dataset/concurrency shape,
    every model (and reasoning effort) in the call path, and content
    fingerprints of the config/prompt files. LangSmith only auto-attaches
    the commit -- none of the rest is visible without this.

    Args:
        cfg: Loaded evaluation configuration.
        app_cfg: Loaded application configuration.
        eval_config_path: The ``--config`` path passed on the command line,
            or ``None`` when the packaged default was used.
        upload_results: The effective upload-results setting for this run
            (``cfg.experiment.upload_results`` unless overridden by
            ``--no-upload``), recorded here so the local summary reflects
            what actually happened rather than only what was configured.

    Returns:
        Metadata payload for LangSmith experiment tracking and the local
        summary.json, with empty/``None`` values dropped.

    """
    metadata: dict[str, object] = {
        **_git_metadata(),
        **_experiment_metadata(cfg, upload_results=upload_results),
        **_model_metadata(app_cfg, cfg),
        **_config_fingerprint_metadata(app_cfg, eval_config_path),
    }
    return _drop_empty(metadata)


def _git_metadata() -> dict[str, object]:
    """Capture the repository commit and dirty-tree state for the run.

    Uses ``langsmith.env.get_git_info``, the same helper LangSmith itself
    calls when auto-attaching a ``git`` block to the experiment -- so
    ``git_sha`` always matches the commit LangSmith's UI links to. Also
    mirrors it into ``summary.json`` locally, which LangSmith's auto-attached
    block never reaches. Note this is always ``HEAD``: with a dirty working
    tree, the linked commit is not the code that actually ran, which is why
    ``git_dirty`` is recorded alongside it.

    Returns:
        Mapping with ``git_sha`` (full commit SHA) and ``git_dirty``.

    """
    git_info = ls_env.get_git_info()
    return {
        "git_sha": git_info.get("commit"),
        "git_dirty": git_info.get("dirty"),
    }


def _experiment_metadata(
    cfg: EvalConfig,
    *,
    upload_results: bool,
) -> dict[str, object]:
    """Capture dataset identity and run-shape settings for the experiment.

    Args:
        cfg: Loaded evaluation configuration.
        upload_results: The effective upload-results setting for this run.

    Returns:
        Mapping describing which dataset ran, at what concurrency, and
        whether results were uploaded.

    """
    return {
        "dataset_name": cfg.experiment.dataset_name,
        "dataset_limit": cfg.experiment.limit,
        "max_concurrency": cfg.experiment.max_concurrency,
        "upload_results": upload_results,
    }


def _model_metadata(app_cfg: AppConfig, cfg: EvalConfig) -> dict[str, object]:
    """Capture every LLM/embedding model actually used by the run.

    Includes the reasoning-effort hint for the three OpenAI chat models
    (router, info, booking); the judge and Ragas models are Gemini-backed,
    and this codebase has no reasoning-effort knob for them.

    Args:
        app_cfg: Loaded application configuration (router/info/booking models).
        cfg: Loaded evaluation configuration (judge/Ragas models).

    Returns:
        Mapping of model-identity fields.

    """
    return {
        "router_model": app_cfg.orchestration.llm.model,
        "router_reasoning_effort": app_cfg.orchestration.llm.reasoning_effort,
        "info_model": app_cfg.info.llm.model,
        "info_reasoning_effort": app_cfg.info.llm.reasoning_effort,
        "info_embeddings_model": app_cfg.info.embeddings.model,
        "booking_model": app_cfg.booking.llm.model,
        "booking_reasoning_effort": app_cfg.booking.llm.reasoning_effort,
        "judge_model": cfg.judge.model,
        "ragas_llm_model": cfg.ragas.llm_model,
        "ragas_embedding_model": cfg.ragas.embedding_model,
    }


def _config_fingerprint_metadata(
    app_cfg: AppConfig,
    eval_config_path: str | None,
) -> dict[str, object]:
    """Fingerprint the config and prompt files that shaped this run.

    ``git_sha`` alone does not prove what actually ran: a dirty working tree
    or a ``--config`` override can change behavior without changing the
    commit. Hashing the resolved file contents catches both.

    Args:
        app_cfg: Loaded application configuration, used to resolve which
            prompt files were actually read.
        eval_config_path: The ``--config`` path passed on the command line,
            or ``None`` when the packaged default was used.

    Returns:
        Mapping of short content-hash fingerprints.

    """
    return {
        "app_config_sha256": _hash_text(app_config_source_text()),
        "eval_config_sha256": _hash_text(eval_config_source_text(eval_config_path)),
        **_prompt_fingerprint_metadata(app_cfg),
    }


def _prompt_fingerprint_metadata(app_cfg: AppConfig) -> dict[str, object]:
    """Fingerprint each system/parser prompt actually used by the agents.

    Hashed separately per prompt, rather than one combined hash, so a diff
    between two runs' metadata points at which specific prompt changed
    instead of just "something in the prompts changed".

    Args:
        app_cfg: Loaded application configuration.

    Returns:
        Mapping of per-prompt short content-hash fingerprints.

    """
    orchestration_prompts = app_cfg.orchestration.prompts
    info_prompts = app_cfg.info.prompts
    booking_prompts = app_cfg.booking.prompts
    return {
        "router_prompt_sha256": _hash_text(
            load_packaged_text(
                _prompt_resource_path(
                    orchestration_prompts.folder,
                    orchestration_prompts.orchestration_prompt_filename,
                ),
            ),
        ),
        "info_system_prompt_sha256": _hash_text(
            load_packaged_text(
                _prompt_resource_path(
                    info_prompts.folder,
                    info_prompts.system_prompt_filename,
                ),
            ),
        ),
        "info_parser_prompt_sha256": _hash_text(
            load_packaged_text(
                _prompt_resource_path(
                    info_prompts.folder,
                    info_prompts.parser_prompt_filename,
                ),
            ),
        ),
        "booking_prompt_sha256": _hash_text(
            load_packaged_text(
                _prompt_resource_path(
                    booking_prompts.folder,
                    booking_prompts.system_prompt_filename,
                ),
            ),
        ),
    }


def _prompt_resource_path(folder: str, filename: str) -> str:
    """Build a packaged-resource path for a prompt file.

    Mirrors the folder/filename join each agent's resource loader performs
    internally, so the hashed text matches what the agent actually reads.

    Args:
        folder: Prompt folder relative to the ``blue_horizon`` package root.
        filename: Prompt filename within that folder.

    Returns:
        The resource path, or bare ``filename`` when ``folder`` is empty.

    """
    stripped_folder = folder.strip("/")
    if stripped_folder:
        return f"{stripped_folder}/{filename}"
    return filename


def _hash_text(text: str) -> str:
    """Compute a short content fingerprint for a text blob.

    Args:
        text: Text to hash.

    Returns:
        The first 12 hex characters of the text's SHA-256 digest.

    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _drop_empty(metadata: dict[str, object]) -> dict[str, object]:
    """Drop ``None``/empty-string values from a metadata mapping.

    Keeps falsy-but-meaningful values (``False``, ``0``) that a plain
    truthiness filter would silently discard.

    Args:
        metadata: Metadata mapping to filter.

    Returns:
        Filtered mapping containing only meaningfully-set values.

    """
    return {
        key: value
        for key, value in metadata.items()
        if value is not None and value != ""
    }


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


def _filter_examples_by_case_id(
    examples: list[Example],
    case_ids: list[str] | None,
) -> list[Example]:
    """Restrict examples to the requested ``case_id`` values, if any.

    Args:
        examples: Loaded LangSmith examples.
        case_ids: Case IDs to keep, or ``None`` to keep every example.

    Returns:
        The filtered list of examples, in their original relative order.

    Raises:
        ValueError: If any requested ``case_id`` matches no loaded example.

    """
    if case_ids is None:
        return examples
    wanted = set(case_ids)
    matched = [
        ex for ex in examples if (ex.inputs or {}).get("case_id") in wanted
    ]
    found = {(ex.inputs or {}).get("case_id") for ex in matched}
    missing = wanted - found
    if missing:
        msg = f"No dataset example(s) found for case_id(s): {sorted(missing)}"
        raise ValueError(msg)
    return matched


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


async def _reset_eval_database(cfg: EvalConfig) -> None:
    """Reset the eval database's Neon branch to its parent baseline.

    The ``PGSQL_RW_EVAL_DB_URL``/``PGSQL_RO_EVAL_DB_URL`` override is applied
    separately, unconditionally, near the top of ``main()`` -- every
    ``load_app_config()`` call needs it, not just booking-case runs, so it
    can't wait until here (this function only runs when
    ``_has_booking_cases()`` is true).

    Args:
        cfg: The loaded evaluation configuration.

    Raises:
        RuntimeError: If the Neon branch reset fails.

    """
    logger.info(
        "Resetting Neon branch %r in project %r.",
        cfg.neon.branch_name,
        cfg.neon.project_id,
    )
    await reset_neon_branch(cfg.neon, api_key=cfg.neon_api_key)
    logger.info("Eval database ready (Neon branch reset complete).")


def _override_eval_db_url(cfg: EvalConfig) -> None:
    """Override ``PGSQL_RW_DB_URL``/``PGSQL_RO_DB_URL`` with the eval overrides.

    If ``PGSQL_RW_EVAL_DB_URL`` is present in *cfg*, this function writes its
    value to ``PGSQL_RW_DB_URL``; if ``PGSQL_RO_EVAL_DB_URL`` is present, its
    value is written to ``PGSQL_RO_DB_URL``. When either override is applied,
    the ``load_app_config`` LRU cache is cleared so the next
    ``load_app_config()`` call picks up the overridden URLs. This ensures the
    booking agent inside ``OrchestrationManager`` connects to the correct
    Neon branch database (write pool and read-only pool alike) rather than
    the default application database.

    Setting only the read-write override and leaving the read-only override
    unset means ``run_sql`` keeps using ``PGSQL_RO_DB_URL`` as configured
    outside the eval run, which may point at a different database than the
    one being reset -- so the two are logged together to make a partial
    override obvious.

    If neither ``PGSQL_RW_EVAL_DB_URL`` nor ``PGSQL_RO_EVAL_DB_URL`` is set,
    this function is a no-op.

    Args:
        cfg: The loaded evaluation configuration.

    """
    eval_db_url = cfg.pgsql_rw_eval_db_url
    eval_ro_db_url = cfg.pgsql_ro_eval_db_url
    if not eval_db_url and not eval_ro_db_url:
        return
    if eval_db_url:
        os.environ["PGSQL_RW_DB_URL"] = eval_db_url
    if eval_ro_db_url:
        os.environ["PGSQL_RO_DB_URL"] = eval_ro_db_url
    load_app_config.cache_clear()
    logger.info(
        "Eval DB URL override applied: PGSQL_RW_DB_URL updated=%s, "
        "PGSQL_RO_DB_URL updated=%s.",
        bool(eval_db_url),
        bool(eval_ro_db_url),
    )


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
    parser.add_argument(
        "--case-id",
        metavar="CASE_ID",
        action="append",
        default=None,
        help=(
            "Restrict the run to one dataset case_id (e.g. case_0117). "
            "May be passed multiple times to run several specific cases. "
            "Defaults to the full configured dataset when omitted."
        ),
    )
    parser.add_argument(
        "--router-only",
        action="store_true",
        help=(
            "Run only the routing-accuracy evaluator, skipping every other "
            "evaluator (including the LLM judge and Ragas metrics)."
        ),
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help=(
            "Run entirely locally: bypass aevaluate's LangSmith tracing "
            "altogether, so no run traces (and no results) reach LangSmith. "
            "aevaluate's own upload_results=False still traces every target "
            "run due to a hardcoded tracing_context(enabled=True) in the "
            "installed langsmith SDK, so this flag skips aevaluate for the "
            "target/evaluator execution rather than relying on that "
            "parameter. Local results.jsonl/summary.json are written as "
            "usual; the dataset examples are still read from LangSmith "
            "(a lightweight read, not a trace)."
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
    *,
    upload_results: bool,
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
        upload_results: The effective upload-results setting for this run
            (``cfg.experiment.upload_results`` unless overridden by
            ``--no-upload``).

    Returns:
        A mapping of keyword arguments ready to unpack into ``aevaluate``.

    """
    kwargs: MutableMapping[str, Any] = {
        "evaluators": evaluators,
        "max_concurrency": cfg.experiment.max_concurrency,
        "upload_results": upload_results,
        "metadata": metadata,
    }
    signature = inspect.signature(aevaluate)
    if "experiment_name" in signature.parameters:
        kwargs["experiment_name"] = experiment_name
    elif "experiment_prefix" in signature.parameters:
        kwargs["experiment_prefix"] = experiment_name
    return kwargs


@dataclass(frozen=True)
class _LocalRun:
    """Minimal local stand-in for a LangSmith ``Run``, holding only outputs.

    Evaluators only ever read ``run.outputs`` (see
    ``eval.evaluators._common``), so this needs no other fields. Defines
    ``__getattr__`` purely so ``eval._result_utils``'s duck-typing (which
    accepts a ``Mapping`` or an object exposing ``__getattr__``) treats this
    the same way it treats a real LangSmith ``Run``.

    Attributes:
        outputs: The target function's output payload for one example.

    """

    outputs: dict[str, object]

    def __getattr__(self, name: str) -> object:
        """Raise ``AttributeError`` for any field other than ``outputs``.

        Args:
            name: Attribute name being accessed.

        Raises:
            AttributeError: Always, for any name reaching this fallback.

        """
        msg = f"{type(self).__name__!r} object has no attribute {name!r}"
        raise AttributeError(msg)


async def _run_locally(
    cfg: EvalConfig,
    examples: list[Example],
    evaluators: list[Any],
) -> list[dict[str, object]]:
    """Run the target and evaluators for every example with no LangSmith tracing.

    Unlike ``aevaluate(..., upload_results=False)`` -- which still traces
    every target run because the installed LangSmith SDK's ``_aforward``
    helper hardcodes ``tracing_context(enabled=True)`` regardless of that
    parameter -- this calls ``run_example`` directly with no tracing wrapper
    at all, so zero requests reach LangSmith's trace-ingestion endpoint.

    Args:
        cfg: Loaded evaluation configuration.
        examples: Dataset examples to evaluate.
        evaluators: Evaluator callables (a mix of sync and async).

    Returns:
        Result-row dicts in the same shape ``_build_results_row`` produces
        for `aevaluate`-backed runs.

    """
    semaphore = asyncio.Semaphore(cfg.experiment.max_concurrency)

    async def _run_one(example: Example) -> dict[str, object]:
        async with semaphore:
            return await _run_and_score_example_locally(cfg, example, evaluators)

    return list(await asyncio.gather(*(_run_one(example) for example in examples)))


async def _run_and_score_example_locally(
    cfg: EvalConfig,
    example: Example,
    evaluators: list[Any],
) -> dict[str, object]:
    """Run one dataset example and score it, entirely outside LangSmith tracing.

    Args:
        cfg: Loaded evaluation configuration.
        example: Dataset example to run.
        evaluators: Evaluator callables (a mix of sync and async).

    Returns:
        A result-row dict built via ``_build_results_row``.

    """
    error: str | None = None
    run_output: dict[str, object] = {}
    try:
        run_output = await run_example(dict(example.inputs or {}), cfg=cfg)
    except Exception as exc:  # captured as a per-example error, not raised
        error = str(exc)
        logger.exception("Error running example %r locally.", example.id)

    local_run = _LocalRun(outputs=run_output)
    feedback: list[dict[str, object]] = []
    if error is None:
        # Evaluators only read `.outputs`, so `_LocalRun` satisfies `Run`
        # structurally; the cast documents that deliberate stand-in.
        feedback = await _score_example_locally(
            cast("Run", local_run), example, evaluators,
        )

    result: dict[str, object] = {
        "example": example,
        "run": local_run,
        "feedback": feedback,
        "error": error,
    }
    return _build_results_row(result)


async def _score_example_locally(
    run: Run,
    example: Example,
    evaluators: list[Any],
) -> list[dict[str, object]]:
    """Apply every evaluator to one run/example pair and flatten their results.

    Args:
        run: Local ``Run`` stand-in for the completed target execution.
        example: Dataset example being scored.
        evaluators: Evaluator callables (a mix of sync and async).

    Returns:
        Flat list combining every evaluator's per-metric feedback dicts.

    """
    feedback: list[dict[str, object]] = []
    for evaluator in evaluators:
        response = evaluator(run, example)
        if inspect.isawaitable(response):
            response = await response
        if isinstance(response, list):
            feedback.extend(response)
    return feedback


async def _run_and_write_results(  # noqa: PLR0913
    cfg: EvalConfig,
    artifacts: RunArtifacts,
    examples: list[Example],
    aevaluate_kwargs: MutableMapping[str, Any],
    started_at: datetime,
    *,
    local: bool,
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
        aevaluate_kwargs: Keyword arguments forwarded to ``aevaluate`` (also
            supplies ``evaluators``/``experiment_name``/``metadata`` for the
            local path, so the summary looks the same either way).
        started_at: Timestamp when the experiment was started.
        local: When ``True``, bypass ``aevaluate`` entirely via
            ``_run_locally`` so no LangSmith trace requests are made.

    Raises:
        Exception: Re-raises any exception thrown during evaluation or
            result collection after writing an error summary to disk.

    """
    result_rows: list[Any] = []
    caught_exc: BaseException | None = None
    try:
        if local:
            evaluators = cast("list[Any]", aevaluate_kwargs["evaluators"])
            result_rows = await _run_locally(cfg, examples, evaluators)
        else:
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
        run_metadata = aevaluate_kwargs.get("metadata") or {}
        context = SummaryContext(
            dataset_name=cfg.experiment.dataset_name,
            experiment_name=exp_name,
            max_concurrency=cfg.experiment.max_concurrency,
            started_at=started_at,
            finished_at=datetime.now(UTC),
            upload_results=bool(aevaluate_kwargs.get("upload_results")),
            metadata=run_metadata,
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


def _build_evaluators(cfg: EvalConfig, *, router_only: bool) -> list[Any]:
    """Build the list of evaluators to pass to ``aevaluate``.

    Args:
        cfg: Loaded evaluation configuration, forwarded to evaluators that
            need it.
        router_only: When ``True``, return only the routing-accuracy
            evaluator, skipping every other evaluator (including the LLM
            judge and Ragas metrics).

    Returns:
        List of evaluator callables ready to pass to ``aevaluate``.

    """
    if router_only:
        return [eval_routing_accuracy]
    return [
        eval_routing_accuracy,
        partial(eval_injection_tripwires, cfg=cfg),
        partial(eval_booking_outcome_and_invariants, cfg=cfg),
        partial(eval_llm_rubrics, cfg=cfg),
        partial(eval_rag_metrics_info_turns, cfg=cfg),
        partial(eval_info_reference_subset, cfg=cfg),
        partial(eval_info_expected_filters, cfg=cfg),
        eval_turn_latency,
    ]


async def main() -> None:
    """Run the LangSmith experiment and persist local artifacts."""
    args = _parse_args()
    cfg = load_eval_config(path=args.config)
    _override_eval_db_url(cfg)
    app_cfg = load_app_config()
    started_at = datetime.now(UTC)
    dataset_name = cfg.experiment.dataset_name
    experiment_name = _build_experiment_name(
        cfg.experiment.experiment_prefix,
        _resolve_run_notes(cfg.experiment.run_notes, args.case_id),
        started_at,
    )
    artifacts = _build_output_paths(cfg.experiment.output_dir, experiment_name)

    configure_logging(experiment_name, cfg.experiment.log_dir)

    upload_results = cfg.experiment.upload_results and not args.no_upload

    # Setting LANGSMITH_TEST_CACHE enables caching of example runs to reduce cost/time.
    metadata = _build_metadata(
        cfg, app_cfg, args.config, upload_results=upload_results,
    )

    evaluators = _build_evaluators(cfg, router_only=args.router_only)
    aevaluate_kwargs = _build_aevaluate_kwargs(
        cfg, experiment_name, metadata, evaluators, upload_results=upload_results,
    )

    examples = _load_dataset_examples(dataset_name, cfg.experiment.limit)
    examples = _filter_examples_by_case_id(examples, args.case_id)
    if _has_booking_cases(examples):
        await _reset_eval_database(cfg)

    await _run_and_write_results(
        cfg, artifacts, examples, aevaluate_kwargs, started_at, local=args.no_upload,
    )


if __name__ == "__main__":
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
