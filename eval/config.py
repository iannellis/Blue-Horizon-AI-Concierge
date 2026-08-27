"""Evaluation configuration parsing for Blue Horizon.

By default, loads eval_config.toml from the eval package directory.
"""

from __future__ import annotations

import tomllib
from datetime import date  # noqa: TC003
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import BeforeValidator, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from blue_horizon.config import (
    FrozenModel,
    NeonConfig,
    NonNegInt,
    PositiveInt,
    read_packaged_toml,
)

BASE_PACKAGE = "eval"
_EVAL_CONFIG_RESOURCE = "eval_config.toml"
_STRESS_CONFIG_RESOURCE = "stress_config.toml"
_BASE_DIR = Path(__file__).resolve().parents[1]


def _resolve_output_paths(value: object) -> Path:
    """Resolve a configured output/log path relative to the repository root.

    Shared `field_validator` body for `ExperimentConfig` and
    `StressOutputConfig`, both of which resolve `output_dir`/`log_dir`
    the same way.

    Args:
        value: Raw config value representing a path.

    Returns:
        Resolved Path instance.

    Raises:
        TypeError: If the value cannot be interpreted as a path.

    """
    return _resolve_path(value, base_dir=_BASE_DIR)


class _EvalDbSettings(BaseSettings):
    """Shared read-write/read-only eval database URL overrides.

    Both `EvalConfig` and `StressEvalConfig` declare the same pair of
    optional Postgres URL overrides (plus the Neon API key), used to point
    the booking agent's pools at an eval-specific database branch instead of
    the application's default `PGSQL_RW_DB_URL`/`PGSQL_RO_DB_URL`.

    Attributes:
        pgsql_rw_eval_db_url: Read-write PostgreSQL database URL override
            (`bh_agent_rw`). When set, takes precedence over
            ``PGSQL_RW_DB_URL`` for the booking agent's write pool,
            propose/write tools, and evaluator pool connections.
        pgsql_ro_eval_db_url: Read-only PostgreSQL database URL override
            (`bh_agent_ro`), used exclusively by `run_sql`. When set, takes
            precedence over ``PGSQL_RO_DB_URL``. Must be set alongside
            `pgsql_rw_eval_db_url` and point at the same database -- leaving
            it unset while overriding the read-write URL runs the model's
            searches against the write role, defeating the point of the
            split.
        neon_api_key: API key for authenticating with the Neon management
            API, required when resetting a Neon branch before a run.

    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    pgsql_rw_eval_db_url: str | None = Field(
        default=None,
        validation_alias="PGSQL_RW_EVAL_DB_URL",
    )
    pgsql_ro_eval_db_url: str | None = Field(
        default=None,
        validation_alias="PGSQL_RO_EVAL_DB_URL",
    )
    neon_api_key: str | None = Field(
        default=None,
        validation_alias="NEON_API_KEY",
    )


class ExperimentConfig(FrozenModel):
    """Configuration for LangSmith experiment execution.

    Attributes:
        dataset_name: LangSmith dataset name to evaluate.
        experiment_prefix: Prefix used when generating experiment names.
        run_notes: Optional run notes appended to experiment names.
        max_concurrency: Maximum concurrent evaluations.
        output_dir: Base directory for local artifacts.
        log_dir: Directory for evaluation logs.
        upload_results: Whether to upload results to LangSmith.
        limit: Optional limit on the number of dataset examples to run.

    """

    dataset_name: str
    experiment_prefix: str
    run_notes: str | None = None
    max_concurrency: PositiveInt
    output_dir: Path
    log_dir: Path
    upload_results: bool
    limit: int | None = None

    @field_validator("run_notes", "limit", mode="before")
    @classmethod
    def _empty_str_to_none(cls, value: object) -> object:
        """Convert empty strings to None for optional fields."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    _normalize_output_paths = field_validator(
        "output_dir", "log_dir", mode="before",
    )(_resolve_output_paths)


class EvaluatorLimitsConfig(FrozenModel):
    """Limits controlling judge LLM inputs and stored evaluator output sizes.

    Attributes:
        context_max_chars: Maximum characters per context snippet passed to the
            judge LLM.
        context_max_items: Maximum number of context snippets passed to the
            judge LLM per turn.
        assistant_max_chars: Maximum characters from assistant text passed to
            the judge LLM.
        user_max_chars: Maximum characters from user text passed to the judge
            LLM.
        info_filter_failures_max: Max stored failures for info filter checks.
        required_tool_failures_max: Max stored failures for required tool checks.
        json_value_max: Maximum characters for JSON-encoded evaluator values.
        rag_per_turn_json_max: Maximum characters for per-turn RAG JSON values.
        tripwire_hits_max: Maximum tripwire hits to store.

    """

    context_max_chars: PositiveInt
    context_max_items: PositiveInt
    assistant_max_chars: PositiveInt
    user_max_chars: PositiveInt
    info_filter_failures_max: PositiveInt
    required_tool_failures_max: PositiveInt
    json_value_max: PositiveInt
    rag_per_turn_json_max: PositiveInt
    tripwire_hits_max: PositiveInt


class OrchestrationConfig(FrozenModel):
    """Configuration for orchestration startup timing.

    Attributes:
        ready_timeout_s: Timeout (seconds) to wait for orchestration readiness.

    """

    ready_timeout_s: Annotated[
        float, BeforeValidator(lambda v: max(0.1, v)), Field(ge=0.1),
    ]


class JudgeConfig(FrozenModel):
    """Configuration for the LLM-as-judge evaluator.

    Attributes:
        model: Judge model name.
        info_cards_max: Maximum number of amenity/service cards the information
            agent is instructed to present per response. The judge uses this to
            avoid penalising the agent for not listing every valid option.

    """

    model: str
    info_cards_max: PositiveInt


class RagasConfig(FrozenModel):
    """Configuration for Ragas metrics evaluation.

    Attributes:
        turns_max: Maximum number of turns to score.
        contexts_max: Maximum retrieved contexts per turn.
        context_chars: Maximum characters per context.
        query_chars: Maximum characters for the user query.
        response_chars: Maximum characters for the assistant response.
        reference_chars: Maximum characters for the reference answer.
        llm_model: Model name for the Ragas LLM backend.
        llm_max_tokens: Maximum output tokens for the Ragas LLM.
        embedding_model: Model name for the Ragas embeddings backend.
        custom_precision_prompt: Whether to replace the stock Ragas
            context-precision prompt with the hotel-tuned variant that scores
            multi-part questions clause by clause.
        no_match_reference: Reference answer text marking a turn whose expected
            answer is "nothing matched". Context precision is skipped for turns
            whose reference consists solely of this sentinel. Empty disables
            the skip.

    """

    turns_max: NonNegInt
    contexts_max: NonNegInt
    context_chars: NonNegInt
    query_chars: NonNegInt
    response_chars: NonNegInt
    reference_chars: NonNegInt
    llm_model: str
    llm_max_tokens: PositiveInt
    embedding_model: str
    custom_precision_prompt: bool = False
    no_match_reference: str = ""


class StressWorkloadConfig(FrozenModel):
    """Workload and operation-mix parameters for stress-test runs.

    Attributes:
        users: Number of concurrent simulated users.
        ops_per_user: Number of operations per user.
        max_concurrency: Maximum concurrent users.
        book_weight: Relative weight for BOOK operations in the op-type mix.
        modify_weight: Relative weight for MODIFY operations in the op-type mix.
        cancel_weight: Relative weight for CANCEL operations in the op-type mix.

    """

    users: PositiveInt
    ops_per_user: PositiveInt
    max_concurrency: PositiveInt
    book_weight: Annotated[float, Field(ge=0.0)] = 0.5
    modify_weight: Annotated[float, Field(ge=0.0)] = 0.25
    cancel_weight: Annotated[float, Field(ge=0.0)] = 0.25


class StressTargetsConfig(FrozenModel):
    """Availability target discovery parameters for stress-test runs.

    Attributes:
        stay_nights: Nights per booking in generated targets.
        num_targets: Number of available targets to precompute.
        hot_target_count: Hot contention target subset size.
        hot_target_probability: Probability of selecting a hot target vs any target.
        start_date: Search start date for availability targets.
        horizon_days: Search horizon in days from start_date.

    """

    stay_nights: PositiveInt
    num_targets: PositiveInt
    hot_target_count: NonNegInt
    hot_target_probability: Annotated[float, Field(ge=0.0, le=1.0)] = 0.8
    start_date: date
    horizon_days: PositiveInt


class StressDbConfig(FrozenModel):
    """Database connection and retry parameters for stress-test runs.

    Attributes:
        pool_max: Maximum size of the database connection pool.
        db_retry_attempts: Retry attempts on transient DB connection errors.
        db_retry_delay_s: Base delay in seconds between DB retry attempts.
        reconcile_max_detail: Maximum entries in reconciliation failure detail lists.

    """

    pool_max: PositiveInt
    db_retry_attempts: PositiveInt = 3
    db_retry_delay_s: Annotated[float, Field(ge=0.0)] = 2.0
    reconcile_max_detail: PositiveInt = 10


class StressOutputConfig(FrozenModel):
    """Output path configuration for stress-test runs.

    Attributes:
        output_dir: Base output directory for stress artifacts.
        log_dir: Directory for stress run logs.

    """

    output_dir: Path
    log_dir: Path

    _normalize_output_paths = field_validator(
        "output_dir", "log_dir", mode="before",
    )(_resolve_output_paths)


class StressConfig(FrozenModel):
    """Configuration for stress-test runs, composed from sub-sections.

    Attributes:
        workload: Workload and operation-mix parameters.
        targets: Availability target discovery parameters.
        db: Database connection and retry parameters.
        output: Output path configuration.

    """

    workload: StressWorkloadConfig
    targets: StressTargetsConfig
    db: StressDbConfig
    output: StressOutputConfig


class EvalConfig(_EvalDbSettings):
    """Parsed evaluation configuration container.

    TOML configuration is loaded via ``load_eval_config()``.  Environment
    variables are read from the process environment or a ``.env`` file.

    Attributes:
        experiment: Experiment execution settings.
        evaluator_limits: Limits used by evaluators and summaries.
        neon: Neon branch reset configuration for eval runs.
        orchestration: Orchestration readiness timing.
        judge: Judge model configuration.
        ragas: Ragas scoring configuration.

    """

    experiment: ExperimentConfig
    evaluator_limits: EvaluatorLimitsConfig
    neon: NeonConfig
    orchestration: OrchestrationConfig
    judge: JudgeConfig
    ragas: RagasConfig


class StressEvalConfig(_EvalDbSettings):
    """Parsed stress-test configuration container.

    TOML configuration is loaded via ``load_stress_config()``.  Environment
    variables are read from the process environment or a ``.env`` file.

    Attributes:
        neon: Neon branch reset configuration for stress runs.
        orchestration: Orchestration readiness timing.
        stress: Stress-test workload, target, and database parameters.

    """

    neon: NeonConfig
    orchestration: OrchestrationConfig
    stress: StressConfig


def _load_toml(path: Path | str | None, resource_name: str) -> dict[str, object]:
    """Load a TOML file from a path or from the packaged eval resources.

    Args:
        path: Explicit path to a TOML file, or ``None`` to load the named
            packaged resource from the ``eval`` package.
        resource_name: Filename of the packaged resource to use when
            ``path`` is ``None``.

    Returns:
        Parsed TOML data as a plain dict.

    Raises:
        RuntimeError: If ``path`` points to a missing or non-file path.

    """
    return tomllib.loads(
        read_packaged_toml(path, package=BASE_PACKAGE, resource=resource_name),
    )


def eval_config_source_text(path: Path | str | None = None) -> str:
    """Read the raw TOML source text backing the evaluation configuration.

    Exposed separately from ``load_eval_config`` so callers that need to
    fingerprint the exact configuration in effect (e.g. attaching a content
    hash to eval-run metadata) can do so without re-parsing it.

    Args:
        path: Optional explicit path to a TOML file; defaults to the packaged
            ``eval_config.toml`` resource.

    Returns:
        The raw, unparsed TOML text.

    Raises:
        RuntimeError: If ``path`` is provided but does not point to an
            existing file.

    """
    return read_packaged_toml(
        path, package=BASE_PACKAGE, resource=_EVAL_CONFIG_RESOURCE,
    )


@lru_cache(maxsize=1)
def load_eval_config(path: Path | str | None = None) -> EvalConfig:
    """Load evaluation configuration from a TOML file.

    Args:
        path: Optional explicit path to a TOML file; defaults to the packaged
            eval_config.toml resource.

    Returns:
        Parsed EvalConfig instance.

    Raises:
        RuntimeError: If the provided path is invalid or TOML cannot be parsed.

    """
    return EvalConfig.model_validate(_load_toml(path, _EVAL_CONFIG_RESOURCE))


@lru_cache(maxsize=1)
def load_stress_config(path: Path | str | None = None) -> StressEvalConfig:
    """Load stress-test configuration from a TOML file.

    Args:
        path: Optional explicit path to a TOML file; defaults to the packaged
            stress_config.toml resource.

    Returns:
        Parsed StressEvalConfig instance.

    Raises:
        RuntimeError: If the provided path is invalid or TOML cannot be parsed.

    """
    return StressEvalConfig.model_validate(_load_toml(path, _STRESS_CONFIG_RESOURCE))


def _resolve_path(value: object, *, base_dir: Path) -> Path:
    """Resolve a configuration path value to an absolute Path.

    Args:
        value: Raw config value representing a path.
        base_dir: Base directory for relative paths.

    Returns:
        Resolved Path instance.

    Raises:
        TypeError: If the value cannot be interpreted as a path.

    """
    if isinstance(value, Path):
        path = value.expanduser()
    elif isinstance(value, str):
        if not value.strip():
            msg = "Path values cannot be empty."
            raise TypeError(msg)
        path = Path(value).expanduser()
    else:
        msg = f"Invalid path value: {value!r}"
        raise TypeError(msg)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()
