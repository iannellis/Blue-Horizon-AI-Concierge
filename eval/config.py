"""Evaluation configuration parsing for Blue Horizon.

By default, loads eval_config.toml from the eval package directory.
"""

from __future__ import annotations

import tomllib
from datetime import date  # noqa: TC003
from functools import lru_cache
from importlib import resources as importlib_resources
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_PACKAGE = "eval"
_EVAL_CONFIG_RESOURCE = "eval_config.toml"
_STRESS_CONFIG_RESOURCE = "stress_config.toml"
_BASE_DIR = Path(__file__).resolve().parents[1]


class ExperimentConfig(BaseModel):
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

    model_config = {"frozen": True}

    dataset_name: str
    experiment_prefix: str
    run_notes: str | None = None
    max_concurrency: Annotated[int, Field(ge=1)]
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

    @field_validator("max_concurrency", mode="before")
    @classmethod
    def _clamp_max_concurrency(cls, value: int) -> int:
        """Ensure max_concurrency is at least 1."""
        return max(1, value)

    @field_validator("output_dir", "log_dir", mode="before")
    @classmethod
    def _resolve_paths(cls, value: object) -> Path:
        """Resolve paths relative to repository root."""
        return _resolve_path(value, base_dir=_BASE_DIR)


class MetadataConfig(BaseModel):
    """Optional metadata attached to LangSmith runs.

    Attributes:
        git_sha: Git commit SHA to record (optional).
        router_model: Router model identifier (optional).
        judge_model: Judge model identifier (optional).
        schema_version: Database schema version label (optional).

    """

    model_config = {"frozen": True}

    git_sha: str | None = None
    router_model: str | None = None
    judge_model: str | None = None
    schema_version: str | None = None

    @field_validator(
        "git_sha",
        "router_model",
        "judge_model",
        "schema_version",
        mode="before",
    )
    @classmethod
    def _empty_str_to_none(cls, value: object) -> object:
        """Convert empty strings to None for optional metadata."""
        if isinstance(value, str) and not value.strip():
            return None
        return value


class EvaluatorLimitsConfig(BaseModel):
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

    model_config = {"frozen": True}

    context_max_chars: Annotated[int, Field(ge=1)]
    context_max_items: Annotated[int, Field(ge=1)]
    assistant_max_chars: Annotated[int, Field(ge=1)]
    user_max_chars: Annotated[int, Field(ge=1)]
    info_filter_failures_max: Annotated[int, Field(ge=1)]
    required_tool_failures_max: Annotated[int, Field(ge=1)]
    json_value_max: Annotated[int, Field(ge=1)]
    rag_per_turn_json_max: Annotated[int, Field(ge=1)]
    tripwire_hits_max: Annotated[int, Field(ge=1)]

    @field_validator("*", mode="before")
    @classmethod
    def _clamp_to_one(cls, value: int) -> int:
        """Ensure all limit values are at least 1."""
        return max(1, value)


class NeonConfig(BaseModel):
    """Configuration for the Neon branch reset used before each eval run.

    Attributes:
        project_id: Neon project ID (visible in the console URL:
            ``console.neon.tech/app/projects/<project_id>``).
        branch_name: Name of the branch to restore to its parent baseline.
        lock_retry_attempts: Retry attempts when the branch is locked (HTTP 423).
        lock_retry_delay_s: Seconds to wait between lock retry attempts.

    """

    model_config = {"frozen": True}

    project_id: str
    branch_name: str
    lock_retry_attempts: Annotated[int, Field(ge=1)] = 8
    lock_retry_delay_s: Annotated[float, Field(ge=0.0)] = 5.0

    @field_validator("lock_retry_attempts", mode="before")
    @classmethod
    def _clamp_to_one(cls, value: int) -> int:
        """Ensure lock_retry_attempts is at least 1."""
        return max(1, value)


class OrchestrationConfig(BaseModel):
    """Configuration for orchestration startup timing.

    Attributes:
        ready_timeout_s: Timeout (seconds) to wait for orchestration readiness.

    """

    model_config = {"frozen": True}

    ready_timeout_s: Annotated[float, Field(ge=0.1)]

    @field_validator("ready_timeout_s", mode="before")
    @classmethod
    def _clamp_timeout(cls, value: float) -> float:
        """Ensure timeout is at least 0.1 seconds."""
        return max(0.1, value)


class JudgeConfig(BaseModel):
    """Configuration for the LLM-as-judge evaluator.

    Attributes:
        model: Judge model name.

    """

    model_config = {"frozen": True}

    model: str


class RagasConfig(BaseModel):
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

    """

    model_config = {"frozen": True}

    turns_max: Annotated[int, Field(ge=0)]
    contexts_max: Annotated[int, Field(ge=0)]
    context_chars: Annotated[int, Field(ge=0)]
    query_chars: Annotated[int, Field(ge=0)]
    response_chars: Annotated[int, Field(ge=0)]
    reference_chars: Annotated[int, Field(ge=0)]
    llm_model: str
    llm_max_tokens: Annotated[int, Field(ge=1)]
    embedding_model: str

    @field_validator(
        "turns_max",
        "contexts_max",
        "context_chars",
        "query_chars",
        "response_chars",
        "reference_chars",
        mode="before",
    )
    @classmethod
    def _clamp_to_zero(cls, value: int) -> int:
        """Ensure values are non-negative."""
        return max(0, value)

    @field_validator("llm_max_tokens", mode="before")
    @classmethod
    def _clamp_to_one(cls, value: int) -> int:
        """Ensure max_tokens is at least 1."""
        return max(1, value)


class StressWorkloadConfig(BaseModel):
    """Workload and operation-mix parameters for stress-test runs.

    Attributes:
        users: Number of concurrent simulated users.
        ops_per_user: Number of operations per user.
        max_concurrency: Maximum concurrent users.
        book_weight: Relative weight for BOOK operations in the op-type mix.
        modify_weight: Relative weight for MODIFY operations in the op-type mix.
        cancel_weight: Relative weight for CANCEL operations in the op-type mix.

    """

    model_config = {"frozen": True}

    users: Annotated[int, Field(ge=1)]
    ops_per_user: Annotated[int, Field(ge=1)]
    max_concurrency: Annotated[int, Field(ge=1)]
    book_weight: Annotated[float, Field(ge=0.0)] = 0.5
    modify_weight: Annotated[float, Field(ge=0.0)] = 0.25
    cancel_weight: Annotated[float, Field(ge=0.0)] = 0.25

    @field_validator("users", "ops_per_user", "max_concurrency", mode="before")
    @classmethod
    def _clamp_to_one(cls, value: int) -> int:
        """Ensure values are at least 1."""
        return max(1, value)


class StressTargetsConfig(BaseModel):
    """Availability target discovery parameters for stress-test runs.

    Attributes:
        stay_nights: Nights per booking in generated targets.
        num_targets: Number of available targets to precompute.
        hot_target_count: Hot contention target subset size.
        hot_target_probability: Probability of selecting a hot target vs any target.
        start_date: Search start date for availability targets.
        horizon_days: Search horizon in days from start_date.

    """

    model_config = {"frozen": True}

    stay_nights: Annotated[int, Field(ge=1)]
    num_targets: Annotated[int, Field(ge=1)]
    hot_target_count: Annotated[int, Field(ge=0)]
    hot_target_probability: Annotated[float, Field(ge=0.0, le=1.0)] = 0.8
    start_date: date
    horizon_days: Annotated[int, Field(ge=1)]

    @field_validator("stay_nights", "num_targets", "horizon_days", mode="before")
    @classmethod
    def _clamp_to_one(cls, value: int) -> int:
        """Ensure values are at least 1."""
        return max(1, value)

    @field_validator("hot_target_count", mode="before")
    @classmethod
    def _clamp_to_zero(cls, value: int) -> int:
        """Ensure hot_target_count is non-negative."""
        return max(0, value)


class StressDbConfig(BaseModel):
    """Database connection and retry parameters for stress-test runs.

    Attributes:
        pool_max: Maximum size of the database connection pool.
        db_retry_attempts: Retry attempts on transient DB connection errors.
        db_retry_delay_s: Base delay in seconds between DB retry attempts.
        reconcile_max_detail: Maximum entries in reconciliation failure detail lists.

    """

    model_config = {"frozen": True}

    pool_max: Annotated[int, Field(ge=1)]
    db_retry_attempts: Annotated[int, Field(ge=1)] = 3
    db_retry_delay_s: Annotated[float, Field(ge=0.0)] = 2.0
    reconcile_max_detail: Annotated[int, Field(ge=1)] = 10

    @field_validator(
        "pool_max", "db_retry_attempts", "reconcile_max_detail", mode="before",
    )
    @classmethod
    def _clamp_to_one(cls, value: int) -> int:
        """Ensure values are at least 1."""
        return max(1, value)


class StressOutputConfig(BaseModel):
    """Output path configuration for stress-test runs.

    Attributes:
        output_dir: Base output directory for stress artifacts.
        log_dir: Directory for stress run logs.

    """

    model_config = {"frozen": True}

    output_dir: Path
    log_dir: Path

    @field_validator("output_dir", "log_dir", mode="before")
    @classmethod
    def _resolve_paths(cls, value: object) -> Path:
        """Resolve paths relative to repository root."""
        return _resolve_path(value, base_dir=_BASE_DIR)


class StressConfig(BaseModel):
    """Configuration for stress-test runs, composed from sub-sections.

    Attributes:
        workload: Workload and operation-mix parameters.
        targets: Availability target discovery parameters.
        db: Database connection and retry parameters.
        output: Output path configuration.

    """

    model_config = {"frozen": True}

    workload: StressWorkloadConfig
    targets: StressTargetsConfig
    db: StressDbConfig
    output: StressOutputConfig


class EvalConfig(BaseSettings):
    """Parsed evaluation configuration container.

    TOML configuration is loaded via ``load_eval_config()``.  Environment
    variables are read from the process environment or a ``.env`` file.

    Attributes:
        experiment: Experiment execution settings.
        metadata: Optional LangSmith metadata values.
        evaluator_limits: Limits used by evaluators and summaries.
        neon: Neon branch reset configuration for eval runs.
        orchestration: Orchestration readiness timing.
        judge: Judge model configuration.
        ragas: Ragas scoring configuration.
        pgsql_eval_db_url: PostgreSQL database URL override for eval runs.
            When set, takes precedence over ``PGSQL_DB_URL`` for the rooms
            agent and evaluator pool connections.
        neon_api_key: API key for authenticating with the Neon management API,
            required when resetting a Neon branch before an eval run.

    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    experiment: ExperimentConfig
    metadata: MetadataConfig
    evaluator_limits: EvaluatorLimitsConfig
    neon: NeonConfig
    orchestration: OrchestrationConfig
    judge: JudgeConfig
    ragas: RagasConfig
    pgsql_eval_db_url: str | None = Field(
        default=None,
        validation_alias="PGSQL_EVAL_DB_URL",
    )
    neon_api_key: str | None = Field(
        default=None,
        validation_alias="NEON_API_KEY",
    )


class StressEvalConfig(BaseSettings):
    """Parsed stress-test configuration container.

    TOML configuration is loaded via ``load_stress_config()``.  Environment
    variables are read from the process environment or a ``.env`` file.

    Attributes:
        neon: Neon branch reset configuration for stress runs.
        orchestration: Orchestration readiness timing.
        stress: Stress-test workload, target, and database parameters.
        pgsql_eval_db_url: PostgreSQL database URL override for stress runs.
            When set, takes precedence over ``PGSQL_DB_URL`` for the rooms
            agent and pool connections.
        neon_api_key: API key for authenticating with the Neon management API,
            required when resetting a Neon branch before a stress run.

    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    neon: NeonConfig
    orchestration: OrchestrationConfig
    stress: StressConfig
    pgsql_eval_db_url: str | None = Field(
        default=None,
        validation_alias="PGSQL_EVAL_DB_URL",
    )
    neon_api_key: str | None = Field(
        default=None,
        validation_alias="NEON_API_KEY",
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
    if path is None:
        # Load the packaged TOML resource
        toml_content = (
            importlib_resources.files(BASE_PACKAGE)
            .joinpath(_EVAL_CONFIG_RESOURCE)
            .read_text(encoding="utf-8")
        )
        toml_data = tomllib.loads(toml_content)
        return EvalConfig.model_validate(toml_data)
    target_path = Path(path).expanduser().resolve()
    if not target_path.exists() or not target_path.is_file():
        msg = f"Config file not found: {target_path}"
        raise RuntimeError(msg)
    # Load custom TOML file
    toml_content = target_path.read_text(encoding="utf-8")
    toml_data = tomllib.loads(toml_content)
    return EvalConfig.model_validate(toml_data)


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
    if path is None:
        toml_content = (
            importlib_resources.files(BASE_PACKAGE)
            .joinpath(_STRESS_CONFIG_RESOURCE)
            .read_text(encoding="utf-8")
        )
        toml_data = tomllib.loads(toml_content)
        return StressEvalConfig.model_validate(toml_data)
    target_path = Path(path).expanduser().resolve()
    if not target_path.exists() or not target_path.is_file():
        msg = f"Config file not found: {target_path}"
        raise RuntimeError(msg)
    toml_content = target_path.read_text(encoding="utf-8")
    toml_data = tomllib.loads(toml_content)
    return StressEvalConfig.model_validate(toml_data)


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
