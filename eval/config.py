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

from pydantic import BaseModel, Field, field_validator, model_validator

BASE_PACKAGE = "eval"
_EVAL_CONFIG_RESOURCE = "eval_config.toml"
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
    """Limits for evaluator summaries and expected filter checks.

    Attributes:
        context_max_chars: Maximum characters per context snippet.
        context_max_items: Maximum number of context snippets to include.
        assistant_max_chars: Maximum characters from assistant text.
        user_max_chars: Maximum characters from user text.
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


class RoomsDataConfig(BaseModel):
    """Configuration for rooms baseline data used in eval runs.

    Attributes:
        data_path: Filesystem path to the pickled rooms datasets.

    """

    model_config = {"frozen": True}

    data_path: Path

    @field_validator("data_path", mode="before")
    @classmethod
    def _resolve_data_path(cls, value: object) -> Path:
        """Resolve data_path relative to repository root."""
        return _resolve_path(value, base_dir=_BASE_DIR)


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


class SchemaSlotsConfig(BaseModel):
    """Configuration for distributed schema slot throttling.

    Attributes:
        max_active_schemas: Maximum concurrent schemas (0 disables throttling).
        stale_after_s: Time before schema slots are considered stale.
        wait_timeout_s: Time to wait for an available slot.
        poll_interval_s: Poll interval while waiting for a slot.

    """

    model_config = {"frozen": True}

    max_active_schemas: Annotated[int, Field(ge=0)]
    stale_after_s: Annotated[int, Field(ge=0)]
    wait_timeout_s: Annotated[float, Field(ge=0.1)]
    poll_interval_s: Annotated[float, Field(ge=0.1)]

    @field_validator("max_active_schemas", "stale_after_s", mode="before")
    @classmethod
    def _clamp_to_zero(cls, value: int) -> int:
        """Ensure values are non-negative."""
        return max(0, value)

    @field_validator("wait_timeout_s", "poll_interval_s", mode="before")
    @classmethod
    def _clamp_timeout(cls, value: float) -> float:
        """Ensure timeout values are at least 0.1 seconds."""
        return max(0.1, value)


class ResetPoolConfig(BaseModel):
    """Configuration for schema reset connection pools.

    Attributes:
        max_size: Maximum pool size.
        min_size: Minimum pool size.

    """

    model_config = {"frozen": True}

    max_size: Annotated[int, Field(ge=1)]
    min_size: Annotated[int, Field(ge=1)]

    @field_validator("max_size", "min_size", mode="before")
    @classmethod
    def _clamp_to_one(cls, value: int) -> int:
        """Ensure pool sizes are at least 1."""
        return max(1, value)

    @model_validator(mode="after")
    def _validate_pool_sizes(self) -> ResetPoolConfig:
        """Ensure min_size <= max_size."""
        if self.min_size > self.max_size:
            object.__setattr__(self, "min_size", self.max_size)
        return self


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


class StressConfig(BaseModel):
    """Configuration for stress-test runs.

    Attributes:
        db_schema: Optional database schema name override for the stress run.
        users: Number of concurrent simulated users.
        ops_per_user: Number of operations per user.
        max_concurrency: Maximum concurrent users.
        baseline_data_path: Baseline data directory for schema setup.
        output_dir: Base output directory for stress artifacts.
        stay_nights: Nights per booking in generated targets.
        num_targets: Number of available targets to precompute.
        hot_target_count: Hot contention target subset size.
        hot_target_probability: Probability of selecting a hot target vs any target.
        start_date: Search start date for availability targets.
        horizon_days: Search horizon in days from start_date.
        pool_max: Maximum size of the database connection pool.

    """

    model_config = {"frozen": True}

    db_schema: str | None = None
    users: Annotated[int, Field(ge=1)]
    ops_per_user: Annotated[int, Field(ge=1)]
    max_concurrency: Annotated[int, Field(ge=1)]
    baseline_data_path: Path
    output_dir: Path
    stay_nights: Annotated[int, Field(ge=1)]
    num_targets: Annotated[int, Field(ge=1)]
    hot_target_count: Annotated[int, Field(ge=0)]
    hot_target_probability: Annotated[float, Field(ge=0.0, le=1.0)] = 0.8
    start_date: date
    horizon_days: Annotated[int, Field(ge=1)]
    pool_max: Annotated[int, Field(ge=1)]

    @field_validator("db_schema", mode="before")
    @classmethod
    def _empty_str_to_none(cls, value: object) -> object:
        """Convert empty strings to None for optional db_schema."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator(
        "users",
        "ops_per_user",
        "max_concurrency",
        "stay_nights",
        "num_targets",
        "horizon_days",
        "pool_max",
        mode="before",
    )
    @classmethod
    def _clamp_to_one(cls, value: int) -> int:
        """Ensure values are at least 1."""
        return max(1, value)

    @field_validator("hot_target_count", mode="before")
    @classmethod
    def _clamp_to_zero(cls, value: int) -> int:
        """Ensure hot_target_count is non-negative."""
        return max(0, value)

    @field_validator("baseline_data_path", "output_dir", mode="before")
    @classmethod
    def _resolve_paths(cls, value: object) -> Path:
        """Resolve paths relative to repository root."""
        return _resolve_path(value, base_dir=_BASE_DIR)


class EvalConfig(BaseModel):
    """Parsed evaluation configuration container.

    Attributes:
        experiment: Experiment execution settings.
        metadata: Optional LangSmith metadata values.
        evaluator_limits: Limits used by evaluators and summaries.
        rooms: Rooms dataset paths.
        orchestration: Orchestration readiness timing.
        schema_slots: Schema slot throttling parameters.
        reset_pool: Schema reset pool sizes.
        judge: Judge model configuration.
        ragas: Ragas scoring configuration.
        stress: Stress-test configuration values.

    """

    model_config = {"frozen": True}

    experiment: ExperimentConfig
    metadata: MetadataConfig
    evaluator_limits: EvaluatorLimitsConfig
    rooms: RoomsDataConfig
    orchestration: OrchestrationConfig
    schema_slots: SchemaSlotsConfig
    reset_pool: ResetPoolConfig
    judge: JudgeConfig
    ragas: RagasConfig
    stress: StressConfig


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
