"""Evaluation configuration parsing for Blue Horizon.

By default, loads eval_config.toml from the eval package directory.
"""

from __future__ import annotations

import importlib.resources as importlib_resources
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

BASE_PACKAGE = "eval"
_EVAL_CONFIG_RESOURCE = "eval_config.toml"
_BASE_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Configuration for LangSmith experiment execution.

    Attributes:
        dataset_name: LangSmith dataset name to evaluate.
        experiment_prefix: Prefix used when generating experiment names.
        run_notes: Optional run notes appended to experiment names.
        max_concurrency: Maximum concurrent evaluations.
        output_dir: Base directory for local artifacts.
        upload_results: Whether to upload results to LangSmith.
        limit: Optional limit on the number of dataset examples to run.

    """

    dataset_name: str
    experiment_prefix: str
    run_notes: str | None
    max_concurrency: int
    output_dir: Path
    upload_results: bool
    limit: int | None


@dataclass(frozen=True, slots=True)
class MetadataConfig:
    """Optional metadata attached to LangSmith runs.

    Attributes:
        git_sha: Git commit SHA to record (optional).
        router_model: Router model identifier (optional).
        judge_model: Judge model identifier (optional).
        schema_version: Database schema version label (optional).

    """

    git_sha: str | None
    router_model: str | None
    judge_model: str | None
    schema_version: str | None


@dataclass(frozen=True, slots=True)
class EvaluatorLimitsConfig:
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

    context_max_chars: int
    context_max_items: int
    assistant_max_chars: int
    user_max_chars: int
    info_filter_failures_max: int
    required_tool_failures_max: int
    json_value_max: int
    rag_per_turn_json_max: int
    tripwire_hits_max: int


@dataclass(frozen=True, slots=True)
class RoomsDataConfig:
    """Configuration for rooms baseline data used in eval runs.

    Attributes:
        data_path: Filesystem path to the pickled rooms datasets.

    """

    data_path: Path


@dataclass(frozen=True, slots=True)
class OrchestrationConfig:
    """Configuration for orchestration startup timing.

    Attributes:
        ready_timeout_s: Timeout (seconds) to wait for orchestration readiness.

    """

    ready_timeout_s: float


@dataclass(frozen=True, slots=True)
class SchemaSlotsConfig:
    """Configuration for distributed schema slot throttling.

    Attributes:
        max_active_schemas: Maximum concurrent schemas (0 disables throttling).
        stale_after_s: Time before schema slots are considered stale.
        wait_timeout_s: Time to wait for an available slot.
        poll_interval_s: Poll interval while waiting for a slot.

    """

    max_active_schemas: int
    stale_after_s: int
    wait_timeout_s: float
    poll_interval_s: float


@dataclass(frozen=True, slots=True)
class ResetPoolConfig:
    """Configuration for schema reset connection pools.

    Attributes:
        max_size: Maximum pool size.
        min_size: Minimum pool size.

    """

    max_size: int
    min_size: int


@dataclass(frozen=True, slots=True)
class JudgeConfig:
    """Configuration for the LLM-as-judge evaluator.

    Attributes:
        model: Judge model name.

    """

    model: str


@dataclass(frozen=True, slots=True)
class RagasConfig:
    """Configuration for Ragas metrics evaluation.

    Attributes:
        turns_max: Maximum number of turns to score.
        contexts_max: Maximum retrieved contexts per turn.
        context_chars: Maximum characters per context.
        query_chars: Maximum characters for the user query.
        response_chars: Maximum characters for the assistant response.
        reference_chars: Maximum characters for the reference answer.

    """

    turns_max: int
    contexts_max: int
    context_chars: int
    query_chars: int
    response_chars: int
    reference_chars: int


@dataclass(frozen=True, slots=True)
class StressConfig:
    """Configuration for stress-test runs.

    Attributes:
        schema: Optional schema name override for the stress run.
        users: Number of concurrent simulated users.
        ops_per_user: Number of operations per user.
        max_concurrency: Maximum concurrent users.
        baseline_data_path: Baseline data directory for schema setup.
        output_dir: Base output directory for stress artifacts.
        stay_nights: Nights per booking in generated targets.
        num_targets: Number of available targets to precompute.
        hot_target_count: Hot contention target subset size.
        start_date: Search start date for availability targets.
        horizon_days: Search horizon in days from start_date.
        pool_max: Maximum size of the database connection pool.

    """

    schema: str | None
    users: int
    ops_per_user: int
    max_concurrency: int
    baseline_data_path: Path
    output_dir: Path
    stay_nights: int
    num_targets: int
    hot_target_count: int
    start_date: date
    horizon_days: int
    pool_max: int


@dataclass(frozen=True, slots=True)
class EvalConfig:
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
    data = _load_eval_toml(path)
    return parse_eval_config(data)


def _load_eval_toml(path: Path | str | None) -> dict[str, Any]:
    """Load the eval configuration TOML.

    Args:
        path: Optional filesystem path to a TOML file.

    Returns:
        Parsed TOML data.

    Raises:
        RuntimeError: If the TOML file cannot be read or parsed.

    """
    if path is None:
        return _load_toml_resource(_EVAL_CONFIG_RESOURCE)
    target_path = Path(path).expanduser().resolve()
    if not target_path.exists() or not target_path.is_file():
        msg = f"Config file not found: {target_path}"
        raise RuntimeError(msg)
    return _load_toml_path(target_path)


def _load_toml_resource(relative_path: str) -> dict[str, Any]:
    """Load a TOML resource packaged under the eval package.

    Args:
        relative_path: Path within the eval package to a TOML file.

    Returns:
        Parsed TOML data.

    Raises:
        RuntimeError: If the resource cannot be read or parsed.

    """
    try:
        text = (
            importlib_resources.files(BASE_PACKAGE)
            .joinpath(relative_path)
            .read_text(encoding="utf-8")
        )
    except FileNotFoundError as exc:
        msg = f"Resource not found: {BASE_PACKAGE}/{relative_path}"
        raise RuntimeError(msg) from exc
    except OSError as exc:
        msg = f"Failed to read resource: {BASE_PACKAGE}/{relative_path}"
        raise RuntimeError(msg) from exc

    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        msg = f"Invalid TOML in resource: {BASE_PACKAGE}/{relative_path}"
        raise RuntimeError(msg) from exc


def _load_toml_path(path: Path) -> dict[str, Any]:
    """Load TOML data from a filesystem path.

    Args:
        path: Filesystem path to a TOML configuration file.

    Returns:
        Parsed TOML data.

    Raises:
        RuntimeError: If the file cannot be read or contains invalid TOML.

    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        msg = f"Failed to read config file: {path}"
        raise RuntimeError(msg) from exc

    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        msg = f"Invalid TOML in config file: {path}"
        raise RuntimeError(msg) from exc


def parse_eval_config(data: Mapping[str, Any]) -> EvalConfig:
    """Parse the root eval configuration dictionary.

    Args:
        data: Root TOML data mapping.

    Returns:
        Parsed EvalConfig.

    Raises:
        ValueError: If required sections or values are missing or invalid.

    """
    experiment = _expect_table(data, "experiment", "root")
    metadata = _expect_table(data, "metadata", "root")
    evaluator_limits = _expect_table(data, "evaluator_limits", "root")
    rooms = _expect_table(data, "rooms", "root")
    orchestration = _expect_table(data, "orchestration", "root")
    schema_slots = _expect_table(data, "schema_slots", "root")
    reset_pool = _expect_table(data, "reset_pool", "root")
    judge = _expect_table(data, "judge", "root")
    ragas = _expect_table(data, "ragas", "root")
    stress = _expect_table(data, "stress", "root")

    return EvalConfig(
        experiment=parse_experiment_config(experiment),
        metadata=parse_metadata_config(metadata),
        evaluator_limits=parse_evaluator_limits_config(evaluator_limits),
        rooms=parse_rooms_config(rooms),
        orchestration=parse_orchestration_config(orchestration),
        schema_slots=parse_schema_slots_config(schema_slots),
        reset_pool=parse_reset_pool_config(reset_pool),
        judge=parse_judge_config(judge),
        ragas=parse_ragas_config(ragas),
        stress=parse_stress_config(stress),
    )


def _expect_table(
    data: Mapping[str, Any],
    key: str,
    section: str,
) -> Mapping[str, Any]:
    """Return a nested table while validating that it exists.

    Args:
        data: Mapping containing the root configuration.
        key: Entry key that should contain the nested table.
        section: Section name for error context.

    Returns:
        The nested table under ``key``.

    Raises:
        ValueError: If the key is missing.
        TypeError: If the retrieved value is not a table mapping.

    """
    raw = data.get(key)
    if raw is None:
        msg = f"Missing config section: {section}.{key}"
        raise ValueError(msg)
    if not isinstance(raw, Mapping):
        msg = f"Expected {section}.{key} to be a table"
        raise TypeError(msg)
    return raw


def parse_experiment_config(data: Mapping[str, Any]) -> ExperimentConfig:
    """Parse experiment configuration.

    Args:
        data: Config table containing experiment values.

    Returns:
        Parsed ExperimentConfig.

    """
    section = "experiment"
    max_concurrency = _get_required_value(data, "max_concurrency", section, int)
    max_concurrency = max(1, max_concurrency)
    output_dir = _get_required_value(
        data,
        "output_dir",
        section,
        lambda value: _resolve_path(value, base_dir=_BASE_DIR),
    )
    upload_results = _get_required_value(
        data,
        "upload_results",
        section,
        _parse_bool,
    )
    limit = _get_optional_value(data, "limit", section, _parse_optional_int)
    return ExperimentConfig(
        dataset_name=_get_required_value(data, "dataset_name", section, str),
        experiment_prefix=_get_required_value(
            data,
            "experiment_prefix",
            section,
            str,
        ),
        run_notes=_normalize_optional_str(
            _get_optional_value(data, "run_notes", section, str),
        ),
        max_concurrency=max_concurrency,
        output_dir=output_dir,
        upload_results=upload_results,
        limit=limit,
    )


def _get_required_value[T](
    data: Mapping[str, Any],
    key: str,
    section: str,
    converter: Callable[[Any], T],
) -> T:
    """Return a typed value while validating presence and format.

    Args:
        data: Config table containing the target key.
        key: Parameter name expected in ``data``.
        section: Section name used for error messages.
        converter: Callable used to coerce the raw value to the target type.

    Returns:
        The converted value.

    Raises:
        ValueError: If the key is missing or the converter fails.

    """
    if key not in data:
        msg = f"Missing config key: {section}.{key}"
        raise ValueError(msg)
    value = data[key]
    try:
        return converter(value)
    except (TypeError, ValueError) as exc:
        msg = f"Invalid value for config key: {section}.{key}"
        raise ValueError(msg) from exc


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


def _parse_bool(value: object) -> bool:
    """Parse a boolean from configuration values.

    Args:
        value: Raw config value representing a boolean.

    Returns:
        Parsed boolean value.

    Raises:
        ValueError: If the value cannot be parsed as a boolean.

    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    msg = f"Invalid boolean value: {value!r}"
    raise ValueError(msg)


def _parse_optional_int(value: object) -> int | None:
    """Parse an optional integer from configuration values.

    Args:
        value: Raw config value representing an optional integer.

    Returns:
        Parsed integer value, or None when empty.

    Raises:
        ValueError: If the value cannot be parsed as an integer.

    """
    if value is None:
        return None
    if isinstance(value, bool):
        msg = f"Invalid integer value: {value!r}"
        raise TypeError(msg)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(stripped)
        except ValueError as exc:
            msg = f"Invalid integer value: {value!r}"
            raise ValueError(msg) from exc
    msg = f"Invalid integer value: {value!r}"
    raise TypeError(msg)


def _get_optional_value[T](
    data: Mapping[str, Any],
    key: str,
    section: str,
    converter: Callable[[Any], T],
) -> T | None:
    """Return an optional typed value when present.

    Args:
        data: Config table containing the target key.
        key: Parameter name that may appear in ``data``.
        section: Section name used for error messages.
        converter: Callable used to coerce the raw value to the target type.

    Returns:
        Converted value when present, otherwise None.

    Raises:
        ValueError: If the key is present but conversion fails.

    """
    if key not in data:
        return None
    value = data[key]
    try:
        return converter(value)
    except (TypeError, ValueError) as exc:
        msg = f"Invalid value for config key: {section}.{key}"
        raise ValueError(msg) from exc


def _normalize_optional_str(value: str | None) -> str | None:
    """Normalize optional strings by trimming and converting empties to None.

    Args:
        value: Raw string value, if any.

    Returns:
        Trimmed string or None when empty.

    """
    if value is None:
        return None
    trimmed = value.strip()
    return trimmed or None


def parse_metadata_config(data: Mapping[str, Any]) -> MetadataConfig:
    """Parse optional metadata configuration.

    Args:
        data: Config table containing metadata values.

    Returns:
        Parsed MetadataConfig.

    """
    section = "metadata"
    return MetadataConfig(
        git_sha=_normalize_optional_str(
            _get_optional_value(data, "git_sha", section, str),
        ),
        router_model=_normalize_optional_str(
            _get_optional_value(data, "router_model", section, str),
        ),
        judge_model=_normalize_optional_str(
            _get_optional_value(data, "judge_model", section, str),
        ),
        schema_version=_normalize_optional_str(
            _get_optional_value(data, "schema_version", section, str),
        ),
    )


def parse_evaluator_limits_config(
    data: Mapping[str, Any],
) -> EvaluatorLimitsConfig:
    """Parse evaluator limits configuration.

    Args:
        data: Config table containing evaluator limit values.

    Returns:
        Parsed EvaluatorLimitsConfig.

    """
    section = "evaluator_limits"
    context_max_chars = _get_required_value(data, "context_max_chars", section, int)
    context_max_items = _get_required_value(data, "context_max_items", section, int)
    assistant_max_chars = _get_required_value(
        data,
        "assistant_max_chars",
        section,
        int,
    )
    user_max_chars = _get_required_value(data, "user_max_chars", section, int)
    info_filter_failures_max = _get_required_value(
        data,
        "info_filter_failures_max",
        section,
        int,
    )
    required_tool_failures_max = _get_required_value(
        data,
        "required_tool_failures_max",
        section,
        int,
    )
    json_value_max = _get_required_value(data, "json_value_max", section, int)
    rag_per_turn_json_max = _get_required_value(
        data,
        "rag_per_turn_json_max",
        section,
        int,
    )
    tripwire_hits_max = _get_required_value(data, "tripwire_hits_max", section, int)

    return EvaluatorLimitsConfig(
        context_max_chars=max(1, context_max_chars),
        context_max_items=max(1, context_max_items),
        assistant_max_chars=max(1, assistant_max_chars),
        user_max_chars=max(1, user_max_chars),
        info_filter_failures_max=max(1, info_filter_failures_max),
        required_tool_failures_max=max(1, required_tool_failures_max),
        json_value_max=max(1, json_value_max),
        rag_per_turn_json_max=max(1, rag_per_turn_json_max),
        tripwire_hits_max=max(1, tripwire_hits_max),
    )


def parse_rooms_config(data: Mapping[str, Any]) -> RoomsDataConfig:
    """Parse rooms baseline data configuration.

    Args:
        data: Config table containing rooms paths.

    Returns:
        Parsed RoomsDataConfig.

    """
    section = "rooms"
    data_path = _get_required_value(
        data,
        "data_path",
        section,
        lambda value: _resolve_path(value, base_dir=_BASE_DIR),
    )
    return RoomsDataConfig(data_path=data_path)


def parse_orchestration_config(data: Mapping[str, Any]) -> OrchestrationConfig:
    """Parse orchestration timing configuration.

    Args:
        data: Config table containing orchestration values.

    Returns:
        Parsed OrchestrationConfig.

    """
    section = "orchestration"
    timeout_s = _get_required_value(data, "ready_timeout_s", section, float)
    timeout_s = max(0.1, timeout_s)
    return OrchestrationConfig(ready_timeout_s=timeout_s)


def parse_schema_slots_config(data: Mapping[str, Any]) -> SchemaSlotsConfig:
    """Parse schema slot throttling configuration.

    Args:
        data: Config table containing schema slot values.

    Returns:
        Parsed SchemaSlotsConfig.

    """
    section = "schema_slots"
    max_active = _get_required_value(data, "max_active_schemas", section, int)
    stale_after_s = _get_required_value(data, "stale_after_s", section, int)
    wait_timeout_s = _get_required_value(data, "wait_timeout_s", section, float)
    poll_interval_s = _get_required_value(data, "poll_interval_s", section, float)
    return SchemaSlotsConfig(
        max_active_schemas=max(0, max_active),
        stale_after_s=max(0, stale_after_s),
        wait_timeout_s=max(0.1, wait_timeout_s),
        poll_interval_s=max(0.1, poll_interval_s),
    )


def parse_reset_pool_config(data: Mapping[str, Any]) -> ResetPoolConfig:
    """Parse schema reset pool configuration.

    Args:
        data: Config table containing pool sizes.

    Returns:
        Parsed ResetPoolConfig.

    """
    section = "reset_pool"
    max_size = _get_required_value(data, "max_size", section, int)
    min_size = _get_required_value(data, "min_size", section, int)
    max_size = max(1, max_size)
    min_size = max(1, min(min_size, max_size))
    return ResetPoolConfig(max_size=max_size, min_size=min_size)


def parse_judge_config(data: Mapping[str, Any]) -> JudgeConfig:
    """Parse judge model configuration.

    Args:
        data: Config table containing judge values.

    Returns:
        Parsed JudgeConfig.

    """
    section = "judge"
    return JudgeConfig(model=_get_required_value(data, "model", section, str))


def parse_ragas_config(data: Mapping[str, Any]) -> RagasConfig:
    """Parse Ragas evaluation configuration.

    Args:
        data: Config table containing Ragas values.

    Returns:
        Parsed RagasConfig.

    """
    section = "ragas"
    return RagasConfig(
        turns_max=max(0, _get_required_value(data, "turns_max", section, int)),
        contexts_max=max(0, _get_required_value(data, "contexts_max", section, int)),
        context_chars=max(0, _get_required_value(data, "context_chars", section, int)),
        query_chars=max(0, _get_required_value(data, "query_chars", section, int)),
        response_chars=max(
            0,
            _get_required_value(data, "response_chars", section, int),
        ),
        reference_chars=max(
            0,
            _get_required_value(data, "reference_chars", section, int),
        ),
    )


def parse_stress_config(data: Mapping[str, Any]) -> StressConfig:
    """Parse stress-test configuration.

    Args:
        data: Config table containing stress-test values.

    Returns:
        Parsed StressConfig.

    """
    section = "stress"
    schema = _normalize_optional_str(_get_optional_value(data, "schema", section, str))
    users = _get_required_value(data, "users", section, int)
    ops_per_user = _get_required_value(data, "ops_per_user", section, int)
    max_concurrency = _get_required_value(data, "max_concurrency", section, int)
    baseline_data_path = _get_required_value(
        data,
        "baseline_data_path",
        section,
        lambda value: _resolve_path(value, base_dir=_BASE_DIR),
    )
    output_dir = _get_required_value(
        data,
        "output_dir",
        section,
        lambda value: _resolve_path(value, base_dir=_BASE_DIR),
    )
    stay_nights = _get_required_value(data, "stay_nights", section, int)
    num_targets = _get_required_value(data, "num_targets", section, int)
    hot_target_count = _get_required_value(data, "hot_target_count", section, int)
    start_date = _get_required_value(
        data,
        "start_date",
        section,
        _parse_date,
    )
    horizon_days = _get_required_value(data, "horizon_days", section, int)
    pool_max = _get_required_value(data, "pool_max", section, int)

    return StressConfig(
        schema=schema,
        users=max(1, users),
        ops_per_user=max(1, ops_per_user),
        max_concurrency=max(1, max_concurrency),
        baseline_data_path=baseline_data_path,
        output_dir=output_dir,
        stay_nights=max(1, stay_nights),
        num_targets=max(1, num_targets),
        hot_target_count=max(0, hot_target_count),
        start_date=start_date,
        horizon_days=max(1, horizon_days),
        pool_max=max(1, pool_max),
    )


def _parse_date(value: object) -> date:
    """Parse an ISO-8601 date string.

    Args:
        value: Raw config value containing a date.

    Returns:
        Parsed date value.

    Raises:
        ValueError: If the date is not a valid ISO-8601 date string.

    """
    if not isinstance(value, str):
        msg = "Date values must be ISO-8601 strings."
        raise TypeError(msg)
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        msg = f"Invalid date '{value}', expected YYYY-MM-DD"
        raise ValueError(msg) from exc
