"""Shared configuration parsing for Blue Horizon.

By default, this module loads from ``app_config.toml`` in the package root.
"""

import importlib.resources as importlib_resources
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

BASE_PACKAGE = "blue_horizon"
_APP_CONFIG_RESOURCE = "app_config.toml"


def _expect_table(data: Mapping[str, Any], key: str, section: str) -> Mapping[str, Any]:
    """Return a nested table while validating that it exists.

    Args:
        data: Mapping containing the root configuration.
        key: Entry key that should contain the nested table.
        section: Section name for error context.

    Returns:
        Mapping[str, Any]: The nested table under ``key``.

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
        T: The converted value.

    Raises:
        ValueError: If the key is missing or the converter fails.

    """
    if key not in data:
        msg = f"Missing config key: {section}.{key}"
        raise ValueError(msg)
    value = data[key]
    try:
        return converter(value)
    except (TypeError, ValueError) as err:
        msg = f"Invalid value for config key: {section}.{key}"
        raise ValueError(msg) from err


@dataclass(frozen=True, slots=True)
class InfoRetrievalConfig:
    """Retrieval configuration used by the information agent.

    Attributes:
        top_k: Number of results returned per retriever invocation.
        vector_dims: Dimensionality expected by the Redis vector index schema.
        retriever_cache_max: Maximum cached retrievers kept per process.

    """

    top_k: int
    vector_dims: int
    retriever_cache_max: int


@dataclass(frozen=True, slots=True)
class InfoEmbeddingsConfig:
    """Embedding generation configuration.

    Attributes:
        model: Embedding model name passed to OpenAI.
        batch_size: Number of records batched per embedding call.
        timeout_s: Network timeout in seconds for embedding requests.
        max_retries: Retry limit for transient embedding failures.

    """

    model: str
    batch_size: int
    timeout_s: float
    max_retries: int


@dataclass(frozen=True, slots=True)
class InfoLlmConfig:
    """LLM configuration for the information response agent.

    Attributes:
        model: Chat model identifier.
        temperature: Sampling temperature for generation calls.
        timeout_s: Client-side request timeout in seconds.
        max_retries: Retry budget for transient LLM failures.
        reasoning_effort: Reasoning effort hint (e.g., low, medium, high).

    """

    model: str
    temperature: float
    timeout_s: float
    max_retries: int
    reasoning_effort: str


@dataclass(frozen=True, slots=True)
class InfoRedisConfig:
    """Redis connectivity configuration.

    Attributes:
        connect_timeout_s: Timeout for establishing new Redis connections.
        socket_timeout_s: Socket read/write timeout in seconds.
        health_check_interval_s: Interval between health checks on idle connections.
        retry_max_retries: Retry count for transient Redis failures.
        retry_backoff_base_s: Base backoff (seconds) for Redis retries.
        retry_backoff_max_s: Maximum backoff (seconds) for Redis retries.

    """

    connect_timeout_s: float
    socket_timeout_s: float
    health_check_interval_s: int
    retry_max_retries: int
    retry_backoff_base_s: float
    retry_backoff_max_s: float


@dataclass(frozen=True, slots=True)
class InfoPromptsConfig:
    """Prompt template location configuration.

    Attributes:
        folder: Directory relative to the package root that contains prompts.
        system_prompt_filename: Filename for the information agent system prompt.

    """

    folder: str
    system_prompt_filename: str


@dataclass(frozen=True, slots=True)
class InfoRagConfig:
    """Top-level configuration for the information RAG agent.

    Attributes:
        retrieval: Retrieval configuration section.
        embeddings: Embedding generation settings.
        llm: Chat completion configuration.
        redis: Redis connection tuning.
        prompts: Prompt template metadata.

    """

    retrieval: InfoRetrievalConfig
    embeddings: InfoEmbeddingsConfig
    llm: InfoLlmConfig
    redis: InfoRedisConfig
    prompts: InfoPromptsConfig


@dataclass(frozen=True, slots=True)
class RoomsLlmConfig:
    """LLM settings for the rooms SQL agent.

    Attributes:
        model: Chat model identifier.
        temperature: Sampling temperature for SQL generation.
        reasoning_effort: Reasoning effort hint metadata.
        timeout_s: Timeout in seconds for LLM requests.
        max_retries: Retry count for transient LLM failures.

    """

    model: str
    temperature: float
    reasoning_effort: str
    timeout_s: float
    max_retries: int


@dataclass(frozen=True, slots=True)
class RoomsAgentConfig:
    """Agent prompt settings for SQL generation.

    Attributes:
        top_k: Prompt parameter used to describe search breadth.

    """

    top_k: int


@dataclass(frozen=True, slots=True)
class RoomsPromptsConfig:
    """Prompt template location for the rooms agent.

    Attributes:
        folder: Directory relative to the package containing prompts.
        system_prompt_filename: Filename of the rooms system prompt template.

    """

    folder: str
    system_prompt_filename: str


@dataclass(frozen=True, slots=True)
class DbPoolConfig:
    """Client-side connection pool tuning.

    Attributes:
        min_size: Minimum pool connections maintained.
        max_size: Maximum pool connections allowed.
        timeout_s: Timeout when acquiring a pooled connection.

    """

    min_size: int
    max_size: int
    timeout_s: float


@dataclass(frozen=True, slots=True)
class DbGuardrailsConfig:
    """SQL guardrail settings for the rooms agent.

    Attributes:
        max_rows: Maximum number of rows returned to the model.
        allow_only_hotel_tables: Whether to enforce the hotel table allowlist.

    """

    max_rows: int
    allow_only_hotel_tables: bool


@dataclass(frozen=True, slots=True)
class DbTimeoutsConfig:
    """Database-side timeout tuning.

    Attributes:
        statement_timeout_ms: PostgreSQL statement timeout in milliseconds.

    """

    statement_timeout_ms: int


@dataclass(frozen=True, slots=True)
class DbRetryConfig:
    """Transient retry configuration for DB operations.

    Attributes:
        max_transient_retries: Retry count after the initial attempt.
        transient_retry_backoff_s: Base backoff (seconds) per retry.
        retry_writes_on_transient_errors: Whether writes may be retried.

    """

    max_transient_retries: int
    transient_retry_backoff_s: float
    retry_writes_on_transient_errors: bool


@dataclass(frozen=True, slots=True)
class RoomsDbConfig:
    """Combined database configuration for the rooms agent.

    Attributes:
        pool: Client-side pool settings.
        guardrails: SQL validation constraints.
        timeouts: Database statement timeout values.
        retry: Transient retry policy.

    """

    pool: DbPoolConfig
    guardrails: DbGuardrailsConfig
    timeouts: DbTimeoutsConfig
    retry: DbRetryConfig


@dataclass(frozen=True, slots=True)
class RoomsSqlConfig:
    """Top-level configuration for the rooms SQL agent.

    Attributes:
        llm: Chat model configuration.
        agent: Prompt metadata.
        prompts: Prompt template paths.
        db: Database behavior configuration.

    """

    llm: RoomsLlmConfig
    agent: RoomsAgentConfig
    prompts: RoomsPromptsConfig
    db: RoomsDbConfig


@dataclass(frozen=True, slots=True)
class LoadDataInfoConfig:
    """Data ingestion configuration for Redis loaders.

    Attributes:
        data_path: Relative path to the information pickle dataset folder.

    """

    data_path: Path


@dataclass(frozen=True, slots=True)
class LoadDataRoomsConfig:
    """Data ingestion configuration for PostgreSQL loaders.

    Attributes:
        data_path: Relative path to the rooms pickle dataset folder.

    """

    data_path: Path


@dataclass(frozen=True, slots=True)
class LoadDataConfig:
    """Grouped data ingestion configuration.

    Attributes:
        information_redis: Redis data loader configuration.
        rooms_pgsql: PostgreSQL data loader configuration.

    """

    information_redis: LoadDataInfoConfig
    rooms_pgsql: LoadDataRoomsConfig


@dataclass(frozen=True, slots=True)
class OrchestrationLlmConfig:
    """LLM configuration for the router component.

    Attributes:
        model: Router chat model identifier.
        temperature: Sampling temperature.
        reasoning_effort: Reasoning effort hint.
        timeout_s: Network timeout for router LLM calls.
        max_retries: Retry budget for transient router failures.

    """

    model: str
    temperature: float
    reasoning_effort: str
    timeout_s: float
    max_retries: int


@dataclass(frozen=True, slots=True)
class OrchestrationRuntimeConfig:
    """Runtime controls for orchestration retries and timeouts.

    Attributes:
        init_retry_base_s: Initial backoff between initialization attempts.
        init_retry_max_s: Maximum backoff between retries.
        router_timeout_s: Timeout for the router decision step.
        info_timeout_s: Timeout for the info agent node.
        rooms_timeout_s: Timeout for the rooms agent node.

    """

    init_retry_base_s: float
    init_retry_max_s: float
    router_timeout_s: float
    info_timeout_s: float
    rooms_timeout_s: float


@dataclass(frozen=True, slots=True)
class OrchestrationPromptsConfig:
    """Prompt template location for orchestration.

    Attributes:
        folder: Directory relative to the package containing router prompts.
        orchestration_prompt_filename: Router system prompt filename.

    """

    folder: str
    orchestration_prompt_filename: str


@dataclass(frozen=True, slots=True)
class MessagesConfig:
    """User-facing message strings for orchestration.

    Attributes:
        refusal: Response when the user request is refused.
        error: Response for technical failures.
        unavailable: Message used while initialization is pending.

    """

    refusal: str
    error: str
    unavailable: str


@dataclass(frozen=True, slots=True)
class OrchestrationConfig:
    """Top-level orchestration configuration.

    Attributes:
        llm: Router LLM settings.
        orchestration: Runtime retry/timeouts.
        prompts: Prompt file metadata.
        messages: Configured user-facing strings.

    """

    llm: OrchestrationLlmConfig
    orchestration: OrchestrationRuntimeConfig
    prompts: OrchestrationPromptsConfig
    messages: MessagesConfig


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Parsed application configuration container.

    Attributes:
        orchestration: Router/orchestrator settings.
        info: Information agent settings.
        rooms: Rooms SQL agent settings.
        load_data: Data ingestion settings for local loaders.

    """

    orchestration: OrchestrationConfig
    info: InfoRagConfig
    rooms: RoomsSqlConfig
    load_data: LoadDataConfig


def parse_info_config(data: Mapping[str, Any]) -> InfoRagConfig:
    """Parse the info configuration from a dict.

    Args:
        data: Dict containing the ``info`` section.

    Returns:
        InfoRagConfig: Parsed configuration for the information agent.

    Raises:
        ValueError: If required keys are missing or values have invalid types.

    """
    section = "info"
    retrieval = _expect_table(data, "retrieval", section)
    embeddings = _expect_table(data, "embeddings", section)
    llm = _expect_table(data, "llm", section)
    redis = _expect_table(data, "redis", section)
    prompts = _expect_table(data, "prompts", section)

    return InfoRagConfig(
        retrieval=InfoRetrievalConfig(
            top_k=_get_required_value(retrieval, "top_k", section + ".retrieval", int),
            vector_dims=_get_required_value(
                retrieval, "vector_dims", section + ".retrieval", int,
            ),
            retriever_cache_max=max(
                1,
                _get_required_value(
                    retrieval,
                    "retriever_cache_max",
                    section + ".retrieval",
                    int,
                ),
            ),
        ),
        embeddings=InfoEmbeddingsConfig(
            model=_get_required_value(
                embeddings, "model", section + ".embeddings", str,
            ),
            batch_size=_get_required_value(
                embeddings, "batch_size", section + ".embeddings", int,
            ),
            timeout_s=_get_required_value(
                embeddings, "timeout_s", section + ".embeddings", float,
            ),
            max_retries=_get_required_value(
                embeddings, "max_retries", section + ".embeddings", int,
            ),
        ),
        llm=InfoLlmConfig(
            model=_get_required_value(llm, "model", section + ".llm", str),
            temperature=_get_required_value(
                llm, "temperature", section + ".llm", float,
            ),
            timeout_s=_get_required_value(llm, "timeout_s", section + ".llm", float),
            max_retries=_get_required_value(llm, "max_retries", section + ".llm", int),
            reasoning_effort=_get_required_value(
                llm,
                "reasoning_effort",
                section + ".llm",
                str,
            ),
        ),
        redis=InfoRedisConfig(
            connect_timeout_s=_get_required_value(
                redis,
                "connect_timeout_s",
                section + ".redis",
                float,
            ),
            socket_timeout_s=_get_required_value(
                redis,
                "socket_timeout_s",
                section + ".redis",
                float,
            ),
            health_check_interval_s=_get_required_value(
                redis,
                "health_check_interval_s",
                section + ".redis",
                int,
            ),
            retry_max_retries=_get_required_value(
                redis,
                "retry_max_retries",
                section + ".redis",
                int,
            ),
            retry_backoff_base_s=_get_required_value(
                redis,
                "retry_backoff_base_s",
                section + ".redis",
                float,
            ),
            retry_backoff_max_s=_get_required_value(
                redis,
                "retry_backoff_max_s",
                section + ".redis",
                float,
            ),
        ),
        prompts=InfoPromptsConfig(
            folder=_get_required_value(prompts, "folder", section + ".prompts", str),
            system_prompt_filename=_get_required_value(
                prompts,
                "system_prompt_filename",
                section + ".prompts",
                str,
            ),
        ),
    )


def parse_rooms_config(data: Mapping[str, Any]) -> RoomsSqlConfig:
    """Parse the rooms SQL configuration from a dict.

    Args:
        data: Dict containing the ``rooms`` section.

    Returns:
        RoomsSqlConfig: Parsed rooms SQL configuration.

    Raises:
        ValueError: If required keys are missing or values are invalid.

    """
    section = "rooms"
    llm = _expect_table(data, "llm", section)
    agent = _expect_table(data, "agent", section)
    prompts = _expect_table(data, "prompts", section)
    db = _expect_table(data, "db", section)
    pool = _expect_table(db, "pool", section + ".db")
    guardrails = _expect_table(db, "guardrails", section + ".db")
    timeouts = _expect_table(db, "timeouts", section + ".db")
    retry = _expect_table(db, "retry", section + ".db")

    return RoomsSqlConfig(
        llm=RoomsLlmConfig(
            model=_get_required_value(llm, "model", section + ".llm", str),
            temperature=_get_required_value(
                llm, "temperature", section + ".llm", float,
            ),
            reasoning_effort=_get_required_value(
                llm,
                "reasoning_effort",
                section + ".llm",
                str,
            ),
            timeout_s=_get_required_value(llm, "timeout_s", section + ".llm", float),
            max_retries=_get_required_value(llm, "max_retries", section + ".llm", int),
        ),
        agent=RoomsAgentConfig(
            top_k=_get_required_value(agent, "top_k", section + ".agent", int),
        ),
        prompts=RoomsPromptsConfig(
            folder=_get_required_value(prompts, "folder", section + ".prompts", str),
            system_prompt_filename=_get_required_value(
                prompts,
                "system_prompt_filename",
                section + ".prompts",
                str,
            ),
        ),
        db=RoomsDbConfig(
            pool=DbPoolConfig(
                min_size=_get_required_value(
                    pool, "min_size", section + ".db.pool", int,
                ),
                max_size=_get_required_value(
                    pool, "max_size", section + ".db.pool", int,
                ),
                timeout_s=_get_required_value(
                    pool, "timeout_s", section + ".db.pool", float,
                ),
            ),
            guardrails=DbGuardrailsConfig(
                max_rows=_get_required_value(
                    guardrails,
                    "max_rows",
                    section + ".db.guardrails",
                    int,
                ),
                allow_only_hotel_tables=_get_required_value(
                    guardrails,
                    "allow_only_hotel_tables",
                    section + ".db.guardrails",
                    bool,
                ),
            ),
            timeouts=DbTimeoutsConfig(
                statement_timeout_ms=_get_required_value(
                    timeouts,
                    "statement_timeout_ms",
                    section + ".db.timeouts",
                    int,
                ),
            ),
            retry=DbRetryConfig(
                max_transient_retries=_get_required_value(
                    retry,
                    "max_transient_retries",
                    section + ".db.retry",
                    int,
                ),
                transient_retry_backoff_s=_get_required_value(
                    retry,
                    "transient_retry_backoff_s",
                    section + ".db.retry",
                    float,
                ),
                retry_writes_on_transient_errors=_get_required_value(
                    retry,
                    "retry_writes_on_transient_errors",
                    section + ".db.retry",
                    bool,
                ),
            ),
        ),
    )


def parse_orchestration_config(data: Mapping[str, Any]) -> OrchestrationConfig:
    """Parse the orchestration configuration from a dict.

    Args:
        data: Dict containing the ``orchestration`` section.

    Returns:
        OrchestrationConfig: Parsed router and runtime configuration.

    Raises:
        ValueError: If required keys are missing or values are invalid.

    """
    section = "orchestration"
    llm = _expect_table(data, "llm", section)
    runtime = _expect_table(data, "orchestration", section)
    prompts = _expect_table(data, "prompts", section)
    messages = _expect_table(data, "messages", section)

    return OrchestrationConfig(
        llm=OrchestrationLlmConfig(
            model=_get_required_value(llm, "model", section + ".llm", str),
            temperature=_get_required_value(
                llm, "temperature", section + ".llm", float,
            ),
            reasoning_effort=_get_required_value(
                llm,
                "reasoning_effort",
                section + ".llm",
                str,
            ),
            timeout_s=_get_required_value(llm, "timeout_s", section + ".llm", float),
            max_retries=_get_required_value(llm, "max_retries", section + ".llm", int),
        ),
        orchestration=OrchestrationRuntimeConfig(
            init_retry_base_s=_get_required_value(
                runtime,
                "init_retry_base_s",
                section + ".orchestration",
                float,
            ),
            init_retry_max_s=_get_required_value(
                runtime,
                "init_retry_max_s",
                section + ".orchestration",
                float,
            ),
            router_timeout_s=_get_required_value(
                runtime,
                "router_timeout_s",
                section + ".orchestration",
                float,
            ),
            info_timeout_s=_get_required_value(
                runtime,
                "info_timeout_s",
                section + ".orchestration",
                float,
            ),
            rooms_timeout_s=_get_required_value(
                runtime,
                "rooms_timeout_s",
                section + ".orchestration",
                float,
            ),
        ),
        prompts=OrchestrationPromptsConfig(
            folder=_get_required_value(prompts, "folder", section + ".prompts", str),
            orchestration_prompt_filename=_get_required_value(
                prompts,
                "orchestration_prompt_filename",
                section + ".prompts",
                str,
            ),
        ),
        messages=MessagesConfig(
            refusal=_get_required_value(
                messages, "refusal", section + ".messages", str,
            ),
            error=_get_required_value(messages, "error", section + ".messages", str),
            unavailable=_get_required_value(
                messages,
                "unavailable",
                section + ".messages",
                str,
            ),
        ),
    )


def parse_load_data_config(data: Mapping[str, Any]) -> LoadDataConfig:
    """Parse the load data configuration from a dict.

    Args:
        data: Dict containing the ``load_data`` section.

    Returns:
        LoadDataConfig: Parsed data loader configuration.

    Raises:
        ValueError: If required keys are missing or values are invalid.

    """
    section = "load_data"
    information = _expect_table(data, "information_redis", section)
    rooms = _expect_table(data, "rooms_pgsql", section)

    return LoadDataConfig(
        information_redis=LoadDataInfoConfig(
            data_path=_get_required_value(
                information,
                "data_path",
                section + ".information_redis",
                Path,
            ),
        ),
        rooms_pgsql=LoadDataRoomsConfig(
            data_path=_get_required_value(
                rooms,
                "data_path",
                section + ".rooms_pgsql",
                Path,
            ),
        ),
    )


def parse_app_config(data: Mapping[str, Any]) -> AppConfig:
    """Parse the root ``app_config`` dictionary.

    Args:
        data: Entire TOML tree representing the packaged app config.

    Returns:
        AppConfig: Parsed orchestration, info, and rooms configurations.

    Raises:
        ValueError: If any of the top-level sections are missing or invalid.

    """
    orchestration = _expect_table(data, "orchestration", "root")
    info = _expect_table(data, "info", "root")
    rooms = _expect_table(data, "rooms", "root")
    load_data = _expect_table(data, "load_data", "root")

    return AppConfig(
        orchestration=parse_orchestration_config(orchestration),
        info=parse_info_config(info),
        rooms=parse_rooms_config(rooms),
        load_data=parse_load_data_config(load_data),
    )


def _load_toml_resource(relative_path: str) -> dict[str, Any]:
    """Load a TOML resource packaged under the base package.

    Args:
        relative_path: Path within the package pointing to a TOML file.

    Returns:
        dict[str, Any]: Parsed TOML content.

    Raises:
        RuntimeError: If the resource cannot be read or parsed.

    """
    try:
        text = (
            importlib_resources.files(BASE_PACKAGE)
            .joinpath(relative_path)
            .read_text(
                encoding="utf-8",
            )
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
    """Load TOML data from an arbitrary filesystem path.

    Args:
        path: Filesystem path to a TOML configuration file.

    Returns:
        dict[str, Any]: Parsed TOML dictionary.

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


@lru_cache(maxsize=1)
def load_app_config(path: Path | str | None = None) -> AppConfig:
    """Load the merged app configuration.

    Args:
        path: Optional explicit path to a TOML file; defaults to the packaged resource.

    Returns:
        AppConfig: Parsed configuration containing all agent sections.

    Raises:
        RuntimeError: If the provided path is invalid or its contents cannot be parsed.

    """
    if path is None:
        data = _load_toml_resource(_APP_CONFIG_RESOURCE)
    else:
        target_path = Path(path).expanduser().resolve()
        if not target_path.exists() or not target_path.is_file():
            msg = f"Config file not found: {target_path}"
            raise RuntimeError(msg)
        data = _load_toml_path(target_path)

    return parse_app_config(data)
