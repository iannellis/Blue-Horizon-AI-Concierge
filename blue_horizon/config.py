"""Shared configuration parsing for Blue Horizon.

By default, this module loads from ``app_config.toml`` in the package root.
"""

import tomllib
from functools import lru_cache
from importlib import resources as importlib_resources
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_PACKAGE = "blue_horizon"
_APP_CONFIG_RESOURCE = "app_config.toml"


class InfoRetrievalConfig(BaseModel):
    """Retrieval configuration used by the information agent.

    Attributes:
        top_k: Number of results returned per retriever invocation.
        max_context_items: Hard ceiling on the total number of retrieved items
            passed to the LLM after merging all sources. Acts as a safety cap
            against runaway context growth when many sub-queries are parsed.
        vector_dims: Dimensionality expected by the Redis vector index schema.
        retriever_cache_max: Maximum cached retrievers kept per process.

    """

    model_config = {"frozen": True}

    top_k: int
    max_context_items: int
    vector_dims: int
    retriever_cache_max: int


class InfoEmbeddingsConfig(BaseModel):
    """Embedding generation configuration.

    Attributes:
        model: Embedding model name passed to OpenAI.
        batch_size: Number of records batched per embedding call.
        timeout_s: Network timeout in seconds for embedding requests.
        max_retries: Retry limit for transient embedding failures.

    """

    model_config = {"frozen": True}

    model: str
    batch_size: int
    timeout_s: float
    max_retries: int


class InfoLlmConfig(BaseModel):
    """LLM configuration for the information response agent.

    Attributes:
        model: Chat model identifier.
        temperature: Sampling temperature for generation calls.
        timeout_s: Client-side request timeout in seconds.
        max_retries: Retry budget for transient LLM failures.
        reasoning_effort: Reasoning effort hint (e.g., low, medium, high).

    """

    model_config = {"frozen": True}

    model: str
    temperature: float
    timeout_s: float
    max_retries: int
    reasoning_effort: str


class InfoRedisConfig(BaseModel):
    """Redis connectivity configuration.

    Attributes:
        connect_timeout_s: Timeout for establishing new Redis connections.
        socket_timeout_s: Socket read/write timeout in seconds.
        health_check_interval_s: Interval between health checks on idle connections.
        retry_max_retries: Retry count for transient Redis failures.
        retry_backoff_base_s: Base backoff (seconds) for Redis retries.
        retry_backoff_max_s: Maximum backoff (seconds) for Redis retries.

    """

    model_config = {"frozen": True}

    connect_timeout_s: float
    socket_timeout_s: float
    health_check_interval_s: int
    retry_max_retries: int
    retry_backoff_base_s: float
    retry_backoff_max_s: float


class InfoPromptsConfig(BaseModel):
    """Prompt template location configuration.

    Attributes:
        folder: Directory relative to the package root that contains prompts.
        system_prompt_filename: Filename for the information agent system prompt.

    """

    model_config = {"frozen": True}

    folder: str
    system_prompt_filename: str


class InfoRagConfig(BaseModel):
    """Top-level configuration for the information RAG agent.

    Attributes:
        retrieval: Retrieval configuration section.
        embeddings: Embedding generation settings.
        llm: Chat completion configuration.
        redis: Redis connection tuning.
        prompts: Prompt template metadata.

    """

    model_config = {"frozen": True}

    retrieval: InfoRetrievalConfig
    embeddings: InfoEmbeddingsConfig
    llm: InfoLlmConfig
    redis: InfoRedisConfig
    prompts: InfoPromptsConfig


class RoomsLlmConfig(BaseModel):
    """LLM settings for the rooms SQL agent.

    Attributes:
        model: Chat model identifier.
        temperature: Sampling temperature for SQL generation.
        reasoning_effort: Reasoning effort hint metadata.
        timeout_s: Timeout in seconds for LLM requests.
        max_retries: Retry count for transient LLM failures.

    """

    model_config = {"frozen": True}

    model: str
    temperature: float
    reasoning_effort: str
    timeout_s: float
    max_retries: int


class RoomsAgentConfig(BaseModel):
    """Agent prompt settings for SQL generation.

    Attributes:
        top_k: Prompt parameter used to describe search breadth.

    """

    model_config = {"frozen": True}

    top_k: int


class RoomsPromptsConfig(BaseModel):
    """Prompt template location for the rooms agent.

    Attributes:
        folder: Directory relative to the package containing prompts.
        system_prompt_filename: Filename of the rooms system prompt template.

    """

    model_config = {"frozen": True}

    folder: str
    system_prompt_filename: str


class DbPoolConfig(BaseModel):
    """Client-side connection pool tuning.

    Attributes:
        min_size: Minimum pool connections maintained.  Use ``0`` for
            on-demand-only connections (recommended for Neon serverless so
            the pool does not proactively open connections that will idle
            and be dropped by the server).
        max_size: Maximum pool connections allowed.
        timeout_s: Timeout in seconds when acquiring a pooled connection.
        max_idle_s: Seconds after which an idle connection is closed and
            discarded.  Set below the hosting provider's compute-suspend
            threshold (e.g. ``240`` for Neon's 300-second window) so stale
            connections are removed before the server drops them.

    """

    model_config = {"frozen": True}

    min_size: int
    max_size: int
    timeout_s: float
    max_idle_s: float


class DbGuardrailsConfig(BaseModel):
    """SQL guardrail settings for the rooms agent.

    Attributes:
        max_rows: Maximum number of rows returned to the model.
        allow_only_hotel_tables: Whether to enforce the hotel table allowlist.

    """

    model_config = {"frozen": True}

    max_rows: int
    allow_only_hotel_tables: bool


class DbRetryConfig(BaseModel):
    """Transient retry configuration for DB operations.

    Attributes:
        max_transient_retries: Retry count after the initial attempt.
        transient_retry_backoff_s: Base backoff (seconds) per retry.
        retry_writes_on_transient_errors: Whether writes may be retried.

    """

    model_config = {"frozen": True}

    max_transient_retries: int
    transient_retry_backoff_s: float
    retry_writes_on_transient_errors: bool


class RoomsDbConfig(BaseModel):
    """Combined database configuration for the rooms agent.

    Note:
        Statement timeout and search_path are set at the database role level
        (``ALTER ROLE … SET …``) rather than in code, so they apply
        consistently under connection pooling without any per-connection
        ``SET`` commands.

    Attributes:
        pool: Client-side pool settings.
        guardrails: SQL validation constraints.
        retry: Transient retry policy.

    """

    model_config = {"frozen": True}

    pool: DbPoolConfig
    guardrails: DbGuardrailsConfig
    retry: DbRetryConfig


class RoomsSqlConfig(BaseModel):
    """Top-level configuration for the rooms SQL agent.

    Attributes:
        llm: Chat model configuration.
        agent: Prompt metadata.
        prompts: Prompt template paths.
        db: Database behavior configuration.

    """

    model_config = {"frozen": True}

    llm: RoomsLlmConfig
    agent: RoomsAgentConfig
    prompts: RoomsPromptsConfig
    db: RoomsDbConfig


class LoadDataInfoConfig(BaseModel):
    """Data ingestion configuration for Redis loaders.

    Attributes:
        data_path: Relative path to the information pickle dataset folder.

    """

    model_config = {"frozen": True}

    data_path: Path


class LoadDataRoomsConfig(BaseModel):
    """Data ingestion configuration for PostgreSQL loaders.

    Attributes:
        data_path: Relative path to the rooms pickle dataset folder.

    """

    model_config = {"frozen": True}

    data_path: Path


class LoadDataConfig(BaseModel):
    """Grouped data ingestion configuration.

    Attributes:
        information_redis: Redis data loader configuration.
        rooms_pgsql: PostgreSQL data loader configuration.

    """

    model_config = {"frozen": True}

    information_redis: LoadDataInfoConfig
    rooms_pgsql: LoadDataRoomsConfig


class OrchestrationLlmConfig(BaseModel):
    """LLM configuration for the router component.

    Attributes:
        model: Router chat model identifier.
        temperature: Sampling temperature.
        reasoning_effort: Reasoning effort hint.
        timeout_s: Network timeout for router LLM calls.
        max_retries: Retry budget for transient router failures.

    """

    model_config = {"frozen": True}

    model: str
    temperature: float
    reasoning_effort: str
    timeout_s: float
    max_retries: int


class OrchestrationRuntimeConfig(BaseModel):
    """Runtime controls for orchestration retries and timeouts.

    Attributes:
        init_retry_base_s: Initial backoff between initialization attempts.
        init_retry_max_s: Maximum backoff between retries.
        router_timeout_s: Timeout for the router decision step.
        info_timeout_s: Timeout for the info agent node.
        rooms_timeout_s: Timeout for the rooms agent node.
        llm_concurrency: Maximum number of concurrent LLM pipeline executions.
            Limits simultaneous ainvoke calls to prevent token-per-minute
            exhaustion under high concurrency.

    """

    model_config = {"frozen": True}

    init_retry_base_s: float
    init_retry_max_s: float
    router_timeout_s: float
    info_timeout_s: float
    rooms_timeout_s: float
    llm_concurrency: Annotated[int, Field(ge=1)] = 10


class OrchestrationPromptsConfig(BaseModel):
    """Prompt template location for orchestration.

    Attributes:
        folder: Directory relative to the package containing router prompts.
        orchestration_prompt_filename: Router system prompt filename.

    """

    model_config = {"frozen": True}

    folder: str
    orchestration_prompt_filename: str


class MessagesConfig(BaseModel):
    """User-facing message strings for orchestration.

    Attributes:
        refusal: Response when the user request is refused.
        error: Response for technical failures.
        unavailable: Message used while initialization is pending.

    """

    model_config = {"frozen": True}

    refusal: str
    error: str
    unavailable: str


class OrchestrationConfig(BaseModel):
    """Top-level orchestration configuration.

    Attributes:
        llm: Router LLM settings.
        orchestration: Runtime retry/timeouts.
        prompts: Prompt file metadata.
        messages: Configured user-facing strings.

    """

    model_config = {"frozen": True}

    llm: OrchestrationLlmConfig
    orchestration: OrchestrationRuntimeConfig
    prompts: OrchestrationPromptsConfig
    messages: MessagesConfig


class AppConfig(BaseSettings):
    """Parsed application configuration container.

    TOML configuration is loaded via load_app_config(), which handles both
    packaged resources and custom file paths. Environment variables are loaded
    from .env files automatically.

    Attributes:
        orchestration: Router/orchestrator settings.
        info: Information agent settings.
        rooms: Rooms SQL agent settings.
        load_data: Data ingestion settings for local loaders.
        redis_url: Redis connection URL from environment.
        pgsql_db_url: PostgreSQL connection URL from environment.

    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    orchestration: OrchestrationConfig
    info: InfoRagConfig
    rooms: RoomsSqlConfig
    load_data: LoadDataConfig
    redis_url: str = Field(validation_alias="REDIS_URL")
    pgsql_db_url: str = Field(validation_alias="PGSQL_DB_URL")


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
        # Load the packaged TOML resource
        toml_content = (
            importlib_resources.files(BASE_PACKAGE)
            .joinpath(_APP_CONFIG_RESOURCE)
            .read_text(encoding="utf-8")
        )
        toml_data = tomllib.loads(toml_content)
        return AppConfig.model_validate(toml_data)
    target_path = Path(path).expanduser().resolve()
    if not target_path.exists() or not target_path.is_file():
        msg = f"Config file not found: {target_path}"
        raise RuntimeError(msg)
    # Load custom TOML file
    toml_content = target_path.read_text(encoding="utf-8")
    toml_data = tomllib.loads(toml_content)
    return AppConfig.model_validate(toml_data)
