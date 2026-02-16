"""Hotel info RAG agent (LangGraph DAG) with RedisVectorStore.

Goals:
- Modular, readable, PEP8-compliant
- Fast startup and steady-state performance
- Async-friendly for FastAPI (no blocking event loop)

Assumptions:
- Redis keys follow the pattern: {prefix}:{id} where prefix is one of:
  Source.FAQ, Source.AMENITIES, Source.SERVICES
- Each key stores fields: "_node_content" (JSON string) and "text" (string)
- booking_required is stored as a Redis TAG field with values "True" / "False".

Configuration:
- All tunables live in a TOML file (see `InfoRagConfig`).
- `system_prompt_filename` in TOML is the *file name only* (no path). All prompts are
  assumed to live in the same prompts folder.

Versions:
- langgraph (>=1.0.4,<2.0.0), llama-index(>=0.14.6,<0.15.0)
- llama-index-vector-stores-redis (>=0.6.1,<0.7.0)
"""

from __future__ import annotations

import asyncio
import heapq
import logging
import math
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, cast

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
from llama_index.core.vector_stores.types import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.redis import RedisVectorStore
from pydantic import BaseModel, Field
from redis.asyncio import Redis as AsyncRedis
from redis.backoff import ExponentialBackoff
from redis.exceptions import BusyLoadingError
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError
from redis.retry import Retry
from redisvl.schema import IndexSchema

from blue_horizon.agents.exceptions import OperationalError
from blue_horizon.agents.prompt_utils import load_prompt_template
from blue_horizon.config import (
    InfoEmbeddingsConfig,
    InfoRagConfig,
    InfoRedisConfig,
    load_app_config,
)

if TYPE_CHECKING:
    from pathlib import Path

    from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)


# ============================
# Settings (loaded from TOML config)
# ============================


def load_info_config(config_path: Path | str | None = None) -> InfoRagConfig:
    """Load the info configuration section.

    For when using the information agent standalone.

    Args:
        config_path: Optional path to override the packaged config. If unset,
            ``app_config.toml`` from the package resources is used.

    Returns:
        InfoRagConfig: Parsed configuration for the information agent.

    """
    app_config = load_app_config(path=config_path)
    return app_config.info


# ============================
# Prompts (file resolution + template loading)
# ============================


def build_system_prompt(*, top_k: int, prompt_resource: str) -> str:
    """Render the system prompt with runtime substitutions.

    Args:
        top_k: Maximum number of retrieval items referenced by the prompt.
        prompt_resource: Package resource path to the prompt template file.

    Returns:
        Rendered system prompt with runtime placeholders filled in.

    Raises:
        FileNotFoundError: If the prompt template resource cannot be located.
        RuntimeError: If the template cannot be loaded or substituted.

    """
    template = load_prompt_template(prompt_resource)
    return template.safe_substitute(top_k=top_k)


# ============================
# Domain models
# ============================


class Source(StrEnum):
    """Enumeration of knowledge sources.

    Values correspond to:
        - Redis index names
        - Redis key prefixes
        - The ``source`` field in retrieval results

    """

    FAQ = "faq"
    AMENITIES = "amenities"
    SERVICES = "services"


class RetrievalItem(BaseModel):
    """Retrieval result containing text and metadata.

    Attributes:
        source: Origin source of the item (faq/amenities/services).
        metadata: Metadata payload associated with the node.
        text: Human-readable text for the node.
        score: Similarity score used for ranking.

    """

    source: Source = Field(..., description="Which retriever produced this item")
    metadata: dict[str, Any] = Field(
        ...,
        description="Metadata associated with the text",
    )
    text: str = Field(..., description="The item name and description")
    score: float = Field(..., description="Similarity score; higher is more relevant")


class RerankInput(BaseModel):
    """Input schema for the reranker tool.

    Attributes:
        faq_results: Results returned by ``query_faq``.
        amenities_results: Results returned by ``query_amenities``.
        services_results: Results returned by ``query_services``.

    """

    faq_results: list[RetrievalItem] = Field(
        ...,
        description="Output from query_faq",
    )
    amenities_results: list[RetrievalItem] = Field(
        ...,
        description="Output from query_amenities",
    )
    services_results: list[RetrievalItem] = Field(
        ...,
        description="Output from query_services",
    )


class ParsedQuery(BaseModel):
    """Structured interpretation of a user request for retrieval tools.

    This model captures the core query plus optional constraints that map directly
    to the amenity/service metadata filters in Redis.
    """

    query: str = Field(
        ...,
        description=("The primary, cleaned query string capturing the user's intent."),
    )
    booking_required: bool | None = Field(
        default=None,
        description=(
            "If the user explicitly requires booking (not cases where booking required "
            "is just OK or they're fine with booking required), set True. "
            "If the user explicitly wants no booking (e.g., 'no booking', "
            "'no reservation', 'walk-in'), set False. "
            "Do not infer notice requirements from booking status."
        ),
    )
    min_price: float | None = Field(
        default=None,
        description=("Minimum acceptable price in USD (inclusive)."),
    )
    max_price: float | None = Field(
        default=None,
        description=("Maximum acceptable price in USD (inclusive)."),
    )
    min_notice_hours: int | None = Field(
        default=None,
        description=(
            "Minimum advance notice required in hours (inclusive). "
            "Only set if explicitly mentioned (e.g., 'need at least X hours notice')."
        ),
    )
    max_notice_hours: int | None = Field(
        default=None,
        description=(
            "Maximum acceptable advance notice in hours (inclusive). "
            "Only set if explicitly mentioned (e.g., 'can only give X hours notice'). "
            "Do not infer from 'walk-in' or 'no booking' - "
            "those only affect booking_required."
        ),
    )
    min_duration_minutes: int | None = Field(
        default=None,
        description=(
            "Minimum duration in minutes (inclusive). "
            "Set BOTH min and max to X when the user requests a structured service "
            "with a specific duration (e.g., 'X-minute massage/spa/class/tour'). "
            "Structured services typically have fixed durations. "
            "Only set min alone for 'at least X minutes' phrases."
        ),
    )
    max_duration_minutes: int | None = Field(
        default=None,
        description=(
            "Maximum duration in minutes (inclusive). "
            "For structured services (massage, spa, class, tour): "
            "set BOTH min and max to the same value (exact duration). "
            "For flexible services (food, quick activities) or time constraints: "
            "set ONLY max (e.g., '15-minute bite' → max=15, min=None). "
            "Use context to determine if duration is exact or a limit."
        ),
    )


class InfoState(MessagesState, total=False):
    """LangGraph state for the information DAG.

    Attributes:
        faq_results: Retrieved FAQ items.
        amenities_results: Retrieved amenity items.
        services_results: Retrieved service items.
        top_results: Reranked items across sources.

    """

    faq_results: list[RetrievalItem]
    amenities_results: list[RetrievalItem]
    services_results: list[RetrievalItem]
    top_results: list[RetrievalItem]


class ParsedState(InfoState):
    """State with required parsed query info along with extracted constraints."""

    parsed: ParsedQuery


INFO_AGENT_ENDING = (
    "Is there any other information I can provide about the hotel? "
    "I can also help finding and booking a room."
)

# ============================
# Retrieval helpers
# ============================


def build_filters(  # noqa: PLR0913
    *,
    booking_required: bool | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
    min_notice_hours: int | None = None,
    max_notice_hours: int | None = None,
    min_duration_minutes: int | None = None,
    max_duration_minutes: int | None = None,
) -> MetadataFilters | None:
    """Build metadata filters for LlamaIndex retrieval.

    All provided constraints are combined using logical AND.

    Args:
        booking_required: If provided, filters on booking_required == "True"/"False".
        min_price: Minimum price (inclusive).
        max_price: Maximum price (inclusive).
        min_notice_hours: Minimum notice hours (inclusive).
        max_notice_hours: Maximum notice hours (inclusive).
        min_duration_minutes: Minimum duration minutes (inclusive).
        max_duration_minutes: Maximum duration minutes (inclusive).

    Returns:
        MetadataFilters | None: Filter object when any constraints are provided;
        otherwise None.

    """
    filters: list[MetadataFilter | MetadataFilters] = []

    if booking_required is not None:
        filters.append(
            MetadataFilter(
                key="booking_required",
                operator=FilterOperator.EQ,
                value="True" if booking_required else "False",
            ),
        )

    if min_price is not None:
        filters.append(
            MetadataFilter(
                key="price",
                operator=FilterOperator.GTE,
                value=float(min_price),
            ),
        )
    if max_price is not None:
        filters.append(
            MetadataFilter(
                key="price",
                operator=FilterOperator.LTE,
                value=float(max_price),
            ),
        )

    if min_notice_hours is not None:
        filters.append(
            MetadataFilter(
                key="min_notice_hours",
                operator=FilterOperator.GTE,
                value=int(min_notice_hours),
            ),
        )
    if max_notice_hours is not None:
        filters.append(
            MetadataFilter(
                key="min_notice_hours",
                operator=FilterOperator.LTE,
                value=int(max_notice_hours),
            ),
        )

    if min_duration_minutes is not None:
        filters.append(
            MetadataFilter(
                key="duration",
                operator=FilterOperator.GTE,
                value=int(min_duration_minutes),
            ),
        )
    if max_duration_minutes is not None:
        filters.append(
            MetadataFilter(
                key="duration",
                operator=FilterOperator.LTE,
                value=int(max_duration_minutes),
            ),
        )

    return MetadataFilters(filters=filters) if filters else None



def build_index_schema(
    *,
    name: str,
    prefix: str,
    extra_fields: list[dict[str, Any]],
    vector_dims: int,
) -> IndexSchema:
    """Create a RedisVL index schema for a RedisVectorStore.

    Args:
        name: RedisSearch index name.
        prefix: Key prefix for documents in this index.
        extra_fields: Additional schema fields for metadata filtering.
        vector_dims: Dimensionality of the vector field.

    Returns:
        IndexSchema: RedisVL schema instance.

    """
    fields: list[dict[str, Any]] = [
        {"type": "tag", "name": "id"},
        {"type": "tag", "name": "doc_id"},
        {"type": "text", "name": "text"},
        {
            "type": "vector",
            "name": "vector",
            "attrs": {
                "dims": vector_dims,
                "algorithm": "hnsw",
                "distance_metric": "cosine",
            },
        },
        *extra_fields,
    ]

    schema_dict = {"index": {"name": name, "prefix": prefix}, "fields": fields}
    return IndexSchema.from_dict(schema_dict)


# ============================
# Shared resources (Redis + indexes)
# ============================


@dataclass(frozen=True)
class VectorIndexes:
    """Container for the three VectorStoreIndex instances.

    Attributes:
        faq: Vector index for FAQs.
        amenities: Vector index for amenities.
        services: Vector index for services.

    """

    faq: VectorStoreIndex
    amenities: VectorStoreIndex
    services: VectorStoreIndex


class InfoRagResources:
    """Shared, async-first resources used by retrieval tools.

    This class owns:
        - the async Redis client
        - LlamaIndex vector indexes + retrievers
        - a bounded cache of per-filter retrievers

    It is intended to be created once per process and reused across requests.

    """

    __slots__ = (
        "_catalog_retrievers",
        "_config",
        "_embed_batch_size",
        "_faq_retriever",
        "_retriever_cache_max",
        "_system_prompt_resource",
        "_top_k",
        "_vector_dims",
        "indexes",
        "redis_async",
        "system_prompt",
    )

    _config: InfoRagConfig
    _top_k: int
    _vector_dims: int
    _embed_batch_size: int
    _retriever_cache_max: int
    redis_async: AsyncRedis
    indexes: VectorIndexes
    _faq_retriever: BaseRetriever
    _catalog_retrievers: OrderedDict[
        tuple[Source, tuple[tuple[str, str, str], ...]],
        VectorIndexRetriever,
    ]
    _system_prompt_resource: str
    system_prompt: str | None

    def __init__(
        self,
        *,
        redis_url: str,
        config: InfoRagConfig,
    ) -> None:
        """Initialize embedding settings, Redis client, and indexes.

        Note: This constructor is synchronous; call ``await startup_check()`` during
        FastAPI startup to validate connectivity and index schema.

        Args:
            redis_url: Redis connection URL.
            config: Parsed TOML configuration.

        """
        self._config = config
        self._top_k = int(config.retrieval.top_k)
        self._vector_dims = int(config.retrieval.vector_dims)
        self._embed_batch_size = int(config.embeddings.batch_size)
        self._retriever_cache_max = max(1, int(config.retrieval.retriever_cache_max))

        self._init_llamaindex(config.embeddings)

        retry = self._build_redis_retry(config.redis)
        self.redis_async: AsyncRedis = AsyncRedis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=float(config.redis.connect_timeout_s),
            socket_timeout=float(config.redis.socket_timeout_s),
            health_check_interval=int(config.redis.health_check_interval_s),
            retry=retry,
        )

        self.indexes = self._build_indexes(
            self.redis_async,
            vector_dims=self._vector_dims,
        )

        self._faq_retriever = self.indexes.faq.as_retriever(
            similarity_top_k=self._top_k,
        )

        self._catalog_retrievers: OrderedDict[
            tuple[Source, tuple[tuple[str, str, str], ...]],
            VectorIndexRetriever,
        ] = OrderedDict()

        prompts_folder = config.prompts.folder.strip("/")
        if prompts_folder:
            self._system_prompt_resource = (
                f"{prompts_folder}/{config.prompts.system_prompt_filename}"
            )
        else:
            self._system_prompt_resource = config.prompts.system_prompt_filename
        self.system_prompt: str | None = None

    @staticmethod
    def _build_redis_retry(config: InfoRedisConfig) -> Retry:
        """Build a Redis retry policy from configuration.

        Args:
            config: Redis configuration section.

        Returns:
            Retry: Configured retry policy for transient Redis errors.

        """
        backoff = ExponentialBackoff(
            base=float(config.retry_backoff_base_s),
            cap=float(config.retry_backoff_max_s),
        )
        return Retry(
            backoff=backoff,
            retries=int(config.retry_max_retries),
            supported_errors=(
                RedisConnectionError,
                RedisTimeoutError,
                TimeoutError,
                BusyLoadingError,
            ),
        )

    async def startup_check(self) -> None:
        """Validate Redis connectivity, retriever capability, and index schema.

        Raises:
            OperationalError: If Redis is unreachable, retrievers do not support
                async retrieval, or the index schema does not match configuration.

        """
        try:
            await self.redis_async.ping()
        except Exception as exc:
            msg = "Redis ping failed"
            raise OperationalError(msg) from exc

        if not hasattr(self._faq_retriever, "aretrieve"):
            msg = "FAQ retriever does not support aretrieve()"
            raise OperationalError(msg)

        _ = self._get_catalog_retriever(source=Source.AMENITIES, filters=None)
        await self._validate_vector_dims()

        try:
            self.system_prompt = await asyncio.to_thread(self._render_system_prompt)
        except Exception as exc:
            msg = "Failed to render system prompt"
            raise OperationalError(msg) from exc

    async def aclose(self) -> None:
        """Close network resources owned by this instance.

        This method should be called during FastAPI shutdown to ensure Redis
        connections are properly released.

        """
        try:
            await self.redis_async.close()
        finally:
            await self.redis_async.connection_pool.disconnect()  # type: ignore[misc]

    def _render_system_prompt(self) -> str:
        """Render and return the system prompt string.

        Returns:
            str: Rendered system prompt.

        Raises:
            RuntimeError: If prompt resolution or template loading fails.

        """
        return build_system_prompt(
            top_k=self._top_k,
            prompt_resource=self._system_prompt_resource,
        )

    async def _validate_vector_dims(self) -> None:
        """Validate that Redis vector index dimensions match the configured value.

        Raises:
            OperationalError: If any index reports a vector dimension that does not
                match the configured value.

        """
        for src in (Source.FAQ, Source.AMENITIES, Source.SERVICES):
            expected = self._vector_dims
            actual = await self._get_index_vector_dims(src.value)
            if actual != expected:
                msg = (
                    f"Index '{src.value}' vector dims mismatch: "
                    f"expected={expected} actual={actual}"
                )
                raise OperationalError(msg)

    async def _get_index_vector_dims(self, index_name: str) -> int:
        """Extract vector dims for the 'vector' field from FT.INFO.

        Args:
            index_name: RedisSearch index name.

        Returns:
            int: Vector dimensionality for the "vector" field.

        Raises:
            OperationalError: If FT.INFO fails or the response is missing required
                fields.

        """
        try:
            reply = await self.redis_async.execute_command("FT.INFO", index_name)
        except Exception as exc:
            msg = f"FT.INFO failed for index '{index_name}'"
            raise OperationalError(msg) from exc

        info = self._redis_kv_list_to_dict(reply)
        attributes = info.get("attributes") or info.get("fields")
        if not isinstance(attributes, list):
            msg = f"FT.INFO for '{index_name}' missing attributes/fields section"
            raise OperationalError(msg)

        for attr in attributes:
            attr_dict = self._redis_kv_list_to_dict(attr)
            name = (
                attr_dict.get("attribute")
                or attr_dict.get("identifier")
                or attr_dict.get("name")
            )
            if str(name).lower() != "vector":
                continue

            dims = attr_dict.get("dims") or attr_dict.get("dim")
            if dims is None:
                for v in attr_dict.values():
                    nested = self._redis_kv_list_to_dict(v)
                    dims = nested.get("dims") or nested.get("dim")
                    if dims is not None:
                        break

            if dims is None:
                msg = f"FT.INFO for '{index_name}' has vector field but no dims"
                raise OperationalError(msg)

            try:
                return int(dims)
            except (TypeError, ValueError) as exc:
                msg = f"FT.INFO for '{index_name}' returned non-int dims: {dims!r}"
                raise OperationalError(msg) from exc

        msg = (
            f"FT.INFO for '{index_name}' did not include a vector field named 'vector'"
        )
        raise OperationalError(msg)

    @staticmethod
    def _redis_kv_list_to_dict(reply: object | None) -> dict[str, Any]:
        """Convert Redis module replies (flat [k, v, k, v, ...]) to dict.

        Args:
            reply: Redis module response (mapping-like or flat sequence) or None.

        Returns:
            dict[str, Any]: Lowercased-key dictionary representation.

        """
        if reply is None:
            return {}
        if isinstance(reply, Mapping):
            return {str(k).lower(): v for k, v in reply.items()}
        if not isinstance(reply, Sequence) or isinstance(
            reply,
            (str, bytes, bytearray),
        ):
            return {}

        out: dict[str, Any] = {}
        it = iter(reply)
        for k in it:
            try:
                v = next(it)
            except StopIteration:
                break
            out[str(k).lower()] = v
        return out

    @staticmethod
    def _init_llamaindex(cfg: InfoEmbeddingsConfig) -> None:
        """Configure LlamaIndex global embedding settings.

        Side Effects:
            Updates the global ``llama_index.core.Settings.embed_model`` for the
            current process.

        Args:
            cfg: Embedding configuration.

        """
        Settings.embed_model = OpenAIEmbedding(
            model=cfg.model,
            embed_batch_size=int(cfg.batch_size),
            timeout=float(cfg.timeout_s),
            max_retries=int(cfg.max_retries),
        )

    @staticmethod
    def _build_indexes(
        redis_client_async: AsyncRedis,
        *,
        vector_dims: int,
    ) -> VectorIndexes:
        """Build VectorStoreIndex instances backed by RedisVectorStore.

        Args:
            redis_client_async: Async Redis client used by RedisVectorStore.
            vector_dims: Vector dimension for the schema.

        Returns:
            VectorIndexes: Container with all three indexes.

        """
        faq_schema = build_index_schema(
            name=Source.FAQ.value,
            prefix=Source.FAQ.value,
            extra_fields=[{"type": "tag", "name": "category"}],
            vector_dims=vector_dims,
        )
        faq_store = RedisVectorStore(
            schema=faq_schema,
            redis_client_async=redis_client_async,
            overwrite=False,
        )
        faq_storage = StorageContext.from_defaults(vector_store=faq_store)
        faq_index = VectorStoreIndex.from_vector_store(
            vector_store=faq_store,
            storage_context=faq_storage,
        )

        amenities_schema = build_index_schema(
            name=Source.AMENITIES.value,
            prefix=Source.AMENITIES.value,
            extra_fields=[
                {"type": "tag", "name": "category"},
                {"type": "numeric", "name": "price"},
                {"type": "numeric", "name": "duration"},
                {"type": "numeric", "name": "min_notice_hours"},
                {"type": "tag", "name": "booking_required"},
            ],
            vector_dims=vector_dims,
        )
        amenities_store = RedisVectorStore(
            schema=amenities_schema,
            redis_client_async=redis_client_async,
            overwrite=False,
        )
        amenities_storage = StorageContext.from_defaults(vector_store=amenities_store)
        amenities_index = VectorStoreIndex.from_vector_store(
            vector_store=amenities_store,
            storage_context=amenities_storage,
        )

        services_schema = build_index_schema(
            name=Source.SERVICES.value,
            prefix=Source.SERVICES.value,
            extra_fields=[
                {"type": "tag", "name": "service_type"},
                {"type": "numeric", "name": "price"},
                {"type": "numeric", "name": "duration"},
                {"type": "numeric", "name": "min_notice_hours"},
                {"type": "tag", "name": "department"},
                {"type": "tag", "name": "booking_required"},
            ],
            vector_dims=vector_dims,
        )
        services_store = RedisVectorStore(
            schema=services_schema,
            redis_client_async=redis_client_async,
            overwrite=False,
        )
        services_storage = StorageContext.from_defaults(vector_store=services_store)
        services_index = VectorStoreIndex.from_vector_store(
            vector_store=services_store,
            storage_context=services_storage,
        )

        return VectorIndexes(
            faq=faq_index,
            amenities=amenities_index,
            services=services_index,
        )

    @property
    def top_k(self) -> int:
        """Return the configured retrieval top-k.

        Returns:
            int: Top-k value used for retrieval.

        """
        return self._top_k

    def get_system_prompt(self) -> str:
        """Return the rendered system prompt.

        The system prompt is rendered during ``startup_check()`` and cached on the
        instance. Callers should use this getter rather than accessing the internal
        attribute directly.

        Returns:
            str: Rendered system prompt.

        Raises:
            RuntimeError: If the system prompt has not been rendered yet.

        """
        if self.system_prompt is None:
            msg = (
                "System prompt is not initialized; call "
                "await InfoRagResources.startup_check() first"
            )
            raise RuntimeError(msg)
        return self.system_prompt

    async def retrieve_faq(self, query: str) -> list[RetrievalItem]:
        """Retrieve FAQ nodes relevant to a query.

        Args:
            query: Natural-language query.

        Returns:
            list[RetrievalItem]: FAQ matches with text and metadata.

        Raises:
            OperationalError: If retrieval fails.

        """
        try:
            nodes = await self._faq_retriever.aretrieve(query)
        except Exception as exc:
            msg = "FAQ retrieval failed"
            raise OperationalError(msg) from exc

        return [
            RetrievalItem(
                source=Source.FAQ,
                metadata=getattr(n.node, "metadata", None) or {},
                text=getattr(n.node, "text", "") or "",
                score=self._coerce_score(n),
            )
            for n in nodes
        ]

    async def retrieve_filtered_catalog_items(
        self,
        *,
        source: Source,
        query: str,
        filters: MetadataFilters | None,
    ) -> list[RetrievalItem]:
        """Retrieve amenity/service nodes for a query with optional metadata filters.

        Args:
            source: Source.AMENITIES or Source.SERVICES.
            query: Natural-language query.
            filters: Optional metadata filters.

        Returns:
            list[RetrievalItem]: Matches with text and metadata for the given source.

        Raises:
            OperationalError: If retrieval fails.

        """
        retriever = self._get_catalog_retriever(source=source, filters=filters)
        try:
            nodes = await retriever.aretrieve(query)
        except Exception as exc:
            msg = f"{source} retrieval failed"
            raise OperationalError(msg) from exc

        return [
            RetrievalItem(
                source=source,
                metadata=getattr(n.node, "metadata", None) or {},
                text=getattr(n.node, "text", "") or "",
                score=self._coerce_score(n),
            )
            for n in nodes
        ]

    def _get_catalog_retriever(
        self,
        *,
        source: Source,
        filters: MetadataFilters | None,
    ) -> VectorIndexRetriever:
        """Return a cached VectorIndexRetriever for amenities/services.

        Args:
            source: Source.AMENITIES or Source.SERVICES.
            filters: Optional metadata filters.

        Returns:
            VectorIndexRetriever: Cached or newly-created retriever.

        Raises:
            OperationalError: If the retriever does not support async retrieval.

        """
        sig = self._filters_signature(filters)
        cache_key = (source, sig)

        existing = self._catalog_retrievers.get(cache_key)
        if existing is not None:
            self._catalog_retrievers.move_to_end(cache_key)
            return existing

        index = {
            Source.AMENITIES: self.indexes.amenities,
            Source.SERVICES: self.indexes.services,
        }[source]

        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=self._top_k,
            filters=filters,
        )

        if not hasattr(retriever, "aretrieve"):
            msg = "VectorIndexRetriever does not support aretrieve()"
            raise OperationalError(msg)

        self._cache_retriever(cache_key, retriever)
        return retriever

    @staticmethod
    def _filters_signature(
        filters: MetadataFilters | None,
    ) -> tuple[tuple[str, str, str], ...]:
        """Create a stable cache key for metadata filters.

        Args:
            filters: Optional metadata filters.

        Returns:
            tuple[tuple[str, str, str], ...]: Order-independent signature for caching.

        """
        if not filters:
            return ()

        signature: list[tuple[str, str, str]] = []
        for f in getattr(filters, "filters", []) or []:
            key = str(getattr(f, "key", ""))
            op = str(getattr(f, "operator", ""))
            val = str(getattr(f, "value", ""))
            signature.append((key, op, val))

        signature.sort()
        return tuple(signature)

    def _cache_retriever(
        self,
        cache_key: tuple[Source, tuple[tuple[str, str, str], ...]],
        retriever: VectorIndexRetriever,
    ) -> None:
        """Insert a retriever into the bounded LRU cache.

        Args:
            cache_key: Cache key (source, filter signature).
            retriever: Retriever instance to cache.

        """
        self._catalog_retrievers[cache_key] = retriever
        self._catalog_retrievers.move_to_end(cache_key)
        while len(self._catalog_retrievers) > self._retriever_cache_max:
            self._catalog_retrievers.popitem(last=False)

    @staticmethod
    def _coerce_score(node: Any) -> float:  # noqa: ANN401
        """Convert a retrieved node score to a safe float.

        Args:
            node: Retrieved node object (may or may not define a score).

        Returns:
            float: A finite float score; returns 0.0 for missing, invalid, NaN, or
            infinite scores.

        """
        raw = getattr(node, "score", None)
        if raw is None:
            return 0.0
        try:
            score = float(raw)
        except (TypeError, ValueError):
            return 0.0
        if math.isnan(score) or math.isinf(score):
            return 0.0
        return score


# ============================
# Agent construction
# ============================


def _latest_user_text(messages: Sequence[BaseMessage]) -> str:
    """Return the most recent user message content as a string.

    Args:
        messages: Ordered conversation messages to scan from newest to oldest.

    Returns:
        Most recent user message content, or an empty string if none exists.

    """
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def _build_context_block(items: Sequence[RetrievalItem]) -> str:
    """Build a plain-text context block from hydrated items.

    Args:
        items: Hydrated retrieval items to serialize into a context block.

    Returns:
        Context block string with one section per item.

    """
    lines = [
        "\n".join(
            [
                f"source={item.source}",
                f"score={item.score}",
                f"text={item.text}",
                f"metadata={item.metadata}",
            ],
        )
        for item in items
    ]
    return "\n\n".join(lines)


class InfoAgentFactory:
    """Factory for constructing the info retrieval DAG.

    This factory wires together:
        - configuration
        - shared async retrieval resources
        - prompt templates

    The resulting agent performs a single retrieval round in a fixed DAG.

    """

    __slots__ = ("_config", "_resources")

    _resources: InfoRagResources
    _config: InfoRagConfig

    def __init__(
        self,
        *,
        resources: InfoRagResources,
        config: InfoRagConfig,
    ) -> None:
        """Initialize the agent factory.

        Args:
            resources: Shared async resources (Redis client, indexes, retrievers).
            config: Parsed TOML configuration.

        """
        self._resources = resources
        self._config = config

    def build(self) -> CompiledStateGraph:  # noqa: C901, PLR0915
        """Build and return a compiled agent graph.

        Returns:
            CompiledStateGraph: A compiled LangGraph DAG for info retrieval.

        """
        resources = self._resources
        top_k = resources.top_k

        llm_cfg = self._config.llm
        parser_llm = ChatOpenAI(
            model=str(llm_cfg.model),
            temperature=0.0,
            timeout=float(llm_cfg.timeout_s),
            max_retries=int(llm_cfg.max_retries),
            reasoning={"effort": llm_cfg.reasoning_effort},
        ).with_structured_output(ParsedQuery)
        responder_llm = ChatOpenAI(
            model=str(llm_cfg.model),
            temperature=float(llm_cfg.temperature),
            timeout=float(llm_cfg.timeout_s),
            max_retries=int(llm_cfg.max_retries),
            reasoning={"effort": llm_cfg.reasoning_effort},
        )

        def _log_node_failure(node_name: str, exc: Exception) -> None:
            """Log a node failure with traceback."""
            logger.exception("Info DAG node %s failed: %s", node_name, exc)

        def _make_catalog_query_node(
            source: Source,
            state_key: str,
        ) -> Callable[..., Awaitable[dict[str, Any]]]:
            """Create a query node for a filtered catalog source.

            Args:
                source: Which catalog source to query (AMENITIES or SERVICES).
                state_key: State dict key for the results (e.g. "amenities_results").

            Returns:
                Async node function suitable for LangGraph.

            """
            tool_name = f"query_{source.value}"

            async def _node(state: ParsedState) -> dict[str, Any]:
                parsed = state["parsed"]
                filters = build_filters(
                    booking_required=parsed.booking_required,
                    min_price=parsed.min_price,
                    max_price=parsed.max_price,
                    min_notice_hours=parsed.min_notice_hours,
                    max_notice_hours=parsed.max_notice_hours,
                    min_duration_minutes=parsed.min_duration_minutes,
                    max_duration_minutes=parsed.max_duration_minutes,
                )
                try:
                    results = await resources.retrieve_filtered_catalog_items(
                        source=source,
                        query=parsed.query,
                        filters=filters,
                    )
                except OperationalError as exc:
                    logger.warning("%s operational failure: %s", tool_name, exc)
                    return {state_key: []}
                except Exception as exc:  # noqa: BLE001
                    _log_node_failure(tool_name, exc)
                    return {state_key: []}
                return {state_key: results}

            _node.__name__ = _node.__qualname__ = tool_name + "_node"
            _node.__doc__ = (
                f"Retrieve {source.value} for the parsed query and constraints."
            )
            return _node

        async def parse_node(state: InfoState) -> dict[str, Any]:
            """Parse the user request into a structured query and constraints."""
            try:
                parsed = cast(
                    "ParsedQuery",
                    await parser_llm.ainvoke(
                        [
                            SystemMessage(
                                content=(
                                    "Extract a single query string plus any "
                                    "constraints (booking, price, notice, duration) "
                                    "from the full conversation. Prefer the latest "
                                    "user request.\n\n"
                                    "Note: Service and amenity descriptions often "
                                    "include duration information. Do not extract "
                                    "these as user duration constraints unless the "
                                    "user explicitly requests a specific duration."
                                ),
                            ),
                            *state["messages"],
                        ],
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                _log_node_failure("parse", exc)
                parsed = ParsedQuery(query=_latest_user_text(state["messages"]))

            return {"parsed": parsed}

        async def query_faq_node(state: ParsedState) -> dict[str, Any]:
            """Retrieve FAQ entries for the parsed query.

            Args:
                state: Parsed state containing the query text.

            Returns:
                State patch with ``faq_results`` populated.

            """
            query = state["parsed"].query
            try:
                results = await resources.retrieve_faq(query)
            except OperationalError as exc:
                logger.warning("query_faq operational failure: %s", exc)
                return {"faq_results": []}
            except Exception as exc:  # noqa: BLE001
                _log_node_failure("query_faq", exc)
                return {"faq_results": []}
            return {"faq_results": results}

        query_amenities_node = _make_catalog_query_node(
            Source.AMENITIES,
            "amenities_results",
        )
        query_services_node = _make_catalog_query_node(
            Source.SERVICES,
            "services_results",
        )

        def rerank_node(state: InfoState) -> dict[str, Any]:
            """Select the top-k results across FAQ, amenities, and services.

            Args:
                state: Current state with retrieval results.

            Returns:
                State patch with ``top_results`` populated.

            """
            all_results = [
                *state.get("faq_results", []),
                *state.get("amenities_results", []),
                *state.get("services_results", []),
            ]
            ranked = heapq.nlargest(top_k, all_results, key=lambda x: x.score)
            return {"top_results": ranked}

        async def respond_node(state: InfoState) -> dict[str, Any]:
            """Generate the final response using retrieved context.

            Args:
                state: Current state with reranked results.

            Returns:
                State patch with the LLM response appended to messages.

            """
            top_items = state.get("top_results", [])
            context_block = _build_context_block(top_items)
            system_prompt = resources.get_system_prompt()
            try:
                response = await responder_llm.ainvoke(
                    [
                        SystemMessage(
                            content=(f"{system_prompt}{context_block}"),
                        ),
                        *state["messages"],
                    ],
                )
            except Exception as exc:
                _log_node_failure("respond", exc)
                raise
            return {"messages": [response]}

        graph = StateGraph(InfoState)
        graph.add_node("parse", parse_node)
        graph.add_node("query_faq", query_faq_node)
        graph.add_node("query_amenities", query_amenities_node)
        graph.add_node("query_services", query_services_node)
        graph.add_node("rerank", rerank_node)
        graph.add_node("respond", respond_node)

        graph.add_edge(START, "parse")
        graph.add_edge("parse", "query_faq")
        graph.add_edge("parse", "query_amenities")
        graph.add_edge("parse", "query_services")
        graph.add_edge("query_faq", "rerank")
        graph.add_edge("query_amenities", "rerank")
        graph.add_edge("query_services", "rerank")
        graph.add_edge("rerank", "respond")
        graph.add_edge("respond", END)

        return graph.compile()
