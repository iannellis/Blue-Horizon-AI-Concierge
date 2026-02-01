"""Hotel info RAG agent (LangChain create_agent + LangGraph) with RedisVectorStore.

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
import json
import logging
import math
import os
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from contextvars import ContextVar, Token
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
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


@dataclass
class EvalInfoToolLog:
    """Capture tool activity and contexts for evaluation runs.

    Attributes:
        tool_summary: Compact tool summaries for LangSmith eval.
        contexts_used: Hydrated context strings used in responses.

    """

    tool_summary: list[dict[str, Any]]
    contexts_used: list[str]


_EVAL_INFO_TOOL_LOG: ContextVar[EvalInfoToolLog | None] = ContextVar(
    "EVAL_INFO_TOOL_LOG",
    default=None,
)


def set_eval_info_tool_log(log: EvalInfoToolLog) -> Token[EvalInfoToolLog | None]:
    """Set the per-task evaluation tool log container.

    Args:
        log: Log container to use for the current task.

    Returns:
        Token used to reset the ContextVar to its prior state.

    """
    return _EVAL_INFO_TOOL_LOG.set(log)


def reset_eval_info_tool_log(token: Token[EvalInfoToolLog | None]) -> None:
    """Reset the evaluation tool log ContextVar.

    Args:
        token: Token returned by ``set_eval_info_tool_log``.

    """
    _EVAL_INFO_TOOL_LOG.reset(token)


def get_eval_info_tool_log() -> EvalInfoToolLog | None:
    """Return the active evaluation tool log container, if any.

    Returns:
        EvalInfoToolLog when set, otherwise ``None``.

    """
    return _EVAL_INFO_TOOL_LOG.get()


def _log_eval_tool_summary(summary: dict[str, Any]) -> None:
    """Append a tool summary entry when evaluation logging is enabled.

    Args:
        summary: Tool summary payload to append.

    """
    log = get_eval_info_tool_log()
    if log is None:
        return
    log.tool_summary.append(summary)


def _log_eval_contexts(contexts: Iterable[str]) -> None:
    """Append hydrated contexts when evaluation logging is enabled.

    Args:
        contexts: Iterable of context strings.

    """
    log = get_eval_info_tool_log()
    if log is None:
        return
    for context in contexts:
        if context:
            log.contexts_used.append(str(context))


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
    """Render the system prompt with runtime substitutions."""
    template = load_prompt_template(prompt_resource)
    return template.safe_substitute(top_k=top_k)


# ============================
# Environment
# ============================


@lru_cache(maxsize=1)
def get_redis_url() -> str:
    """Return the Redis connection URL from environment.

    This function loads environment variables via ``python-dotenv`` and returns the
    ``REDIS_URL`` value.

    Returns:
        str: Redis connection URL.

    Raises:
        RuntimeError: If ``REDIS_URL`` is not set.

    """
    load_dotenv()
    url = os.getenv("REDIS_URL")
    if not url:
        msg = "REDIS_URL is not set"
        raise RuntimeError(msg)
    return url


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


class RetrievalItemLite(BaseModel):
    """Lightweight retrieval result.

    This model is used for cross-source reranking. The actual text and metadata are
    fetched later during hydration.

    Attributes:
        source: Origin source of the item (faq/amenities/services).
        item_id: Stable identifier used to hydrate the item from Redis.
        score: Similarity score used for ranking.

    """

    source: Source = Field(..., description="faq | amenities | services")
    item_id: str = Field(..., description="Stable identifier for this item")
    score: float = Field(..., description="Similarity score; higher is more relevant")


class RetrievalItem(BaseModel):
    """Hydrated retrieval result containing text and metadata.

    Attributes:
        source: Origin source of the item.
        metadata: Metadata payload associated with the node.
        text: Human-readable text for the node.
        score: Similarity score used for ranking.

    """

    source: Source = Field(..., description="Which retriever produced this item")
    metadata: dict[str, Any] = Field(
        ..., description="Metadata associated with the text",
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

    faq_results: list[RetrievalItemLite] = Field(
        ..., description="Output from query_faq",
    )
    amenities_results: list[RetrievalItemLite] = Field(
        ...,
        description="Output from query_amenities",
    )
    services_results: list[RetrievalItemLite] = Field(
        ...,
        description="Output from query_services",
    )


class HydrateInput(BaseModel):
    """Input schema for the hydrate_items tool.

    Attributes:
        items: Top items selected by the reranker.

    """

    items: list[RetrievalItemLite] = Field(..., description="Top items from reranker")


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
                key="price", operator=FilterOperator.GTE, value=float(min_price),
            ),
        )
    if max_price is not None:
        filters.append(
            MetadataFilter(
                key="price", operator=FilterOperator.LTE, value=float(max_price),
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
            self.redis_async.connection_pool.disconnect()

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
            reply, (str, bytes, bytearray),
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

    async def retrieve_faq(self, query: str) -> list[RetrievalItemLite]:
        """Retrieve FAQ nodes relevant to a query.

        Args:
            query: Natural-language query.

        Returns:
            list[RetrievalItemLite]: Lightweight FAQ matches.

        Raises:
            OperationalError: If retrieval fails.

        """
        try:
            nodes = await self._faq_retriever.aretrieve(query)
        except Exception as exc:
            msg = "FAQ retrieval failed"
            raise OperationalError(msg) from exc

        return [
            RetrievalItemLite(
                source=Source.FAQ,
                item_id=getattr(n, "id_", ""),
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
    ) -> list[RetrievalItemLite]:
        """Retrieve amenity/service nodes for a query with optional metadata filters.

        Args:
            source: Source.AMENITIES or Source.SERVICES.
            query: Natural-language query.
            filters: Optional metadata filters.

        Returns:
            list[RetrievalItemLite]: Lightweight matches for the given source.

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
            RetrievalItemLite(
                source=source,
                item_id=getattr(n, "id_", ""),
                score=self._coerce_score(n),
            )
            for n in nodes
        ]

    def _build_hydration_pipeline(
        self,
        items: list[RetrievalItemLite],
    ) -> tuple[Any, list[str]]:
        """Build a Redis pipeline to fetch hydration fields for a list of items.

        Args:
            items: Items to hydrate.

        Returns:
            tuple[Any, list[str]]: (pipeline, redis_keys) in matching order.

        """
        pipe = self.redis_async.pipeline(transaction=False)
        keys: list[str] = []
        for item in items:
            key = f"{item.source}:{item.item_id}"
            keys.append(key)
            pipe.hmget(key, ["_node_content", "text"])
        return pipe, keys

    def _parse_hmget_result(
        self,
        *,
        item: RetrievalItemLite,
        key: str,
        result: object,
        best_effort: bool,
    ) -> tuple[str, str] | None:
        """Parse a single HMGET result into (_node_content, text).

        Args:
            item: Item being hydrated.
            key: Redis key used for hydration.
            result: Raw HMGET result.
            best_effort: If True, return None on malformed/missing results.

        Returns:
            tuple[str, str] | None: (node_content_json, text) or None if best_effort
            skips this item.

        Raises:
            OperationalError: If best_effort is False and the result is malformed.

        """
        try:
            node_content_str, text = result  # type: ignore[misc]
        except Exception as exc:
            if best_effort:
                logger.warning("Hydration malformed result for key=%s", key)
                return None
            msg = f"Hydration malformed result for key={key}"
            raise OperationalError(msg) from exc

        if not node_content_str:
            if best_effort:
                logger.warning(
                    "Missing _node_content for key=%s (source=%s item_id=%s)",
                    key,
                    item.source,
                    item.item_id,
                )
                return None
            msg = (
                f"Could not hydrate item_id={item.item_id} "
                f"from source={item.source} (key={key})"
            )
            raise OperationalError(msg)

        return str(node_content_str), str(text or "")

    def _decode_node_content(
        self,
        *,
        node_content_str: str,
        key: str,
        best_effort: bool,
    ) -> dict[str, Any] | None:
        """Decode the JSON stored in _node_content.

        Args:
            node_content_str: JSON string stored under the _node_content field.
            key: Redis key used for logging and error messages.
            best_effort: If True, return None on invalid payloads.

        Returns:
            dict[str, Any] | None: Parsed JSON dict or None if best_effort skips.

        Raises:
            OperationalError: If best_effort is False and JSON decoding fails.

        """
        try:
            node_content = json.loads(node_content_str)
        except json.JSONDecodeError as exc:
            if best_effort:
                logger.warning("Invalid _node_content JSON for key=%s", key)
                return None
            msg = f"Invalid _node_content JSON for key={key}"
            raise OperationalError(msg) from exc

        if not isinstance(node_content, dict):
            if best_effort:
                logger.warning("Unexpected _node_content type for key=%s", key)
                return None
            msg = f"Unexpected _node_content type for key={key}"
            raise OperationalError(msg)

        return node_content

    @staticmethod
    def _to_hydrated_item(
        *,
        item: RetrievalItemLite,
        node_content: dict[str, Any],
        text: str,
    ) -> RetrievalItem:
        """Convert parsed Redis payload to a hydrated RetrievalItem.

        Args:
            item: Lightweight item containing source/id/score.
            node_content: Parsed JSON dict from _node_content.
            text: Text field fetched from Redis.

        Returns:
            RetrievalItem: Hydrated result.

        """
        metadata = node_content.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        return RetrievalItem(
            source=item.source,
            metadata=metadata,
            text=text,
            score=item.score,
        )

    async def hydrate(
        self,
        items: Iterable[RetrievalItemLite],
        *,
        best_effort: bool = True,
    ) -> list[RetrievalItem]:
        """Hydrate lite results into full objects by fetching Redis document fields.

        This method fetches the Redis fields "_node_content" and "text" for each item
        and converts them into ``RetrievalItem`` instances.

        Args:
            items: Lightweight items selected by the reranker.
            best_effort: If True, logs and skips malformed/missing items instead of
                raising.

        Returns:
            list[RetrievalItem]: Hydrated items.

        Raises:
            OperationalError: If best_effort is False and hydration fails.

        """
        item_list = list(items)
        if not item_list:
            return []

        pipe, keys = self._build_hydration_pipeline(item_list)

        try:
            results = await pipe.execute()
        except Exception as exc:
            if best_effort:
                logger.exception("Hydration pipeline failed")
                return []
            msg = "Hydration pipeline failed"
            raise OperationalError(msg) from exc

        hydrated: list[RetrievalItem] = []
        for item, key, result in zip(item_list, keys, results, strict=True):
            parsed = self._parse_hmget_result(
                item=item,
                key=key,
                result=result,
                best_effort=best_effort,
            )
            if parsed is None:
                continue

            node_content_str, text = parsed
            node_content = self._decode_node_content(
                node_content_str=node_content_str,
                key=key,
                best_effort=best_effort,
            )
            if node_content is None:
                continue

            hydrated.append(
                self._to_hydrated_item(item=item, node_content=node_content, text=text),
            )

        return hydrated


# ============================
# Agent construction
# ============================


class InfoAgentFactory:
    """Factory for constructing the LangChain agent with bound async tools.

    This factory wires together:
        - configuration
        - shared async retrieval resources
        - prompt templates

    The resulting agent uses tool calls for retrieval/reranking/hydration.

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
            CompiledStateGraph: A compiled agent graph created by LangChain.

        """
        resources = self._resources
        top_k = resources.top_k

        llm_cfg = self._config.llm
        llm = ChatOpenAI(
            model=str(llm_cfg.model),
            temperature=float(llm_cfg.temperature),
            timeout=float(llm_cfg.timeout_s),
            max_retries=int(llm_cfg.max_retries),
            reasoning={"effort": llm_cfg.reasoning_effort},
        )

        def _log_tool_failure(tool_name: str, exc: Exception) -> None:
            """Log a tool failure with traceback.

            Args:
                tool_name: Name of the tool that failed.
                exc: Exception raised by the tool.

            """
            logger.exception("Tool %s failed: %s", tool_name, exc)

        @tool(parse_docstring=True)
        async def query_faq(query: str) -> list[RetrievalItemLite]:
            """Retrieve FAQ entries relevant to a user query.

            This tool performs vector search over the **FAQ** knowledge base. Use it
            when the user asks about hotel policies, rules, hours, check-in/check-out,
            deposits, cancellations, parking, pet policy, accessibility, or other
            "how does the hotel work" questions.

            Notes:
                - This tool returns lightweight items (source, item_id, score). The
                  system will later rerank across sources and then hydrate the final
                  selection to retrieve full text + metadata.
                - If you are unsure whether a question is policy-related, it is still
                  safe to call this tool; irrelevant FAQ hits will be dropped later.

            Args:
                query: Natural-language query describing what the user wants.

            Returns:
                list[RetrievalItemLite]: Up to ``top_k`` lightweight FAQ matches.

            Examples:
                - "What time is check-in?"
                - "Do you allow pets?"
                - "Is there parking and how much does it cost?"
                - "What is the cancellation policy?"

            """
            try:
                results = await resources.retrieve_faq(query)
            except OperationalError as exc:
                logger.warning("query_faq operational failure: %s", exc)
                _log_eval_tool_summary(
                    {
                        "tool": "query_faq",
                        "status": "error",
                        "error": str(exc),
                    },
                )
                return []
            except Exception as exc:  # noqa: BLE001
                _log_tool_failure("query_faq", exc)
                _log_eval_tool_summary(
                    {
                        "tool": "query_faq",
                        "status": "error",
                        "error": str(exc),
                    },
                )
                return []
            else:
                _log_eval_tool_summary(
                    {
                        "tool": "query_faq",
                        "status": "ok",
                        "count": len(results),
                    },
                )
                return results

        @tool(parse_docstring=True)
        async def query_amenities(  # noqa: PLR0913
            query: str,
            booking_required: bool | None = None,  # noqa: FBT001
            min_price: float | None = None,
            max_price: float | None = None,
            min_notice_hours: int | None = None,
            max_notice_hours: int | None = None,
            min_duration_minutes: int | None = None,
            max_duration_minutes: int | None = None,
        ) -> list[RetrievalItemLite]:
            """Retrieve hotel amenities relevant to a user query.

            This tool performs vector search over the **amenities** catalog and can
            apply optional metadata constraints. All provided constraints are combined
            using logical **AND**.

            Use this tool when the user asks about amenities such as facilities,
            on-property features, classes, rentals, or other amenity offerings.

            Args:
                query: Natural-language query describing what the user wants.
                booking_required: Restrict results to booking_required==True/False.
                min_price: Minimum price in USD (inclusive).
                max_price: Maximum price in USD (inclusive).
                min_notice_hours: Minimum advance notice in hours (inclusive).
                max_notice_hours: Maximum advance notice in hours (inclusive).
                min_duration_minutes: Minimum duration in minutes (inclusive).
                max_duration_minutes: Maximum duration in minutes (inclusive).

            Returns:
                list[RetrievalItemLite]: Up to ``top_k`` lightweight results.

            """
            filters = build_filters(
                booking_required=booking_required,
                min_price=min_price,
                max_price=max_price,
                min_notice_hours=min_notice_hours,
                max_notice_hours=max_notice_hours,
                min_duration_minutes=min_duration_minutes,
                max_duration_minutes=max_duration_minutes,
            )
            try:
                results = await resources.retrieve_filtered_catalog_items(
                    source=Source.AMENITIES,
                    query=query,
                    filters=filters,
                )
            except OperationalError as exc:
                logger.warning("query_amenities operational failure: %s", exc)
                _log_eval_tool_summary(
                    {
                        "tool": "query_amenities",
                        "status": "error",
                        "error": str(exc),
                    },
                )
                return []
            except Exception as exc:  # noqa: BLE001
                _log_tool_failure("query_amenities", exc)
                _log_eval_tool_summary(
                    {
                        "tool": "query_amenities",
                        "status": "error",
                        "error": str(exc),
                    },
                )
                return []
            else:
                _log_eval_tool_summary(
                    {
                        "tool": "query_amenities",
                        "status": "ok",
                        "count": len(results),
                    },
                )
                return results

        @tool(parse_docstring=True)
        async def query_services(  # noqa: PLR0913
            query: str,
            booking_required: bool | None = None,  # noqa: FBT001
            min_price: float | None = None,
            max_price: float | None = None,
            min_notice_hours: int | None = None,
            max_notice_hours: int | None = None,
            min_duration_minutes: int | None = None,
            max_duration_minutes: int | None = None,
        ) -> list[RetrievalItemLite]:
            """Retrieve hotel services relevant to a user query.

            This tool performs vector search over the **services** catalog (e.g.,
            spa treatments, housekeeping add-ons, transportation, concierge offerings,
            dining-related services, etc.).

            You may optionally apply metadata constraints. All provided constraints are
            combined using logical **AND**.

            Use this tool when the user asks for *staff-provided* or *bookable* services
            rather than physical amenities.

            Args:
                query: Natural-language query describing what the user wants.
                booking_required: Restrict results to booking_required==True/False.
                min_price: Minimum price in USD (inclusive).
                max_price: Maximum price in USD (inclusive).
                min_notice_hours: Minimum advance notice in hours (inclusive).
                max_notice_hours: Maximum advance notice in hours (inclusive).
                min_duration_minutes: Minimum duration in minutes (inclusive).
                max_duration_minutes: Maximum duration in minutes (inclusive).

            Returns:
                list[RetrievalItemLite]: Up to ``top_k`` lightweight results.

            Examples:
                - "airport shuttle"
                - "in-room massage under $200"
                - "late checkout service"
                - "laundry with at least 2 hours notice"

            """
            filters = build_filters(
                booking_required=booking_required,
                min_price=min_price,
                max_price=max_price,
                min_notice_hours=min_notice_hours,
                max_notice_hours=max_notice_hours,
                min_duration_minutes=min_duration_minutes,
                max_duration_minutes=max_duration_minutes,
            )
            try:
                results = await resources.retrieve_filtered_catalog_items(
                    source=Source.SERVICES,
                    query=query,
                    filters=filters,
                )
            except OperationalError as exc:
                logger.warning("query_services operational failure: %s", exc)
                _log_eval_tool_summary(
                    {
                        "tool": "query_services",
                        "status": "error",
                        "error": str(exc),
                    },
                )
                return []
            except Exception as exc:  # noqa: BLE001
                _log_tool_failure("query_services", exc)
                _log_eval_tool_summary(
                    {
                        "tool": "query_services",
                        "status": "error",
                        "error": str(exc),
                    },
                )
                return []
            else:
                _log_eval_tool_summary(
                    {
                        "tool": "query_services",
                        "status": "ok",
                        "count": len(results),
                    },
                )
                return results

        @tool(args_schema=RerankInput)
        def reranker(
            faq_results: list[RetrievalItemLite],
            amenities_results: list[RetrievalItemLite],
            services_results: list[RetrievalItemLite],
        ) -> list[RetrievalItemLite]:
            """Select the top-k results across FAQ, amenities, and services.

            This tool merges results from earlier retrieval tools and selects the
            highest-scoring items.

            Args:
                faq_results: Output from ``query_faq``.
                amenities_results: Output from ``query_amenities``.
                services_results: Output from ``query_services``.

            Returns:
                list[RetrievalItemLite]: The top-k items by score.

            """
            all_results = [*faq_results, *amenities_results, *services_results]
            ranked = heapq.nlargest(top_k, all_results, key=lambda x: x.score)
            _log_eval_tool_summary(
                {
                    "tool": "reranker",
                    "status": "ok",
                    "count": len(ranked),
                },
            )
            return ranked

        @tool(args_schema=HydrateInput)
        async def hydrate_items(items: list[RetrievalItemLite]) -> list[RetrievalItem]:
            """Hydrate reranked items into full text + metadata objects.

            Args:
                items: Lightweight items returned by the reranker.

            Returns:
                list[RetrievalItem]: Hydrated results. Returns an empty list on
                operational failures.

            """
            try:
                hydrated = await resources.hydrate(items, best_effort=True)
            except OperationalError as exc:
                logger.warning("hydrate_items operational failure: %s", exc)
                _log_eval_tool_summary(
                    {
                        "tool": "hydrate_items",
                        "status": "error",
                        "error": str(exc),
                    },
                )
                return []
            except Exception as exc:  # noqa: BLE001
                _log_tool_failure("hydrate_items", exc)
                _log_eval_tool_summary(
                    {
                        "tool": "hydrate_items",
                        "status": "error",
                        "error": str(exc),
                    },
                )
                return []
            else:
                _log_eval_tool_summary(
                    {
                        "tool": "hydrate_items",
                        "status": "ok",
                        "count": len(hydrated),
                    },
                )
                _log_eval_contexts(
                    [
                        item.text
                        for item in hydrated
                        if isinstance(item.text, str) and item.text
                    ],
                )
                return hydrated

        system_prompt = resources.get_system_prompt()

        return create_agent(
            model=llm,
            tools=[query_faq, query_amenities, query_services, reranker, hydrate_items],
            system_prompt=system_prompt,
        )
