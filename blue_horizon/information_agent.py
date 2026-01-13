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

Versions (per user):
- langchain==1.2.3, langgraph==1.0.5, llama-index==0.14.12
- redis==5.3.1, redisvl==0.4.1
"""

import heapq
import json
import logging
import math
import os
from collections import OrderedDict
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from string import Template
from typing import Any, Iterable

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.vector_stores.types import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.redis import RedisVectorStore
from pydantic import BaseModel, Field
from redis.asyncio import Redis as AsyncRedis
from redisvl.schema import IndexSchema


logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 4
DEFAULT_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_VECTOR_DIMS = 1536  # default length for text-embedding-3-small

# OpenAI client behavior: keep these bounded so requests don't hang indefinitely.
DEFAULT_OPENAI_TIMEOUT_S = 20.0
DEFAULT_OPENAI_MAX_RETRIES = 2

# Redis timeouts: keep these short so user-facing requests fail fast.
DEFAULT_REDIS_CONNECT_TIMEOUT_S = 2.0
DEFAULT_REDIS_SOCKET_TIMEOUT_S = 4.0
DEFAULT_REDIS_HEALTH_CHECK_INTERVAL_S = 30

# Retriever cache: bound this to avoid unbounded growth if the LLM emits many filter combos.
DEFAULT_RETRIEVER_CACHE_MAX = 64

# Resolve and validate the prompt path once per worker at import time.
# If packaging/layout differs across environments, override via build_system_prompt(prompt_path=...).
DEFAULT_SYSTEM_PROMPT_PATH = (
    Path(__file__).parent / "system_prompts" / "information_prompt.txt"
).resolve()


class OperationalError(RuntimeError):
    """An expected operational failure (Redis, retrieval, hydration).

    These should be logged and turned into safe/partial outputs rather than bubbling
    up to the user.
    """


def resolve_system_prompt_path(path: Path | None = None) -> Path:
    """Resolve and validate the system prompt template path.

    Args:
        path: Optional path override.

    Returns:
        Path: Resolved, existing file path.

    Raises:
        RuntimeError: If the file does not exist or is not readable.
    """
    candidate = (path or DEFAULT_SYSTEM_PROMPT_PATH).expanduser().resolve()
    if not candidate.exists() or not candidate.is_file():
        msg = f"System prompt template not found: {candidate}"
        raise RuntimeError(msg)
    return candidate


# Fail fast on startup if the default template is missing.
SYSTEM_PROMPT_PATH = resolve_system_prompt_path(DEFAULT_SYSTEM_PROMPT_PATH)


@lru_cache(maxsize=1)
def get_redis_url() -> str:
    """Return the Redis connection URL from environment.

    The value is cached per process to avoid repeated environment parsing.

    Returns:
        str: Redis connection URL.

    Raises:
        RuntimeError: If REDIS_URL is not set.
    """
    load_dotenv()
    url = os.getenv("REDIS_URL")
    if not url:
        msg = "REDIS_URL is not set"
        raise RuntimeError(msg)
    return url


class Source(StrEnum):
    """Canonical sources for hotel knowledge retrieval.

    Values are also used as Redis key prefixes.
    """

    FAQ = "faq"
    AMENITIES = "amenities"
    SERVICES = "services"


class RetrievalItemLite(BaseModel):
    """Lightweight retrieval result used for reranking and hydration."""

    source: Source = Field(..., description="faq | amenities | services")
    item_id: str = Field(..., description="Stable identifier for this item")
    score: float = Field(..., description="Similarity score; higher is more relevant")


class RetrievalItem(BaseModel):
    """Hydrated retrieval result containing text and metadata."""

    source: Source = Field(..., description="Which retriever produced this item")
    metadata: dict[str, Any] = Field(..., description="Metadata associated with the text")
    text: str = Field(..., description="The item name and description")
    score: float = Field(..., description="Similarity score; higher is more relevant")


class RerankInput(BaseModel):
    """Schema for reranker tool input."""

    faq_results: list[RetrievalItemLite] = Field(..., description="Output from query_faq")
    amenities_results: list[RetrievalItemLite] = Field(
        ..., description="Output from query_amenities"
    )
    services_results: list[RetrievalItemLite] = Field(
        ..., description="Output from query_services"
    )


class HydrateInput(BaseModel):
    """Schema for hydrate_items tool input."""

    items: list[RetrievalItemLite] = Field(..., description="Top items from reranker")


def build_filters(
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

    All provided constraints are combined with logical AND.

    Args:
        booking_required: If set, filter by booking requirement.
        min_price: Minimum price (inclusive).
        max_price: Maximum price (inclusive).
        min_notice_hours: Minimum notice in hours (inclusive).
        max_notice_hours: Maximum notice in hours (inclusive).
        min_duration_minutes: Minimum duration in minutes (inclusive).
        max_duration_minutes: Maximum duration in minutes (inclusive).

    Returns:
        MetadataFilters | None: Filters object if any constraints are provided, else None.
    """
    filters: list[MetadataFilter] = []

    if booking_required is not None:
        filters.append(
            MetadataFilter(
                key="booking_required",
                operator=FilterOperator.EQ,
                value="True" if booking_required else "False",
            )
        )

    if min_price is not None:
        filters.append(
            MetadataFilter(key="price", operator=FilterOperator.GTE, value=float(min_price))
        )
    if max_price is not None:
        filters.append(
            MetadataFilter(key="price", operator=FilterOperator.LTE, value=float(max_price))
        )

    if min_notice_hours is not None:
        filters.append(
            MetadataFilter(
                key="min_notice_hours",
                operator=FilterOperator.GTE,
                value=int(min_notice_hours),
            )
        )
    if max_notice_hours is not None:
        filters.append(
            MetadataFilter(
                key="min_notice_hours",
                operator=FilterOperator.LTE,
                value=int(max_notice_hours),
            )
        )

    if min_duration_minutes is not None:
        filters.append(
            MetadataFilter(
                key="duration", operator=FilterOperator.GTE, value=int(min_duration_minutes)
            )
        )
    if max_duration_minutes is not None:
        filters.append(
            MetadataFilter(
                key="duration", operator=FilterOperator.LTE, value=int(max_duration_minutes)
            )
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
        name: RediSearch index name.
        prefix: Redis key prefix for documents in this index.
        extra_fields: Additional metadata fields to index (e.g., tags/numerics).
        vector_dims: Embedding vector dimensionality.

    Returns:
        IndexSchema: Schema instance used to create/open the RediSearch index.
    """
    fields: list[dict[str, Any]] = [
        {"type": "tag", "name": "id"},
        {"type": "tag", "name": "doc_id"},
        {"type": "text", "name": "text"},
        {
            "type": "vector",
            "name": "vector",
            "attrs": {"dims": vector_dims, "algorithm": "hnsw", "distance_metric": "cosine"},
        },
        *extra_fields,
    ]

    schema_dict = {"index": {"name": name, "prefix": prefix}, "fields": fields}
    return IndexSchema.from_dict(schema_dict)


@dataclass(frozen=True)
class VectorIndexes:
    """Container for the three VectorStoreIndex instances."""

    faq: VectorStoreIndex
    amenities: VectorStoreIndex
    services: VectorStoreIndex


class HotelRagResources:
    """Own shared, async-first resources used by retrieval tools.

    Create one instance at FastAPI startup and reuse across requests.

    Attributes:
        redis_async: Async Redis client used for hydration and vector store I/O.
        indexes: Built VectorStoreIndex objects for FAQ, amenities, and services.
    """

    def __init__(
        self,
        *,
        redis_url: str,
        top_k: int = DEFAULT_TOP_K,
        embed_model_name: str = DEFAULT_EMBED_MODEL,
        vector_dims: int = DEFAULT_VECTOR_DIMS,
        embed_batch_size: int = 64,
        retriever_cache_max: int = DEFAULT_RETRIEVER_CACHE_MAX,
        openai_timeout_s: float = DEFAULT_OPENAI_TIMEOUT_S,
        openai_max_retries: int = DEFAULT_OPENAI_MAX_RETRIES,
        redis_connect_timeout_s: float = DEFAULT_REDIS_CONNECT_TIMEOUT_S,
        redis_socket_timeout_s: float = DEFAULT_REDIS_SOCKET_TIMEOUT_S,
        redis_health_check_interval_s: int = DEFAULT_REDIS_HEALTH_CHECK_INTERVAL_S,
    ) -> None:
        """Initialize embedding settings, Redis client, and indexes.

        Note: This constructor is synchronous; call ``await startup_check()`` during
        FastAPI startup to validate connectivity and capability.

        Args:
            redis_url: Redis connection URL.
            top_k: Number of nodes to retrieve per source.
            embed_model_name: OpenAI embedding model name.
            vector_dims: Embedding dimensionality used in Redis schema.
            embed_batch_size: Batch size used by the embedding client.
            retriever_cache_max: Max cached retrievers per process.
            redis_connect_timeout_s: Redis connect timeout in seconds.
            redis_socket_timeout_s: Redis socket timeout in seconds.
            redis_health_check_interval_s: Redis health check interval.
        """
        self._top_k = int(top_k)
        self._vector_dims = int(vector_dims)
        self._embed_batch_size = int(embed_batch_size)
        self._retriever_cache_max = max(1, int(retriever_cache_max))
        self._openai_timeout_s = float(openai_timeout_s)
        self._openai_max_retries = int(openai_max_retries)

        self._init_llamaindex(
            embed_model_name,
            embed_batch_size=self._embed_batch_size,
            timeout_s=self._openai_timeout_s,
            max_retries=self._openai_max_retries,
        )

        self.redis_async: AsyncRedis = AsyncRedis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=float(redis_connect_timeout_s),
            socket_timeout=float(redis_socket_timeout_s),
            health_check_interval=int(redis_health_check_interval_s),
        )

        self.indexes = self._build_indexes(
            self.redis_async,
            vector_dims=self._vector_dims,
        )

        self._faq_retriever = self.indexes.faq.as_retriever(similarity_top_k=self._top_k)

        # Cache catalog retrievers (amenities/services) by (source, filters_signature)
        # to reduce per-request allocations.
        self._catalog_retrievers: OrderedDict[
            tuple[Source, tuple[tuple[str, str, str], ...]], VectorIndexRetriever
        ] = OrderedDict()

    async def startup_check(self) -> None:
        """Validate Redis connectivity, retriever capability, and index schema.

        Call this once per worker during FastAPI startup.

        Raises:
            OperationalError: If Redis is unreachable, retrievers cannot run async,
                or the Redis index schema does not match configured expectations.
        """
        try:
            await self.redis_async.ping()
        except Exception as exc:  # noqa: BLE001
            msg = "Redis ping failed"
            raise OperationalError(msg) from exc

        if not hasattr(self._faq_retriever, "aretrieve"):
            msg = "FAQ retriever does not support aretrieve()"
            raise OperationalError(msg)

        # Validate catalog retrievers support aretrieve() by constructing one.
        _ = self._get_catalog_retriever(source=Source.AMENITIES, filters=None)

        # Verify the Redis index schemas match our expected vector dimensions.
        await self._validate_vector_dims()

    async def _validate_vector_dims(self) -> None:
        """Validate that Redis vector index dimensions match the configured value.

        Raises:
            OperationalError: If any index is missing, FT.INFO fails, the vector field
                cannot be located, or the dims do not match ``self._vector_dims``.
        """
        for src in (Source.FAQ, Source.AMENITIES, Source.SERVICES):
            expected = self._vector_dims
            actual = await self._get_index_vector_dims(str(src))
            if actual != expected:
                msg = f"Index '{src}' vector dims mismatch: expected={expected} actual={actual}"
                raise OperationalError(msg)

    async def _get_index_vector_dims(self, index_name: str) -> int:
        """Extract vector dims for the 'vector' field from FT.INFO.

        Args:
            index_name: RediSearch index name.

        Returns:
            int: Vector dimensionality.

        Raises:
            OperationalError: If the index cannot be inspected or dims cannot be found.
        """
        try:
            reply = await self.redis_async.execute_command("FT.INFO", index_name)
        except Exception as exc:  # noqa: BLE001
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

            # Common keys are "dims" or "dim" depending on module/version.
            dims = attr_dict.get("dims") or attr_dict.get("dim")
            if dims is None:
                # Some outputs nest vector params under "vecsim" or similar.
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

        msg = f"FT.INFO for '{index_name}' did not include a vector field named 'vector'"
        raise OperationalError(msg)

    @staticmethod
    def _redis_kv_list_to_dict(reply: Any) -> dict[str, Any]:
        """Convert Redis module replies (flat [k,v,k,v,...]) to dict recursively."""
        if isinstance(reply, dict):
            return {str(k).lower(): v for k, v in reply.items()}
        if not isinstance(reply, list):
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
    def _init_llamaindex(
        embed_model_name: str,
        *,
        embed_batch_size: int,
        timeout_s: float,
        max_retries: int,
    ) -> None:
        """Configure LlamaIndex global embedding settings.

        This sets the embedding model unconditionally to avoid relying on internal
        attribute names that may change across LlamaIndex versions.

        Side effects:
            Updates the global ``llama_index.core.Settings.embed_model`` for the
            current process.

        Args:
            embed_model_name: OpenAI embedding model name.
            embed_batch_size: Embedding batch size.
            timeout_s: Per-request timeout (seconds) for embedding calls.
            max_retries: Maximum number of retries for embedding calls.
        """
        Settings.embed_model = OpenAIEmbedding(
            model=embed_model_name,
            embed_batch_size=int(embed_batch_size),
            timeout=float(timeout_s),
            max_retries=int(max_retries),
        )

    @staticmethod
    def _coerce_score(node: Any) -> float:
        """Convert a retrieved node score to a safe float.

        Args:
            node: A LlamaIndex node-like object that may have a ``score`` attribute.

        Returns:
            float: A finite float score. Missing/None/unparseable/NaN becomes 0.0.
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
    def _filters_signature(filters: MetadataFilters | None) -> tuple[tuple[str, str, str], ...]:
        """Create a stable cache key for metadata filters.

        Args:
            filters: Filters object or None.

        Returns:
            tuple[tuple[str, str, str], ...]: A stable, hashable signature.
        """
        if not filters:
            return tuple()

        signature: list[tuple[str, str, str]] = []
        for f in getattr(filters, "filters", []) or []:
            key = str(getattr(f, "key", ""))
            op = str(getattr(f, "operator", ""))
            val = str(getattr(f, "value", ""))
            signature.append((key, op, val))

        # Order-independent signature.
        signature.sort()
        return tuple(signature)

    def _cache_retriever(
        self,
        cache_key: tuple[Source, tuple[tuple[str, str, str], ...]],
        retriever: VectorIndexRetriever,
    ) -> None:
        """Insert a retriever into the bounded LRU cache."""
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
            VectorIndexRetriever: Cached or newly created retriever.

        Raises:
            KeyError: If an unsupported source is provided.
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

        # Cache after validation.
        self._cache_retriever(cache_key, retriever)
        return retriever

    @staticmethod
    def _build_indexes(
        redis_client_async: AsyncRedis,
        *,
        vector_dims: int,
    ) -> VectorIndexes:
        """Build VectorStoreIndex instances backed by RedisVectorStore."""
        faq_schema = build_index_schema(
            name=Source.FAQ,
            prefix=Source.FAQ,
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
            vector_store=faq_store, storage_context=faq_storage
        )

        amenities_schema = build_index_schema(
            name=Source.AMENITIES,
            prefix=Source.AMENITIES,
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
            vector_store=amenities_store, storage_context=amenities_storage
        )

        services_schema = build_index_schema(
            name=Source.SERVICES,
            prefix=Source.SERVICES,
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
            vector_store=services_store, storage_context=services_storage
        )

        return VectorIndexes(faq=faq_index, amenities=amenities_index, services=services_index)

    @property
    def top_k(self) -> int:
        """Return the configured retrieval top-k."""
        return self._top_k

    async def retrieve_faq(self, query: str) -> list[RetrievalItemLite]:
        """Retrieve FAQ nodes relevant to a query."""
        try:
            nodes = await self._faq_retriever.aretrieve(query)
        except Exception as exc:  # noqa: BLE001
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
        """Retrieve amenity/service nodes for a query with optional metadata filters."""
        retriever = self._get_catalog_retriever(source=source, filters=filters)
        try:
            nodes = await retriever.aretrieve(query)
        except Exception as exc:  # noqa: BLE001
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

    async def hydrate(
        self,
        items: Iterable[RetrievalItemLite],
        *,
        best_effort: bool = True,
    ) -> list[RetrievalItem]:
        """Hydrate lite results into full objects by fetching Redis document fields.

        If best_effort=True, any individual hydration failures are logged and skipped,
        returning fewer results rather than raising.

        Args:
            items: Lightweight items returned from reranking.
            best_effort: If True, skip bad/missing items instead of raising.

        Returns:
            list[RetrievalItem]: Hydrated items with metadata and text.

        Raises:
            OperationalError: If best_effort is False and an item cannot be hydrated.
        """
        item_list = list(items)
        if not item_list:
            return []

        pipe = self.redis_async.pipeline(transaction=False)
        keys: list[str] = []

        for item in item_list:
            key = f"{item.source}:{item.item_id}"
            keys.append(key)
            pipe.hmget(key, ["_node_content", "text"])

        try:
            results = await pipe.execute()
        except Exception as exc:  # noqa: BLE001
            if best_effort:
                logger.exception("Hydration pipeline failed")
                return []
            msg = "Hydration pipeline failed"
            raise OperationalError(msg) from exc

        hydrated: list[RetrievalItem] = []
        for item, key, result in zip(item_list, keys, results, strict=True):
            try:
                node_content_str, text = result
            except Exception as exc:  # noqa: BLE001
                if best_effort:
                    logger.warning("Hydration malformed result for key=%s", key)
                    continue
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
                    continue
                msg = (
                    f"Could not hydrate item_id={item.item_id} from source={item.source} (key={key})"
                )
                raise OperationalError(msg)

            try:
                node_content = json.loads(node_content_str)
            except json.JSONDecodeError as exc:
                if best_effort:
                    logger.warning("Invalid _node_content JSON for key=%s", key)
                    continue
                msg = f"Invalid _node_content JSON for key={key}"
                raise OperationalError(msg) from exc

            hydrated.append(
                RetrievalItem(
                    source=item.source,
                    metadata=node_content.get("metadata") or {},
                    text=text or "",
                    score=item.score,
                )
            )

        return hydrated


@lru_cache(maxsize=5)
def _load_system_prompt_template(path: Path) -> Template:
    """Load and cache the system prompt template from disk."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        msg = f"Failed to read system prompt template at: {path}"
        raise RuntimeError(msg) from exc
    return Template(text)


def build_system_prompt(*, top_k: int, prompt_path: Path | None = None) -> str:
    """Render the system prompt with runtime substitutions."""
    path = resolve_system_prompt_path(prompt_path)
    template = _load_system_prompt_template(path)
    return template.safe_substitute(top_k=top_k)


def build_chat_model(
    *,
    model: str,
    temperature: float = 0.0,
    timeout_s: float = DEFAULT_OPENAI_TIMEOUT_S,
    max_retries: int = DEFAULT_OPENAI_MAX_RETRIES,
    **kwargs: Any,
) -> ChatOpenAI:
    """Create a ChatOpenAI instance with bounded timeout and retries.

    Args:
        model: Model name.
        temperature: Sampling temperature.
        timeout_s: Request timeout in seconds.
        max_retries: Maximum number of retries.
        **kwargs: Additional ChatOpenAI kwargs.

    Returns:
        ChatOpenAI: Configured chat model.
    """
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        timeout=float(timeout_s),
        max_retries=int(max_retries),
        **kwargs,
    )


class AgentFactory:
    """Factory for constructing the LangChain agent with bound async tools."""

    def __init__(self, *, resources: HotelRagResources, chat_model: Any) -> None:
        self._resources = resources
        self._chat_model = chat_model

    def build(self) -> CompiledStateGraph:
        """Build and return a compiled agent graph."""
        resources = self._resources
        top_k = resources.top_k

        def _log_tool_failure(tool_name: str, exc: Exception) -> None:
            # Single place to log tool failures without leaking details to users.
            logger.exception("Tool %s failed: %s", tool_name, exc)

        @tool(parse_docstring=True)
        async def query_faq(query: str) -> list[RetrievalItemLite]:
            """Provide information about which FAQs are relevant to the passed-in query."""
            try:
                return await resources.retrieve_faq(query)
            except Exception as exc:  # noqa: BLE001
                _log_tool_failure("query_faq", exc)
                return []

        @tool(parse_docstring=True)
        async def query_amenities(
            query: str,
            booking_required: bool | None = None,
            min_price: float | None = None,
            max_price: float | None = None,
            min_notice_hours: int | None = None,
            max_notice_hours: int | None = None,
            min_duration_minutes: int | None = None,
            max_duration_minutes: int | None = None,
        ) -> list[RetrievalItemLite]:
            """Retrieve hotel amenities relevant to a user query.

            This tool performs vector search over the **amenities** catalog and can apply
            optional metadata constraints. All provided constraints are combined using
            logical **AND**.

            Use this tool when the user asks about amenities such as facilities,
            on-property features, classes, rentals, or other amenity offerings.

            Args:
                query: Natural-language query describing what the user wants.
                    Examples: "pool hours", "spa", "yoga class", "bike rental".
                booking_required: If provided, restrict results to amenities where
                    metadata field ``booking_required`` equals "True" or "False".
                    Note: ``False`` does NOT imply ``min_notice_hours == 0``.
                min_price: Minimum price in USD (inclusive).
                max_price: Maximum price in USD (inclusive).
                min_notice_hours: Minimum advance notice in hours (inclusive).
                max_notice_hours: Maximum advance notice in hours (inclusive).
                min_duration_minutes: Minimum duration in minutes (inclusive).
                max_duration_minutes: Maximum duration in minutes (inclusive).

            Returns:
                list[RetrievalItemLite]: Up to ``top_k`` lightweight results containing:
                    - source: always ``amenities``
                    - item_id: stable Redis/LlamaIndex node identifier
                    - score: similarity score for reranking

            Notes:
                - Omit a filter argument to avoid constraining on that field.
                - Range filters are inclusive.
                - Returned items are *not* hydrated; the caller should rerank and then
                  call ``hydrate_items`` to fetch full text + metadata.

            Example:
                Find bookable amenities under $50 that last <= 60 minutes:

                query_amenities(
                    query="massage or spa treatment",
                    booking_required=True,
                    max_price=50,
                    max_duration_minutes=60,
                )
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
                return await resources.retrieve_filtered_catalog_items(
                    source=Source.AMENITIES,
                    query=query,
                    filters=filters,
                )
            except Exception as exc:  # noqa: BLE001
                _log_tool_failure("query_amenities", exc)
                return []

        @tool(parse_docstring=True)
        async def query_services(
            query: str,
            booking_required: bool | None = None,
            min_price: float | None = None,
            max_price: float | None = None,
            min_notice_hours: int | None = None,
            max_notice_hours: int | None = None,
            min_duration_minutes: int | None = None,
            max_duration_minutes: int | None = None,
        ) -> list[RetrievalItemLite]:
            """Retrieve hotel services relevant to a user query.

            This tool performs vector search over the **services** catalog and can apply
            optional metadata constraints. All provided constraints are combined using
            logical **AND**.

            Use this tool when the user asks about operational services such as dining
            delivery, housekeeping, laundry, concierge, transportation, business services,
            or other staff-provided offerings.

            Args:
                query: Natural-language query describing what the user wants.
                    Examples: "airport shuttle", "laundry", "room service", "concierge".
                booking_required: If provided, restrict results to services where
                    metadata field ``booking_required`` equals "True" or "False".
                    Note: ``False`` does NOT imply ``min_notice_hours == 0``.
                min_price: Minimum price in USD (inclusive).
                max_price: Maximum price in USD (inclusive).
                min_notice_hours: Minimum advance notice in hours (inclusive).
                max_notice_hours: Maximum advance notice in hours (inclusive).
                min_duration_minutes: Minimum duration in minutes (inclusive).
                max_duration_minutes: Maximum duration in minutes (inclusive).

            Returns:
                list[RetrievalItemLite]: Up to ``top_k`` lightweight results containing:
                    - source: always ``services``
                    - item_id: stable Redis/LlamaIndex node identifier
                    - score: similarity score for reranking

            Notes:
                - Omit a filter argument to avoid constraining on that field.
                - Range filters are inclusive.
                - Returned items are *not* hydrated; the caller should rerank and then
                  call ``hydrate_items`` to fetch full text + metadata.

            Example:
                Find services that do not require booking and cost <= $25:

                query_services(
                    query="laundry and pressing",
                    booking_required=False,
                    max_price=25,
                )
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
                return await resources.retrieve_filtered_catalog_items(
                    source=Source.SERVICES,
                    query=query,
                    filters=filters,
                )
            except Exception as exc:  # noqa: BLE001
                _log_tool_failure("query_services", exc)
                return []

        @tool(args_schema=RerankInput)
        def reranker(
            faq_results: list[RetrievalItemLite],
            amenities_results: list[RetrievalItemLite],
            services_results: list[RetrievalItemLite],
        ) -> list[RetrievalItemLite]:
            """Select the top-k results across FAQ, amenities, and services."""
            all_results = [*faq_results, *amenities_results, *services_results]
            return heapq.nlargest(top_k, all_results, key=lambda x: x.score)

        @tool(args_schema=HydrateInput)
        async def hydrate_items(items: list[RetrievalItemLite]) -> list[RetrievalItem]:
            """Hydrate reranked items into full text + metadata objects."""
            try:
                return await resources.hydrate(items, best_effort=True)
            except Exception as exc:  # noqa: BLE001
                _log_tool_failure("hydrate_items", exc)
                return []

        system_prompt = build_system_prompt(top_k=top_k)

        return create_agent(
            model=self._chat_model,
            tools=[query_faq, query_amenities, query_services, reranker, hydrate_items],
            system_prompt=system_prompt,
        )
