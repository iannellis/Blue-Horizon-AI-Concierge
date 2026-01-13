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
import math
import os
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from string import Template
from typing import Any, Iterable

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
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


DEFAULT_TOP_K = 4
DEFAULT_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_VECTOR_DIMS = 1536  # default length for text-embedding-3-small

# Resolve and validate the prompt path once per worker at import time.
# If packaging/layout differs across environments, override via build_system_prompt(prompt_path=...).
DEFAULT_SYSTEM_PROMPT_PATH = (
    Path(__file__).parent / "system_prompts" / "information_prompt.txt"
).resolve()


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
        raise RuntimeError(f"System prompt template not found: {candidate}")
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
        raise RuntimeError("REDIS_URL is not set")
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
    ) -> None:
        """Initialize Redis clients, embedding settings, and vector indexes.

        Args:
            redis_url: Redis connection URL.
            top_k: Number of nodes to retrieve per source.
            embed_model_name: OpenAI embedding model name.
            vector_dims: Embedding dimensionality used in Redis schema.
            embed_batch_size: Batch size used by the embedding client.
        """
        self._top_k = int(top_k)
        self._vector_dims = int(vector_dims)
        self._embed_batch_size = int(embed_batch_size)

        self._init_llamaindex(embed_model_name, embed_batch_size=self._embed_batch_size)

        self.redis_async: AsyncRedis = AsyncRedis.from_url(redis_url, decode_responses=True)

        self.indexes = self._build_indexes(
            self.redis_async,
            vector_dims=self._vector_dims,
        )

        self._faq_retriever = self.indexes.faq.as_retriever(similarity_top_k=self._top_k)

        # Cache catalog retrievers (amenities/services) by (source, filters_signature)
        # to reduce per-request allocations.
        self._catalog_retrievers: dict[tuple[Source, tuple[tuple[str, str, str], ...]], VectorIndexRetriever] = {}

    @staticmethod
    def _init_llamaindex(embed_model_name: str, *, embed_batch_size: int) -> None:
        """Configure LlamaIndex global embedding settings.

        This sets the embedding model unconditionally to avoid relying on internal
        attribute names that may change across LlamaIndex versions.

        Args:
            embed_model_name: OpenAI embedding model name.
            embed_batch_size: Embedding batch size.
        """
        Settings.embed_model = OpenAIEmbedding(
            model=embed_model_name,
            embed_batch_size=int(embed_batch_size),
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
        return 0.0 if math.isnan(score) else score

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
            RuntimeError: If the retriever does not support async retrieval.
        """
        sig = self._filters_signature(filters)
        cache_key = (source, sig)

        existing = self._catalog_retrievers.get(cache_key)
        if existing is not None:
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
            raise RuntimeError(
                "VectorIndexRetriever does not support aretrieve(); provide redis_client_async "
                "and use an async-capable LlamaIndex retriever implementation."
            )

        # Cache after validation. Races are acceptable; at worst we build twice.
        self._catalog_retrievers[cache_key] = retriever
        return retriever

    @staticmethod
    def _build_indexes(
        redis_client_async: AsyncRedis,
        *,
        vector_dims: int,
    ) -> VectorIndexes:
        """Build VectorStoreIndex instances backed by RedisVectorStore.

        Args:
            redis_client_async: Async Redis client passed into RedisVectorStore.
            vector_dims: Embedding dimensionality used in index schemas.

        Returns:
            VectorIndexes: FAQ, amenities, and services indexes.
        """
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
        """Retrieve FAQ nodes relevant to a query.

        Args:
            query: User query string.

        Returns:
            list[RetrievalItemLite]: Lightweight results for reranking.

        Raises:
            RuntimeError: If the underlying retriever does not support async retrieval.
        """
        if not hasattr(self._faq_retriever, "aretrieve"):
            raise RuntimeError(
                "FAQ retriever does not support aretrieve(); provide redis_client_async "
                "and use an async-capable LlamaIndex retriever implementation."
            )

        nodes = await self._faq_retriever.aretrieve(query)
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
            query: User query string.
            filters: Optional metadata filters to apply.

        Returns:
            list[RetrievalItemLite]: Lightweight results for reranking.

        Raises:
            KeyError: If an unsupported source is provided.
            RuntimeError: If the retriever does not support async retrieval.
        """
        retriever = self._get_catalog_retriever(source=source, filters=filters)
        nodes = await retriever.aretrieve(query)
        return [
            RetrievalItemLite(
                source=source,
                item_id=getattr(n, "id_", ""),
                score=self._coerce_score(n),
            )
            for n in nodes
        ]

    async def hydrate(self, items: Iterable[RetrievalItemLite]) -> list[RetrievalItem]:
        """Hydrate lite results into full objects by fetching Redis document fields.

        Uses an async Redis pipeline to minimize round trips.

        Args:
            items: Lightweight items returned from reranking.

        Returns:
            list[RetrievalItem]: Hydrated items with metadata and text.

        Raises:
            RuntimeError: If an item cannot be hydrated or stored JSON is invalid.
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

        results = await pipe.execute()

        hydrated: list[RetrievalItem] = []
        for item, key, result in zip(item_list, keys, results, strict=True):
            node_content_str, text = result
            if not node_content_str:
                raise RuntimeError(
                    f"Could not hydrate item_id={item.item_id} from source={item.source} (key={key})"
                )

            try:
                node_content = json.loads(node_content_str)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid _node_content JSON for key={key}") from e

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
    """Load and cache the system prompt template from disk.

    Args:
        path: Path to the template file.

    Returns:
        Template: Parsed template instance.

    Raises:
        RuntimeError: If the file cannot be read.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise RuntimeError(f"Failed to read system prompt template at: {path}") from e
    return Template(text)


def build_system_prompt(*, top_k: int, prompt_path: Path | None = None) -> str:
    """Render the system prompt with runtime substitutions.

    Args:
        top_k: Number of items each retriever returns.
        prompt_path: Optional override for the template file location.

    Returns:
        str: Rendered prompt text.
    """
    path = resolve_system_prompt_path(prompt_path)
    template = _load_system_prompt_template(path)
    return template.safe_substitute(top_k=top_k)


class AgentFactory:
    """Factory for constructing the LangChain agent with bound async tools."""

    def __init__(self, *, resources: HotelRagResources, chat_model: Any) -> None:
        """Initialize the agent factory.

        Args:
            resources: Shared resources used by tool implementations.
            chat_model: LangChain chat model used by the agent.
        """
        self._resources = resources
        self._chat_model = chat_model

    def build(self) -> CompiledStateGraph:
        """Build and return a compiled agent graph.

        Returns:
            CompiledStateGraph: Graph ready for `ainvoke`.
        """
        resources = self._resources
        top_k = resources.top_k

        @tool(parse_docstring=True)
        async def query_faq(query: str) -> list[RetrievalItemLite]:
            """Provide information about which FAQs are relevant to the passed-in query."""
            return await resources.retrieve_faq(query)

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
            return await resources.retrieve_filtered_catalog_items(
                source=Source.AMENITIES,
                query=query,
                filters=filters,
            )

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
            return await resources.retrieve_filtered_catalog_items(
                source=Source.SERVICES,
                query=query,
                filters=filters,
            )

        @tool(args_schema=RerankInput)
        def reranker(
            faq_results: list[RetrievalItemLite],
            amenities_results: list[RetrievalItemLite],
            services_results: list[RetrievalItemLite],
        ) -> list[RetrievalItemLite]:
            """Select the top-k results across FAQ, amenities, and services.

            This tool combines the three retrieval result lists and returns the global
            top ``top_k`` items by similarity score. It does **not** hydrate results; it
            only selects which item IDs should be hydrated next.

            Args:
                faq_results: Lightweight FAQ retrieval results.
                amenities_results: Lightweight amenities retrieval results.
                services_results: Lightweight services retrieval results.

            Returns:
                list[RetrievalItemLite]: The best-scoring items across all sources,
                limited to ``top_k``.

            Notes:
                - Scores are compared directly across sources.
                - If score scales differ between sources, consider normalizing scores
                  before reranking.
            """
            all_results = [*faq_results, *amenities_results, *services_results]
            return heapq.nlargest(top_k, all_results, key=lambda x: x.score)

        @tool(args_schema=HydrateInput)
        async def hydrate_items(items: list[RetrievalItemLite]) -> list[RetrievalItem]:
            """Hydrate reranked items into full text + metadata objects.

            This tool takes the output of ``reranker`` (lite items containing ``source``
            and ``item_id``) and fetches stored fields from Redis to produce hydrated
            results that can be presented to the user.

            Args:
                items: Lightweight items selected by ``reranker``.

            Returns:
                list[RetrievalItem]: Hydrated items containing:
                    - source
                    - text
                    - metadata
                    - score (from reranking)

            Raises:
                RuntimeError: If any item cannot be hydrated (missing Redis key/fields)
                or stored JSON is invalid.
            """
            return await resources.hydrate(items)

        system_prompt = build_system_prompt(top_k=top_k)

        return create_agent(
            model=self._chat_model,
            tools=[query_faq, query_amenities, query_services, reranker, hydrate_items],
            system_prompt=system_prompt,
        )
