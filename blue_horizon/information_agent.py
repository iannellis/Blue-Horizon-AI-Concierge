"""Hotel info RAG agent (LangChain create_agent + LangGraph) with RedisVectorStore.

Goals:
- Modular, readable, PEP8-compliant
- Fast startup and steady-state performance
- Async-friendly for FastAPI (no blocking event loop)

Assumptions:
- Redis keys follow the pattern: {prefix}:{id} where prefix is one of: faq, amenities, services
- Each key stores fields: "_node_content" (JSON string) and "text" (string)
- booking_required is stored as a Redis TAG field with values "True" / "False".

Versions (per user):
- langchain==1.2.3, langgraph==1.0.5, llama-index==0.14.12
- redis==5.3.1, redisvl==0.4.1
"""

from __future__ import annotations

import heapq
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from string import Template
from typing import Any, Optional

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
from redis import Redis
from redis.asyncio import Redis as AsyncRedis
from redisvl.schema import IndexSchema

DEFAULT_TOP_K = 4
DEFAULT_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_VECTOR_DIMS = 1536  # default length for text-embedding-3-small

SYSTEM_PROMPT_PATH = Path(__file__).parent / "system_prompts/information_prompt.txt"


@lru_cache(maxsize=1)
def get_redis_url() -> str:
    load_dotenv()
    url = os.getenv("REDIS_URL")
    if not url:
        raise RuntimeError("REDIS_URL is not set")
    return url


class Source(StrEnum):
    FAQ = "faq"
    AMENITIES = "amenities"
    SERVICES = "services"


class RetrievalItemLite(BaseModel):
    source: Source = Field(..., description="faq | amenities | services")
    item_id: str = Field(..., description="Stable identifier for this item")
    score: float = Field(..., description="Similarity score; higher is more relevant")


class RetrievalItem(BaseModel):
    source: Source = Field(..., description="Which retriever produced this item")
    metadata: dict[str, Any] = Field(..., description="Metadata associated with the text")
    text: str = Field(..., description="The item name and description")
    score: float = Field(..., description="Similarity score; higher is more relevant")


class RerankInput(BaseModel):
    faq_results: list[RetrievalItemLite] = Field(..., description="Output from query_faq")
    amenities_results: list[RetrievalItemLite] = Field(
        ..., description="Output from query_amenities"
    )
    services_results: list[RetrievalItemLite] = Field(
        ..., description="Output from query_services"
    )


class HydrateInput(BaseModel):
    items: list[RetrievalItemLite] = Field(..., description="Top items from reranker")


def build_filters(
    *,
    booking_required: Optional[bool] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    min_notice_hours: Optional[int] = None,
    max_notice_hours: Optional[int] = None,
    min_duration_minutes: Optional[int] = None,
    max_duration_minutes: Optional[int] = None,
) -> Optional[MetadataFilters]:
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
    faq: VectorStoreIndex
    amenities: VectorStoreIndex
    services: VectorStoreIndex


class HotelRagResources:
    """Owns shared resources for the agent."""

    def __init__(
        self,
        *,
        redis_url: str,
        top_k: int = DEFAULT_TOP_K,
        embed_model_name: str = DEFAULT_EMBED_MODEL,
        vector_dims: int = DEFAULT_VECTOR_DIMS,
        embed_batch_size: int = 64,
    ) -> None:
        self._top_k = int(top_k)
        self._vector_dims = int(vector_dims)
        self._embed_batch_size = int(embed_batch_size)

        self._init_llamaindex(embed_model_name, embed_batch_size=self._embed_batch_size)

        # Keep *only* an async client on the instance for request-path operations.
        # A sync Redis client is created transiently for RedisVectorStore initialization.
        self.redis_async: AsyncRedis = AsyncRedis.from_url(redis_url, decode_responses=True)

        redis_sync = Redis.from_url(redis_url, decode_responses=True)
        self.indexes = self._build_indexes(
            redis_sync,
            self.redis_async,
            vector_dims=self._vector_dims,
        )

        self._faq_retriever = self.indexes.faq.as_retriever(similarity_top_k=self._top_k)

    @staticmethod
    def _init_llamaindex(embed_model_name: str, *, embed_batch_size: int) -> None:
        current_embed = getattr(Settings, "embed_model", None)
        current_model = (
            getattr(current_embed, "model", None)
            or getattr(current_embed, "model_name", None)
            or getattr(current_embed, "_model", None)
        )

        if (current_embed is None) or (current_model != embed_model_name):
            Settings.embed_model = OpenAIEmbedding(
                model=embed_model_name,
                embed_batch_size=int(embed_batch_size),
            )

    @staticmethod
    def _build_indexes(
        redis_client: Redis,
        redis_client_async: AsyncRedis,
        *,
        vector_dims: int,
    ) -> VectorIndexes:
        faq_schema = build_index_schema(
            name="faq",
            prefix="faq",
            extra_fields=[{"type": "tag", "name": "category"}],
            vector_dims=vector_dims,
        )
        faq_store = RedisVectorStore(
            schema=faq_schema,
            redis_client=redis_client,
            redis_client_async=redis_client_async,
            overwrite=False,
        )
        faq_storage = StorageContext.from_defaults(vector_store=faq_store)
        faq_index = VectorStoreIndex.from_vector_store(
            vector_store=faq_store, storage_context=faq_storage
        )

        amenities_schema = build_index_schema(
            name="amenities",
            prefix="amenities",
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
            redis_client=redis_client,
            redis_client_async=redis_client_async,
            overwrite=False,
        )
        amenities_storage = StorageContext.from_defaults(vector_store=amenities_store)
        amenities_index = VectorStoreIndex.from_vector_store(
            vector_store=amenities_store, storage_context=amenities_storage
        )

        services_schema = build_index_schema(
            name="services",
            prefix="services",
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
            redis_client=redis_client,
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
        return self._top_k

    async def retrieve_faq(self, query: str) -> list[RetrievalItemLite]:
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
                score=float(getattr(n, "score", 0.0) or 0.0),
            )
            for n in nodes
        ]

    async def retrieve_vector(
        self,
        *,
        source: Source,
        query: str,
        filters: Optional[MetadataFilters],
    ) -> list[RetrievalItemLite]:
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

        nodes = await retriever.aretrieve(query)
        return [
            RetrievalItemLite(
                source=source,
                item_id=getattr(n, "id_", ""),
                score=float(getattr(n, "score", 0.0) or 0.0),
            )
            for n in nodes
        ]

    async def hydrate(self, items: Iterable[RetrievalItemLite]) -> list[RetrievalItem]:
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


@lru_cache(maxsize=8)
def _load_system_prompt_template(path: str) -> Template:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as e:
        raise RuntimeError(f"Failed to read system prompt template at: {Path(path).resolve()}") from e
    return Template(text)


def build_system_prompt(*, top_k: int, prompt_path: Path | None = None) -> str:
    path = str(prompt_path or SYSTEM_PROMPT_PATH)
    template = _load_system_prompt_template(path)
    return template.safe_substitute(top_k=top_k)


class AgentFactory:
    def __init__(self, *, resources: HotelRagResources, chat_model: Any) -> None:
        self._resources = resources
        self._chat_model = chat_model

    def build(self) -> CompiledStateGraph:
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
            """Provide information about which amenities are relevant to the passed-in query."""
            filters = build_filters(
                booking_required=booking_required,
                min_price=min_price,
                max_price=max_price,
                min_notice_hours=min_notice_hours,
                max_notice_hours=max_notice_hours,
                min_duration_minutes=min_duration_minutes,
                max_duration_minutes=max_duration_minutes,
            )
            return await resources.retrieve_vector(
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
            """Provide information about which services are relevant to the passed-in query."""
            filters = build_filters(
                booking_required=booking_required,
                min_price=min_price,
                max_price=max_price,
                min_notice_hours=min_notice_hours,
                max_notice_hours=max_notice_hours,
                min_duration_minutes=min_duration_minutes,
                max_duration_minutes=max_duration_minutes,
            )
            return await resources.retrieve_vector(
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
            all_results = [*faq_results, *amenities_results, *services_results]
            return heapq.nlargest(top_k, all_results, key=lambda x: x.score)

        @tool(args_schema=HydrateInput)
        async def hydrate_items(items: list[RetrievalItemLite]) -> list[RetrievalItem]:
            return await resources.hydrate(items)

        system_prompt = build_system_prompt(top_k=top_k)

        return create_agent(
            model=self._chat_model,
            tools=[query_faq, query_amenities, query_services, reranker, hydrate_items],
            system_prompt=system_prompt,
        )
