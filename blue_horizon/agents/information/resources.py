"""Shared async resources for the information RAG agent.

Owns the Redis client, LlamaIndex vector indexes, and retriever cache.
Intended to be created once per process and reused across requests.
"""

from __future__ import annotations

import asyncio
import math
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.redis import RedisVectorStore
from redis.asyncio import Redis as AsyncRedis
from redis.backoff import ExponentialBackoff
from redis.exceptions import BusyLoadingError
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError
from redis.retry import Retry

from blue_horizon.agents.exceptions import OperationalError
from blue_horizon.agents.information.config import build_system_prompt
from blue_horizon.agents.information.models import RetrievalItem, Source
from blue_horizon.agents.information.retrieval import build_index_schema

if TYPE_CHECKING:
    from llama_index.core.vector_stores.types import MetadataFilters

    from blue_horizon.config import InfoEmbeddingsConfig, InfoRagConfig, InfoRedisConfig


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
        self._top_k = config.retrieval.top_k
        self._vector_dims = config.retrieval.vector_dims
        self._embed_batch_size = config.embeddings.batch_size
        self._retriever_cache_max = max(1, config.retrieval.retriever_cache_max)

        self._init_llamaindex(config.embeddings)

        retry = self._build_redis_retry(config.redis)
        self.redis_async: AsyncRedis = AsyncRedis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=config.redis.connect_timeout_s,
            socket_timeout=config.redis.socket_timeout_s,
            health_check_interval=config.redis.health_check_interval_s,
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
            base=config.retry_backoff_base_s,
            cap=config.retry_backoff_max_s,
        )
        return Retry(
            backoff=backoff,
            retries=config.retry_max_retries,
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
            embed_batch_size=cfg.batch_size,
            timeout=cfg.timeout_s,
            max_retries=cfg.max_retries,
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
