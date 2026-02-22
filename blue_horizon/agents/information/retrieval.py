"""Retrieval helpers for the information RAG agent."""

from __future__ import annotations

from typing import Any

from llama_index.core.vector_stores.types import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from redisvl.schema import IndexSchema


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
