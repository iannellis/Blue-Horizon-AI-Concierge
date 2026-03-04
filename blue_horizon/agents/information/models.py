"""Domain models for the information RAG agent."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field


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

    This model captures one or more atomic search strings plus optional constraints
    that map directly to the amenity/service metadata filters in Redis.
    """

    queries: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "One or more short, dense search strings capturing the user's intent. "
            "If the user asks about multiple distinct topics, produce one string per "
            "topic (e.g. ['gym hours', 'spa full-service']). Each string must contain "
            "only the core noun phrases stripped of context, reasoning, or filler "
            "(e.g. 'so I can plan\u2026', 'because I want\u2026'). Single-topic "
            "requests produce exactly one string."
        ),
    )
    booking_required: bool | None = Field(
        default=None,
        description=(
            "If the user explicitly requires booking (NOT cases where booking required "
            "is OK or they're fine with booking required), set True (will almost "
            "never happen). If the user explicitly wants no booking "
            "(e.g., 'no booking', 'no reservation', 'walk-in'), set False. "
            "For multi-item requests: set False only if ALL requested items carry "
            "a no-booking constraint; leave null if signals are mixed or absent. "
            "Do not set otherwise. "
            "Do not infer notice requirements from booking status."
        ),
    )
    min_price: float | None = Field(
        default=None,
        description=(
            "Minimum acceptable price in USD (inclusive). "
            "Only set for explicit minimum phrases: 'at least $X', "
            "'starting from $X', 'minimum $X'. "
            "Never set for budget-cap phrases ('under $X', 'both under $X', "
            "'budget up to $X') — those set max_price only."
        ),
    )
    max_price: float | None = Field(
        default=None,
        description=(
            "Maximum acceptable price in USD (inclusive). "
            "Set for any budget-cap phrase: 'under $X', 'budget up to $X', "
            "'max $X', 'no more than $X', 'both under $X', 'each under $X'. "
            "For multi-item requests sharing one cap, max_price = that cap. "
            "For multi-item requests with different per-item prices stated, "
            "max_price = the largest value (avoids over-filtering). "
            "If no price is stated for any item, leave null."
        ),
    )
    max_notice_hours: int | None = Field(
        default=None,
        description=(
            "Maximum advance notice the user can give, in hours (inclusive). "
            "Only set if explicitly mentioned (e.g., 'can only give X hours notice'). "
            "For multi-item requests sharing a single notice cap, "
            "max_notice_hours = that cap. "
            "For multi-item requests with different per-item notice constraints, "
            "max_notice_hours = largest value (avoids over-filtering). "
            "Do not infer from 'walk-in' or 'no booking' — "
            "those only affect booking_required."
        ),
    )
    min_duration_minutes: int | None = Field(
        default=None,
        description=(
            "Minimum duration in minutes (inclusive). "
            "Determined by counting ALL distinct services/activities in the request: "
            "if exactly ONE service is requested with an explicit duration, "
            "min = that duration — EXCEPT for a single quick snack/bite item "
            "(e.g. 'in-room bite', 'snack'), where the duration is a time cap "
            "and min = null; "
            "if TWO OR MORE services are requested and ALL carry explicit durations, "
            "min = the SMALLEST value across all of them; "
            "if TWO OR MORE services are requested and ANY lacks an explicit duration, "
            "min = null. "
            "Never apply per-item logic — evaluate the whole request together. "
            "Only set min alone for 'at least X minutes' phrases."
        ),
    )
    max_duration_minutes: int | None = Field(
        default=None,
        description=(
            "Maximum duration in minutes (inclusive). "
            "Determined by counting ALL distinct services/activities in the request: "
            "if exactly ONE service is requested with an explicit duration, "
            "max = that duration; "
            "if TWO OR MORE services are requested and ALL carry explicit durations, "
            "max = the LARGEST value across all of them; "
            "if TWO OR MORE services are requested and ANY lacks an explicit duration, "
            "max = null. "
            "Never apply per-item logic — evaluate the whole request together. "
            "Do not estimate durations for items that have none stated. "
            "Only set max alone for 'at most X minutes' or overall time-cap phrases."
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
