import heapq
import json
import os
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexAutoRetriever
from llama_index.core.vector_stores.types import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    MetadataInfo,
    VectorStoreInfo,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.redis import RedisVectorStore
from pydantic import BaseModel, Field
from redis import Redis
from redisvl.schema import IndexSchema

load_dotenv()
redis_conn_string = os.getenv("REDIS_URL")

gpt_version = "gpt-5.2"
top_k = 4

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = OpenAI(gpt_version, temperature=0, reasoning={"effort":"medium"})
llm = ChatOpenAI(model=gpt_version, temperature=0, reasoning={"effort": "medium"})

redis_client = Redis.from_url(redis_conn_string, decode_responses=True)

# Typing for retrievers
class RetrievalItemLite(BaseModel):
    source: str = Field(..., description="FAQ | amenities | services")
    item_id: str = Field(..., description="Stable identifier for this item")
    score: float = Field(..., description="Similarity score; higher is more relevant")

class RetrievalItem(BaseModel):
    source: str = Field(..., description="Which retriever produced this item: FAQ, amenities, or services")
    metadata: dict[str, Any] = Field(..., description="The metadata associated with the text.")
    text: str = Field(..., description="The item name and description")
    score: float = Field(..., description="Similarity score; higher is more relevant")


# Setup the FAQ retriever
schema = IndexSchema.from_dict(
    {
        "index": {"name": "FAQ", "prefix": "FAQ"},
        # customize fields that are indexed
        "fields": [
            # required fields for llamaindex
            {"type": "tag", "name": "id"},
            {"type": "tag", "name": "doc_id"},
            {"type": "text", "name": "text"},
            # custom for metadata filtering
            {"type": "tag", "name": "category"},
            # custom vector field for bge-small-en-v1.5
            {
                "type": "vector",
                "name": "vector",
                "attrs": {
                    "dims": 384,
                    "algorithm": "hnsw",
                    "distance_metric": "cosine",
                },
            },
        ],
    },
)

faq_vector_store = RedisVectorStore(schema=schema, redis_client=redis_client, overwrite=False)
storage_context = StorageContext.from_defaults(vector_store=faq_vector_store)
faq_index = VectorStoreIndex.from_vector_store(vector_store=faq_vector_store, storage_context=storage_context)
faq_retriever = faq_index.as_retriever(similarity_top_k=top_k)

# Setup the Amenities index (we need to setup the retriever with each tool call)
schema = IndexSchema.from_dict(
    {
        "index": {"name": "Amenities", "prefix": "Amenities"},
        # customize fields that are indexed
        "fields": [
            # required fields for llamaindex
            {"type": "tag", "name": "id"},
            {"type": "tag", "name": "doc_id"},
            {"type": "text", "name": "text"},
            # custom fields for filtering
            {"type": "tag", "name": "category"},
            {"type": "numeric", "name": "price"},
            {"type": "numeric", "name": "duration"},
            {"type": "numeric", "name": "min_notice_hours"},
            {"type": "tag", "name": "booking_required"},
            # custom vector field for bge-small-en-v1.5 embeddings
            {
                "type": "vector",
                "name": "vector",
                "attrs": {
                    "dims": 384,
                    "algorithm": "hnsw",
                    "distance_metric": "cosine",
                },
            },
        ],
    },
)

amenities_vector_store = RedisVectorStore(schema=schema, redis_client=redis_client, overwrite=False)
storage_context = StorageContext.from_defaults(vector_store=amenities_vector_store)
amenities_index = VectorStoreIndex.from_vector_store(vector_store=amenities_vector_store, storage_context=storage_context)

# Setup the Services index (we need to setup the retriever with each tool call)
schema = IndexSchema.from_dict(
    {
        "index": {"name": "Services", "prefix": "Services"},
        # customize fields that are indexed
        "fields": [
            # required fields for llamaindex
            {"type": "tag", "name": "id"},
            {"type": "tag", "name": "doc_id"},
            {"type": "text", "name": "text"},
            # custom fields for filtering
            {"type": "tag", "name": "service_type"},
            {"type": "numeric", "name": "price"},
            {"type": "numeric", "name": "duration"},
            {"type": "numeric", "name": "min_notice_hours"},
            {"type": "tag", "name": "department"},
            {"type": "tag", "name": "booking_required"},
            # custom vector field for bge-small-en-v1.5 embeddings
            {
                "type": "vector",
                "name": "vector",
                "attrs": {
                    "dims": 384,
                    "algorithm": "hnsw",
                    "distance_metric": "cosine",
                },
            },
        ],
    },
)

services_vector_store = RedisVectorStore(schema=schema, redis_client=redis_client, overwrite=False)
storage_context = StorageContext.from_defaults(vector_store=services_vector_store)
services_index = VectorStoreIndex.from_vector_store(vector_store=services_vector_store, storage_context=storage_context)

# used by both amenities and services
def _range_filters(
    *,
    min_price: float | None = None,
    max_price: float | None = None,
    min_notice_hours: int | None = None,
    max_notice_hours: int | None = None,
    min_duration_minutes: int | None = None,
    max_duration_minutes: int | None = None,
) -> MetadataFilters | None:
    filters: list[MetadataFilter] = []

    if min_price is not None:
        filters.append(MetadataFilter(key="price", operator=FilterOperator.GTE, value=float(min_price)))
    if max_price is not None:
        filters.append(MetadataFilter(key="price", operator=FilterOperator.LTE, value=float(max_price)))

    if min_notice_hours is not None:
        filters.append(MetadataFilter(key="min_notice_hours", operator=FilterOperator.GTE, value=int(min_notice_hours)))
    if max_notice_hours is not None:
        filters.append(MetadataFilter(key="min_notice_hours", operator=FilterOperator.LTE, value=int(max_notice_hours)))

    if min_duration_minutes is not None:
        filters.append(MetadataFilter(key="duration", operator=FilterOperator.GTE, value=int(min_duration_minutes)))
    if max_duration_minutes is not None:
        filters.append(MetadataFilter(key="duration", operator=FilterOperator.LTE, value=int(max_duration_minutes)))

    return MetadataFilters(filters=filters) if filters else None # pyright: ignore[reportArgumentType]


@tool(parse_docstring=True)
def query_faq(query: str) -> list[RetrievalItemLite]:
    """Provide information about which FAQs are relevant to the passed-in query.

    Args:
        query (string): The query string.

    Returns:
        list[RetrievalItemLite]: The source, item id, and similarity score for each item

    """
    nodes = faq_retriever.retrieve(query)

    return [RetrievalItemLite(source = "FAQ",
                          item_id = getattr(n, "id_", ""),
                          score =  getattr(n, "score", 0))
            for n in nodes]


index_name = amenities_vector_store.index_name
categories = redis_client.ft(index_name).tagvals("category")
amenities_store_info = VectorStoreInfo(
    content_info="Information about hotel amenities",
    metadata_info=[
        MetadataInfo(
            name="category",
            type="tag",
            description=f"Category of the amenity, one of {categories}",
        ),
        MetadataInfo(
            name="booking_required",
            type="tag",
            description="Whether a booking is requried for the amenity, either 'True' or 'False'. 'False' does NOT imply min_notice_hours=0.",
        ),
    ],
)

@tool(parse_docstring=True)
def query_amenities(
    query: str,
    min_price: float | None = None,
    max_price: float | None = None,
    min_notice_hours: int | None = None,
    max_notice_hours: int | None = None,
    min_duration_minutes: int | None = None,
    max_duration_minutes: int | None = None,
) -> list[RetrievalItemLite]:
    """Provide information about which amenities are relevant to the passed-in query.

    Supports optional range filtering on numeric fields.

    Args:
        query (string): The query string.
        min_price (float, optional): Minimum price (inclusive).
        max_price (float, optional): Maximum price (inclusive).
        min_notice_hours (int, optional): Minimum notice hours (inclusive).
        max_notice_hours (int, optional): Maximum notice hours (inclusive).
        min_duration_minutes (int, optional): Minimum duration in minutes (inclusive).
        max_duration_minutes (int, optional): Maximum duration in minutes (inclusive).

    Returns:
        list[RetrievalItemLite]: The source, item id, and similarity score for each item

    """
    filters = _range_filters(
        min_price=min_price,
        max_price=max_price,
        min_notice_hours=min_notice_hours,
        max_notice_hours=max_notice_hours,
        min_duration_minutes=min_duration_minutes,
        max_duration_minutes=max_duration_minutes,
    )

    amenities_retriever = VectorIndexAutoRetriever(
        amenities_index, extra_filters=filters, vector_store_info=amenities_store_info,
        similarity_top_k=4)

    nodes = amenities_retriever.retrieve(query)

    return [RetrievalItemLite(source = "amenities",
                          item_id = getattr(n, "id_", ""),
                          score =  getattr(n, "score", 0))
            for n in nodes]

index_name = services_vector_store.index_name
service_types = redis_client.ft(index_name).tagvals("service_type")
services_store_info = VectorStoreInfo(
    content_info="Information about hotel services",
    metadata_info=[
        MetadataInfo(
            name="service_type",
            type="tag",
            description=f"The type of service provided, one of {service_types}",
        ),
        MetadataInfo(
            name="booking_required",
            type="tag",
            description="""Whether a booking is requried for the service, either 'True'
            or 'False'. 'False' does NOT imply min_notice_hours=0.""",
        ),
    ],
)

@tool(parse_docstring=True)
def query_services(
    query: str,
    min_price: float | None = None,
    max_price: float | None = None,
    min_notice_hours: int | None = None,
    max_notice_hours: int | None = None,
    min_duration_minutes: int | None = None,
    max_duration_minutes: int | None = None,
) -> list[RetrievalItemLite]:
    """Provide information about which services are relevant to the passed-in query.

    Supports optional range filtering on numeric fields.

    Args:
        query (string): The query string.
        min_price (float, optional): Minimum price (inclusive).
        max_price (float, optional): Maximum price (inclusive).
        min_notice_hours (int, optional): Minimum notice hours (inclusive).
        max_notice_hours (int, optional): Maximum notice hours (inclusive).
        min_duration_minutes (int, optional): Minimum duration in minutes (inclusive).
        max_duration_minutes (int, optional): Maximum duration in minutes (inclusive).

    Returns:
        list[RetrievalItemLite]: The source, item id, and similarity score for each item

    """
    filters = _range_filters(
        min_price=min_price,
        max_price=max_price,
        min_notice_hours=min_notice_hours,
        max_notice_hours=max_notice_hours,
        min_duration_minutes=min_duration_minutes,
        max_duration_minutes=max_duration_minutes,
    )

    retriever = VectorIndexAutoRetriever(services_index, extra_filters=filters, vector_store_info=services_store_info, similarity_top_k=4)

    nodes = retriever.retrieve(query)

    return [RetrievalItemLite(source = "services",
                          item_id = getattr(n, "id_", ""),
                          score =  getattr(n, "score", 0))
            for n in nodes]

class RerankInput(BaseModel):
    faq_results: list[RetrievalItemLite] = Field(..., description="Output from query_faq")
    amenities_results: list[RetrievalItemLite] = Field(..., description="Output from query_amenities")
    services_results: list[RetrievalItemLite] = Field(..., description="Output from query_services")

@tool(args_schema=RerankInput)
def reranker(
    faq_results: list[RetrievalItemLite],
    amenities_results: list[RetrievalItemLite],
    services_results: list[RetrievalItemLite],
    ) -> list[RetrievalItemLite]:
    """Rerank combined results from earlier tool calls according to similarity score.

    Args:
        faq_results (list[RetrievalItemLite]): The results of the call to query_faq
        amenities_results (list[RetrievalItemLite]): The results of the call to query_amenities
        services_results (list[RetrievalItemLite]): The results of the call to query_services

    Returns:
        list[RetrievalItemLite]: The top_k results

    """
    all_results = faq_results + amenities_results + services_results
    return heapq.nlargest(top_k, all_results, key=lambda x: x.score)

class HydrateInput(BaseModel):
    items: list[RetrievalItemLite] = Field(..., description="Top items from reranker")

PREFIX_BY_SOURCE = {
    "faq": "FAQ",
    "amenities": "Amenities",
    "services": "Services",
}

@tool(args_schema=HydrateInput)
def hydrate_items(items: list[RetrievalItemLite]) -> list[RetrievalItem]:
    """Hydrate lite reranked items into full objects (text + metadata)."""
    out: list[RetrievalItem] = []
    for it in items:
        prefix = PREFIX_BY_SOURCE[it.source.lower()]
        data_list = redis_client.hmget(f"{prefix}:{it.item_id}",
                                       ["_node_content", "text"])

        # Fallback: if you can't docstore-get by node_id, you need a deterministic fetch strategy.
        # One common fallback is to store a stable 'doc_id' in metadata and filter retrieval by it.
        if data_list is None:
            error_str = f"Could not hydrate item_id={it.item_id} from source={it.source}."
            raise RuntimeError(error_str)

        data = json.loads(data_list[0])
        text = data_list[1]

        out.append(
            RetrievalItem(
                source=it.source,
                metadata=data["metadata"],
                text=text or "",
                score=it.score,
            ),
        )
    return out

system_prompt = f"""You are a hotel information assistant. You answer user questions
only using information returned by the provided retrieval and reranking tools.

Available tools
- query_faq - retrieves up to {top_k} FAQ nodes
- query_amenities - retrieves up to {top_k} amenity nodes
- query_services - retrieves up to {top_k} service nodes
- reranker - requires THREE arguments:
  - faq_results: output list from query_faq
  - amenities_results: output list from query_amenities
  - services_results: output list from query_services
  It returns a list of the the top {top_k} nodes.
- hydrate_items - takes the list of top nodes and retrieves the associated text and metadata

Non-negotiable rules
- Always call all three retrieval tools for every user query.
- Always call the reranker after collecting retrieval results.
- The final results must come exclusively from the reranker's output.
- Do not invent, infer, or embellish information that is not explicitly present in retrieved nodes.
- Do not mention internal implementation details (RAG, embeddings, vector stores, scores, or node IDs).
- You may say you "looked it up" or "searched."
- Your answer is not exhaustive; never claim completeness.
- Do not offer to perform actions (booking, calling, emailing, etc.) unless explicitly stated in retrieved content.
- Assume prices are USD unless otherwise stated.

Mandatory workflow (must follow exactly)
1. Parse the user request into:
   - primary intent/topic
   - constraints (price, duration, hours, availability, booking, notice)
   - relevant keywords and synonyms
2. Query all three retrieval tools using the parsed intent.
3. Call reranker with named arguments exactly like:
   reranker(faq_results=<output of query_faq>,
            amenities_results=<output of query_amenities>,
            services_results=<output of query_services>).
4. Call hydrate_items with the top {top_k} nodes from the reranker.
5. Receive the final results (nodes with text and metadata).
6. Use only those nodes to construct the final response.
   - If the information in a node is irrelevant, exclude it.

De-duplication and conflicts
- If multiple reranked nodes convey the same information:
  - Merge them into a single entry without losing details.
- If reranked nodes conflict:
  - Present both facts clearly and without reconciliation unless the data explicitly resolves the conflict.

Response formatting
- You may provide a once sentence introduction to the results. List the relevant results afterwards.
- If an FAQ result is relevant to the user's query, place it at the top of the results.
- Present results in a clean, scannable format.
- Include only information explicitly present in the node.
- Place the field information to the right of the field heading.
- If a field is missing, write "Not specified."
- Do not add advice, policies, or instructions not present in the retrieved content.

Explicit formatting by type
- FAQ results should be presented as a one or two sentence response to the user's query.

- Amenity results must be formatted eactly as:
**<Name>**
  **Category:**
  **Price (USD):**
  **Duration (minutes):**
  **Availability:**
  **Booking required:**
  **Minimum notice (hours):**
  **Description:**

- Service results must be formatted exactly as:
**<Name>**
  **Service type:**
  **Department:**
  **Price (USD):**
  **Duration (minutes):**
  **Booking required:**
  **Minimum notice (hours):**
  **Description:**

Handling missing or weak matches
- If fewer than {top_k} reranked nodes are meaningfully relevant:
  - Return fewer results and clearly state what could not be found.
- If the user request is ambiguous:
  - Present the reranked results as-is.

Ending
After the response, end with exactly the following text:
"Is there any other information I can provide about the hotel? I can also help finding and booking a room."
"""

def get_info_agent() -> CompiledStateGraph:
    return create_agent(
        model=llm,
        tools=[query_faq, query_amenities, query_services, reranker, hydrate_items],
        system_prompt=system_prompt,
        )
