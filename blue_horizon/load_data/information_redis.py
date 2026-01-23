"""Script that loads the FAQ, services, and amenities data into Redis.

Loads to FAQ, services, and amenities data from Pandas binary fines. Load each row into
a LlamaIndex TextNode. Then, using LlamaIndex, load it into Redis with vector
embeddings.

"""

import logging
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.redis import RedisVectorStore
from redisvl.schema import IndexSchema

load_dotenv()


logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

DATA_PATH = Path(__file__).parents[2] / "data/pandas/"

redis_conn_string = os.getenv("REDIS_URL")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# ============================
# Load FAQ data into Redis
# ============================

faq_schema = IndexSchema.from_dict(
    {
        "index": {"name": "faq", "prefix": "faq"},
        # customize fields that are indexed
        "fields": [
            # required fields for llamaindex
            {"type": "tag", "name": "id"},
            {"type": "tag", "name": "doc_id"},
            {"type": "text", "name": "text"},
            # custom for metadata filtering
            {"type": "tag", "name": "category"},
            # custom vector field for text-embedding-3-small
            {
                "type": "vector",
                "name": "vector",
                "attrs": {
                    "dims": 1536,
                    "algorithm": "hnsw",
                    "distance_metric": "cosine",
                },
            },
        ],
    },
)


df_faq = pd.read_pickle(DATA_PATH / "faq_knowledge_base.pkl")  # noqa: S301


def get_faq_nodes() -> list:
    """Load the FAQ data into the LlamaIndex TextNode format."""
    nodes = []
    for id_, row in df_faq.iterrows():
        main_text = "Question:\n" + row["question"] + "\n\nAnswer:\n" + row["answer"]
        metadata = {
            "category": row["category"],
            "subcategory": row["subcategory"],
            "keywords": row["keywords"],
            "last_updated": row["last_updated"],
        }
        node = TextNode(text=main_text, id_=id_, metadata=metadata)
        nodes.append(node)
    return nodes


faq_nodes = get_faq_nodes()
faq_vector_store = RedisVectorStore(
    schema=faq_schema, redis_url=redis_conn_string, overwrite=True,
)
faq_storage_context = StorageContext.from_defaults(vector_store=faq_vector_store)
faq_index = VectorStoreIndex(nodes=faq_nodes, storage_context=faq_storage_context)

# ============================
# Load amenities data into Redis
# ============================

amenities_schema = IndexSchema.from_dict(
    {
        "index": {"name": "amenities", "prefix": "amenities"},
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
            # custom vector field for text-embedding-3-small embeddings
            {
                "type": "vector",
                "name": "vector",
                "attrs": {
                    "dims": 1536,
                    "algorithm": "hnsw",
                    "distance_metric": "cosine",
                },
            },
        ],
    },
)

df_amenities = pd.read_pickle(DATA_PATH / "amenities.pkl")  # noqa: S301


def get_amenities_nodes() -> list[TextNode]:
    """Load the amenities data into the LlamaIndex TextNode format."""
    nodes = []
    for id_, row in df_amenities.iterrows():
        main_text = "Name:" + row["name"] + "\n\nDescription:\n" + row["description"]
        metadata = {
            "category": row["category"],
            "price": row["price"],
            "duration": row["duration"],
            "availability": row["availability"],
            "booking_required": str(row["booking_required"]),
            "min_notice_hours": row["min_notice_hours"],
        }
        node = TextNode(text=main_text, id_=id_, metadata=metadata)
        nodes.append(node)
    return nodes


amenities_nodes = get_amenities_nodes()
amenities_vector_store = RedisVectorStore(
    schema=amenities_schema, redis_url=redis_conn_string, overwrite=True,
)
amenities_storage_context = StorageContext.from_defaults(
    vector_store=amenities_vector_store,
)
amenities_index = VectorStoreIndex(
    nodes=amenities_nodes, storage_context=amenities_storage_context,
)


# ============================
# Load services data into Redis
# ============================

services_schema = IndexSchema.from_dict(
    {
        "index": {"name": "services", "prefix": "services"},
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
            # custom vector field for text-embedding-3-small embeddings
            {
                "type": "vector",
                "name": "vector",
                "attrs": {
                    "dims": 1536,
                    "algorithm": "hnsw",
                    "distance_metric": "cosine",
                },
            },
        ],
    },
)


df_services = pd.read_pickle(DATA_PATH / "services.pkl")  # noqa: S301


def get_services_nodes() -> list[TextNode]:
    """Load the services data into the LlamaIndex TextNode format."""
    nodes = []
    for id_, row in df_services.iterrows():
        main_text = "Name:" + row["name"] + "\n\nDescription:\n" + row["description"]
        metadata = {
            "service_type": row["service_type"],
            "duration": row["duration_minutes"],
            "price": row["price"],
            "department": row["department"],
            "booking_required": str(row["booking_required"]),
            "min_notice_hours": row["min_notice_hours"],
        }
        node = TextNode(text=main_text, id_=id_, metadata=metadata)
        nodes.append(node)
    return nodes


services_nodes = get_services_nodes()
services_vector_store = RedisVectorStore(
    schema=services_schema, redis_url=redis_conn_string, overwrite=True,
)
services_storage_context = StorageContext.from_defaults(
    vector_store=services_vector_store,
)
services_index = VectorStoreIndex(
    nodes=services_nodes, storage_context=services_storage_context,
)
