import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.redis import RedisVectorStore
from redisvl.schema import IndexSchema

load_dotenv("../.env")

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

redis_conn_string = os.getenv("REDIS_URL")
schema = IndexSchema.from_dict(
    {
        "index": {"name": "blue_horizon", "prefix": "blue_horizon"},
        # customize fields that are indexed
        "fields": [
            # required fields for llamaindex
            {"type": "tag", "name": "id"},
            {"type": "tag", "name": "doc_id"},
            {"type": "text", "name": "text"},
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

vector_store = RedisVectorStore(schema=schema, redis_url=redis_conn_string, overwrite=False)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_vector_store(vector_store=vector_store, storage_context=storage_context)
retriever = index.as_retriever(similarity_top_k=4)

llm = ChatOpenAI(model="gpt-5.1", temperature=0)

system_prompt = """You are an assistant who helps people find out information about a hotel.
Your sole job is to query the database for information about the hotel using the tool provided.
Provide the information returned from the tool that is relevant to the user's query in a
well-formatted manner. If you decide to include an item and it has a description, be
sure to include that description.

Do not offer to do anything specific for the user. After you have answered the user's
query, simply ask if there is any other information you can provide about the hotel.

Do not imply that your results are exhaustive.

Assume that prices are in dollars.

Do not mention that you are searching a database, but you may mention that you are or
have perfomed a search.

Do not provide any instructions to the user concerning the hotel that were not provided
to you.
"""

@tool(parse_docstring=True)
def query_hotel_info(query: str) -> list[dict]:
    """Provide information about the hotel in response to a passed-in query.

    The query should be concise and not ask for many details.
    Accesses an FAQ database, information about hotel amenities, and information about
    hotel services. Nothing else.

    Args:
        query (string): The query string

    Returns:
        list[dict]: The retrieved results and their associated metadata
    """
    retrieved_nodes = retriever.retrieve(query)

    return [{"metadata": node.metadata, "text": node.text} for node in retrieved_nodes]

def get_info_agent() -> CompiledStateGraph:
    return create_agent(model=llm, tools=[query_hotel_info], system_prompt=system_prompt)
