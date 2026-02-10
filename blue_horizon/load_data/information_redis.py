"""Load FAQ, services, and amenities datasets into Redis vector indices.

Each record is turned into a LlamaIndex TextNode, embedded with OpenAI, and stored
in Redis via a vector store. Guards ensure configuration, files, and schema are
validated before indices are created.

"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

import pandas as pd
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.redis import RedisVectorStore
from redisvl.schema import IndexSchema

from blue_horizon.config import load_app_config
from blue_horizon.load_data import _repo_root

if TYPE_CHECKING:
    from collections.abc import Hashable
    from pathlib import Path


logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def get_data_path() -> Path:
    """Return the configured information data path.

    Returns:
        Path: Absolute path to the information data folder.

    """
    app_config = load_app_config()
    return (_repo_root() / app_config.load_data.information_redis.data_path).resolve()


def load_dataframe(name: str, path: Path, required_columns: list[str]) -> pd.DataFrame:
    """Read a pickle file and validate required columns.

    Args:
        name: Friendly dataset name used in logging messages.
        path: Path pointing to the pickle to load.
        required_columns: Columns that must exist on the loaded DataFrame.

    Returns:
        The loaded DataFrame.

    Raises:
        SystemExit: If the file is missing, unreadable, missing columns, or empty.

    """
    if not path.is_file():
        logger.error("%s data file not found at %s", name, path)
        sys.exit(1)
    try:
        df = pd.read_pickle(path)  # noqa: S301
    except (FileNotFoundError, OSError):
        logger.exception("Unable to read %s data from %s", name, path)
        sys.exit(1)
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(
            "%s data is missing required columns %s (found %s)",
            name,
            missing_columns,
            list(df.columns),
        )
        sys.exit(1)
    if df.empty:
        logger.error("%s data (%s) is empty", name, path)
        sys.exit(1)
    return df


def safe_value(row: pd.Series, key: str, row_id: Hashable, default: str = "") -> str:
    """Return a string value, falling back to a default when missing.

    Args:
        row: The row being inspected.
        key: Column name to pull from the row.
        row_id: Identifier used for logging context.
        default: Value returned when the row does not contain the key or it is NaN.

    Returns:
        Value from `row[key]` or `default`.

    """
    value = row.get(key, default)
    if pd.isna(value):
        logger.warning("Row %s missing %s, defaulting to %s", row_id, key, default)
        return default
    return value


def safe_numeric(
    row: pd.Series, key: str, row_id: Hashable, default: float = 0,
) -> float:
    """Return a numeric value, coercing to float or falling back.

    Args:
        row: The row being inspected.
        key: Column name to read.
        row_id: Identifier used for logging context.
        default: Value returned when the key is missing, NaN, or not castable.

    Returns:
        Float cast of the column or `default`.

    """
    value = row.get(key, default)
    if pd.isna(value):
        logger.warning("Row %s has no %s, defaulting to %s", row_id, key, default)
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        logger.warning(
            "Row %s has invalid %s (%s), defaulting to %s",
            row_id,
            key,
            value,
            default,
        )
        return default


def build_index(
    schema: IndexSchema,
    nodes: list[TextNode],
    label: str,
    redis_url: str,
) -> VectorStoreIndex:
    """Create a Redis vector store index with the provided schema.

    Args:
        schema: Schema describing Redis fields and vectors.
        nodes: TextNodes to load into the index.
        label: Human-readable name for logging.
        redis_url: Redis connection string.

    Returns:
        VectorStoreIndex wrapping the newly built vector store.

    Raises:
        SystemExit: If Redis or index creation fails.

    """
    try:
        vector_store = RedisVectorStore(
            schema=schema,
            redis_url=redis_url,
            overwrite=True,
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return VectorStoreIndex(nodes=nodes, storage_context=storage_context)
    except Exception:
        logger.exception("Could not build %s Redis index", label)
        sys.exit(1)


FAQ_REQUIRED_COLUMNS = [
    "question",
    "answer",
    "category",
    "subcategory",
    "keywords",
    "last_updated",
]
AMENITIES_REQUIRED_COLUMNS = [
    "name",
    "description",
    "category",
    "price",
    "duration",
    "availability",
    "booking_required",
    "min_notice_hours",
]
SERVICES_REQUIRED_COLUMNS = [
    "name",
    "description",
    "service_type",
    "duration_minutes",
    "price",
    "department",
    "booking_required",
    "min_notice_hours",
]


# ============================
# Node builders
# ============================


def get_faq_nodes(df_faq: pd.DataFrame) -> list[TextNode]:
    """Convert each FAQ row into a TextNode with metadata.

    Args:
        df_faq: DataFrame containing FAQ data with required columns.

    Returns:
        List of TextNodes derived from the FAQ dataset.

    """
    nodes = []
    for id_, row in df_faq.iterrows():
        question = safe_value(row, "question", id_, default="")
        answer = safe_value(row, "answer", id_, default="")
        main_text = "Question:\n" + question + "\n\nAnswer:\n" + answer
        metadata = {
            "category": safe_value(row, "category", id_, default="general"),
            "subcategory": safe_value(row, "subcategory", id_, default="general"),
            "keywords": safe_value(row, "keywords", id_, default=""),
            "last_updated": safe_value(row, "last_updated", id_, default=""),
        }
        node = TextNode(text=main_text, id_=id_, metadata=metadata)
        nodes.append(node)
    return nodes


def get_amenities_nodes(df_amenities: pd.DataFrame) -> list[TextNode]:
    """Convert amenities rows into TextNodes with sanitized metadata.

    Args:
        df_amenities: DataFrame containing amenities data with required columns.

    Returns:
        List of TextNodes derived from the amenities dataset.

    """
    nodes = []
    for id_, row in df_amenities.iterrows():
        name = safe_value(row, "name", id_, default="")
        description = safe_value(row, "description", id_, default="")
        main_text = "Name:" + name + "\n\nDescription:\n" + description
        metadata = {
            "category": safe_value(row, "category", id_, default="general"),
            "price": safe_numeric(row, "price", id_, default=0),
            "duration": safe_numeric(row, "duration", id_, default=0),
            "availability": safe_value(row, "availability", id_, default=""),
            "booking_required": str(
                safe_value(row, "booking_required", id_, default="False"),
            ),
            "min_notice_hours": safe_numeric(row, "min_notice_hours", id_, default=0),
        }
        node = TextNode(text=main_text, id_=id_, metadata=metadata)
        nodes.append(node)
    return nodes


def get_services_nodes(df_services: pd.DataFrame) -> list[TextNode]:
    """Convert services rows into TextNodes with sanitized metadata.

    Args:
        df_services: DataFrame containing services data with required columns.

    Returns:
        List of TextNodes derived from the services dataset.

    """
    nodes = []
    for id_, row in df_services.iterrows():
        name = safe_value(row, "name", id_, default="")
        description = safe_value(row, "description", id_, default="")
        main_text = "Name:" + name + "\n\nDescription:\n" + description
        metadata = {
            "service_type": safe_value(row, "service_type", id_, default="general"),
            "duration": safe_numeric(row, "duration_minutes", id_, default=0),
            "price": safe_numeric(row, "price", id_, default=0),
            "department": safe_value(row, "department", id_, default="general"),
            "booking_required": str(
                safe_value(row, "booking_required", id_, default="False"),
            ),
            "min_notice_hours": safe_numeric(row, "min_notice_hours", id_, default=0),
        }
        node = TextNode(text=main_text, id_=id_, metadata=metadata)
        nodes.append(node)
    return nodes


# ============================
# Main entry point
# ============================


def main() -> None:
    """Load FAQ, services, and amenities datasets into Redis vector indices.

    Initialises the OpenAI embedding model, loads each dataset from pickled
    files, converts records into LlamaIndex TextNodes, and writes them to Redis
    via separate vector store indices.

    """
    redis_conn_string = load_app_config().redis_url
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

    data_path = get_data_path()

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

    df_faq = load_dataframe(
        "FAQ",
        data_path / "faq_knowledge_base.pkl",
        FAQ_REQUIRED_COLUMNS,
    )
    faq_nodes = get_faq_nodes(df_faq)
    build_index(faq_schema, faq_nodes, "faq", redis_conn_string)

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

    df_amenities = load_dataframe(
        "Amenities",
        data_path / "amenities.pkl",
        AMENITIES_REQUIRED_COLUMNS,
    )
    amenities_nodes = get_amenities_nodes(df_amenities)
    build_index(amenities_schema, amenities_nodes, "amenities", redis_conn_string)

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

    df_services = load_dataframe(
        "Services",
        data_path / "services.pkl",
        SERVICES_REQUIRED_COLUMNS,
    )
    services_nodes = get_services_nodes(df_services)
    build_index(services_schema, services_nodes, "services", redis_conn_string)


if __name__ == "__main__":
    main()
