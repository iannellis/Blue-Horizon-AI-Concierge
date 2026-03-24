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
import pandera.pandas as pa
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.redis import RedisVectorStore
from pandera.errors import SchemaError, SchemaErrors

from blue_horizon.agents.information.models import Source
from blue_horizon.agents.information.retrieval import build_information_index_schemas
from blue_horizon.config import load_app_config
from blue_horizon.load_data import _repo_root

if TYPE_CHECKING:
    from pathlib import Path

    from redisvl.schema import IndexSchema


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


def load_dataframe(name: str, path: Path, schema: pa.DataFrameSchema) -> pd.DataFrame:
    """Read a pickle file and validate it with a Pandera schema.

    Args:
        name: Friendly dataset name used in logging messages.
        path: Path pointing to the pickle to load.
        schema: Pandera schema used to validate and coerce the loaded DataFrame.

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

    try:
        validated_df = schema.validate(df, lazy=True)
    except (SchemaError, SchemaErrors):
        logger.exception("%s data failed schema validation", name)
        sys.exit(1)
    if validated_df.empty:
        logger.error("%s data (%s) is empty", name, path)
        sys.exit(1)
    return validated_df


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


def _to_float(value: object) -> float:
    """Convert a pandas scalar-like value into a Python float.

    Args:
        value: Scalar value produced by pandas normalization.

    Returns:
        float: Converted numeric value.

    Raises:
        TypeError: If the value is not float-convertible.

    """
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        return float(value)

    msg = f"Value is not float-convertible: {value!r}"
    raise TypeError(msg)


_FAQ_SCHEMA = pa.DataFrameSchema(
    columns={
        "question": pa.Column(str, nullable=True, coerce=True),
        "answer": pa.Column(str, nullable=True, coerce=True),
        "category": pa.Column(str, nullable=True, coerce=True),
        "subcategory": pa.Column(str, nullable=True, coerce=True),
        "keywords": pa.Column(str, nullable=True, coerce=True),
        "last_updated": pa.Column(str, nullable=True, coerce=True),
    },
    strict=False,
    coerce=True,
)
_AMENITIES_SCHEMA = pa.DataFrameSchema(
    columns={
        "name": pa.Column(str, nullable=True, coerce=True),
        "description": pa.Column(str, nullable=True, coerce=True),
        "category": pa.Column(str, nullable=True, coerce=True),
        "price": pa.Column(float, nullable=True, coerce=True),
        "duration": pa.Column(float, nullable=True, coerce=True),
        "availability": pa.Column(str, nullable=True, coerce=True),
        "booking_required": pa.Column(str, nullable=True, coerce=True),
        "min_notice_hours": pa.Column(float, nullable=True, coerce=True),
    },
    strict=False,
    coerce=True,
)
_SERVICES_SCHEMA = pa.DataFrameSchema(
    columns={
        "name": pa.Column(str, nullable=True, coerce=True),
        "description": pa.Column(str, nullable=True, coerce=True),
        "service_type": pa.Column(str, nullable=True, coerce=True),
        "duration_minutes": pa.Column(float, nullable=True, coerce=True),
        "price": pa.Column(float, nullable=True, coerce=True),
        "department": pa.Column(str, nullable=True, coerce=True),
        "booking_required": pa.Column(str, nullable=True, coerce=True),
        "min_notice_hours": pa.Column(float, nullable=True, coerce=True),
    },
    strict=False,
    coerce=True,
)


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
    normalized = df_faq.assign(
        question=df_faq["question"].fillna("").astype(str),
        answer=df_faq["answer"].fillna("").astype(str),
        category=df_faq["category"].fillna("general").astype(str),
        subcategory=df_faq["subcategory"].fillna("general").astype(str),
        keywords=df_faq["keywords"].fillna("").astype(str),
        last_updated=df_faq["last_updated"].fillna("").astype(str),
    )
    return [
        TextNode(
            text=f"Question:\n{row.question}\n\nAnswer:\n{row.answer}",
            id_=row.Index,
            metadata={
                "category": row.category,
                "subcategory": row.subcategory,
                "keywords": row.keywords,
                "last_updated": row.last_updated,
            },
        )
        for row in normalized.itertuples()
    ]


def get_amenities_nodes(df_amenities: pd.DataFrame) -> list[TextNode]:
    """Convert amenities rows into TextNodes with sanitized metadata.

    Args:
        df_amenities: DataFrame containing amenities data with required columns.

    Returns:
        List of TextNodes derived from the amenities dataset.

    """
    normalized = df_amenities.assign(
        name=df_amenities["name"].fillna("").astype(str),
        description=df_amenities["description"].fillna("").astype(str),
        category=df_amenities["category"].fillna("general").astype(str),
        price=pd.to_numeric(df_amenities["price"], errors="coerce").fillna(0.0),
        duration=pd.to_numeric(df_amenities["duration"], errors="coerce").fillna(0.0),
        availability=df_amenities["availability"].fillna("").astype(str),
        booking_required=df_amenities["booking_required"].fillna("False").astype(str),
        min_notice_hours=pd.to_numeric(
            df_amenities["min_notice_hours"],
            errors="coerce",
        ).fillna(0.0),
    )
    return [
        TextNode(
            text=f"Name:{row.name}\n\nDescription:\n{row.description}",
            id_=row.Index,
            metadata={
                "category": row.category,
                "price": _to_float(row.price),
                "duration": _to_float(row.duration),
                "availability": row.availability,
                "booking_required": row.booking_required,
                "min_notice_hours": _to_float(row.min_notice_hours),
            },
        )
        for row in normalized.itertuples()
    ]


def get_services_nodes(df_services: pd.DataFrame) -> list[TextNode]:
    """Convert services rows into TextNodes with sanitized metadata.

    Args:
        df_services: DataFrame containing services data with required columns.

    Returns:
        List of TextNodes derived from the services dataset.

    """
    normalized = df_services.assign(
        name=df_services["name"].fillna("").astype(str),
        description=df_services["description"].fillna("").astype(str),
        service_type=df_services["service_type"].fillna("general").astype(str),
        duration=pd.to_numeric(
            df_services["duration_minutes"],
            errors="coerce",
        ).fillna(0.0),
        price=pd.to_numeric(df_services["price"], errors="coerce").fillna(0.0),
        department=df_services["department"].fillna("general").astype(str),
        booking_required=df_services["booking_required"].fillna("False").astype(str),
        min_notice_hours=pd.to_numeric(
            df_services["min_notice_hours"],
            errors="coerce",
        ).fillna(0.0),
    )
    return [
        TextNode(
            text=f"Name:{row.name}\n\nDescription:\n{row.description}",
            id_=row.Index,
            metadata={
                "service_type": row.service_type,
                "duration": _to_float(row.duration),
                "price": _to_float(row.price),
                "department": row.department,
                "booking_required": row.booking_required,
                "min_notice_hours": _to_float(row.min_notice_hours),
            },
        )
        for row in normalized.itertuples()
    ]


# ============================
# Main entry point
# ============================


def main() -> None:
    """Load FAQ, services, and amenities datasets into Redis vector indices.

    Initialises the OpenAI embedding model, loads each dataset from pickled
    files, converts records into LlamaIndex TextNodes, and writes them to Redis
    via separate vector store indices.

    """
    cfg = load_app_config()
    embeddings_cfg = cfg.info.embeddings
    vector_dims = cfg.info.retrieval.vector_dims

    Settings.embed_model = OpenAIEmbedding(
        model=embeddings_cfg.model,
        embed_batch_size=embeddings_cfg.batch_size,
        timeout=embeddings_cfg.timeout_s,
        max_retries=embeddings_cfg.max_retries,
    )

    data_path = get_data_path()
    redis_conn_string = cfg.redis_url
    index_schemas = build_information_index_schemas(vector_dims=vector_dims)

    # ============================
    # Load FAQ data into Redis
    # ============================

    df_faq = load_dataframe("FAQ", data_path / "faq_knowledge_base.pkl", _FAQ_SCHEMA)
    faq_nodes = get_faq_nodes(df_faq)
    build_index(
        index_schemas[Source.FAQ],
        faq_nodes,
        Source.FAQ.value,
        redis_conn_string,
    )

    # ============================
    # Load amenities data into Redis
    # ============================

    df_amenities = load_dataframe(
        "Amenities",
        data_path / "amenities.pkl",
        _AMENITIES_SCHEMA,
    )
    amenities_nodes = get_amenities_nodes(df_amenities)
    build_index(
        index_schemas[Source.AMENITIES],
        amenities_nodes,
        Source.AMENITIES.value,
        redis_conn_string,
    )

    # ============================
    # Load services data into Redis
    # ============================

    df_services = load_dataframe(
        "Services",
        data_path / "services.pkl",
        _SERVICES_SCHEMA,
    )
    services_nodes = get_services_nodes(df_services)
    build_index(
        index_schemas[Source.SERVICES],
        services_nodes,
        Source.SERVICES.value,
        redis_conn_string,
    )


if __name__ == "__main__":
    main()
