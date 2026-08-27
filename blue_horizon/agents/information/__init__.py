"""Information RAG agent package.

Re-exports the public API so callers can use the same import paths as before:

    from blue_horizon.agents.information import InfoRagResources, build_info_agent
"""

from blue_horizon.agents.information.config import load_info_config
from blue_horizon.agents.information.factory import build_info_agent
from blue_horizon.agents.information.models import (
    InfoState,
    ParsedQuery,
    ParsedState,
    RetrievalItem,
    Source,
)
from blue_horizon.agents.information.resources import InfoRagResources, VectorIndexes
from blue_horizon.agents.information.retrieval import build_filters, build_index_schema

__all__ = [
    "InfoRagResources",
    "InfoState",
    "ParsedQuery",
    "ParsedState",
    "RetrievalItem",
    "Source",
    "VectorIndexes",
    "build_filters",
    "build_index_schema",
    "build_info_agent",
    "load_info_config",
]
