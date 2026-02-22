"""Information RAG agent package.

Re-exports the public API so callers can use the same import paths as before:

    from blue_horizon.agents.information import InfoRagResources, InfoAgentFactory
"""

from blue_horizon.agents.information.config import load_info_config
from blue_horizon.agents.information.factory import InfoAgentFactory
from blue_horizon.agents.information.models import (
    INFO_AGENT_ENDING,
    InfoState,
    ParsedQuery,
    ParsedState,
    RerankInput,
    RetrievalItem,
    Source,
)
from blue_horizon.agents.information.resources import InfoRagResources, VectorIndexes
from blue_horizon.agents.information.retrieval import build_filters, build_index_schema

__all__ = [
    "INFO_AGENT_ENDING",
    "InfoAgentFactory",
    "InfoRagResources",
    "InfoState",
    "ParsedQuery",
    "ParsedState",
    "RerankInput",
    "RetrievalItem",
    "Source",
    "VectorIndexes",
    "build_filters",
    "build_index_schema",
    "load_info_config",
]
