"""Orchestration agent package.

Re-exports the public API so callers can use the same import paths as before::

    from blue_horizon.agents.orchestration import OrchestrationManager
    from blue_horizon.agents.orchestration import format_chat_response
"""

from blue_horizon.agents.orchestration.factory import build_orchestration_agent
from blue_horizon.agents.orchestration.formatting import format_chat_response
from blue_horizon.agents.orchestration.manager import OrchestrationManager
from blue_horizon.agents.orchestration.models import (
    ConversationState,
    RouteDecision,
    RouteStep,
)
from blue_horizon.agents.orchestration.resources import OrchestrationResources

__all__ = [
    "ConversationState",
    "OrchestrationManager",
    "OrchestrationResources",
    "RouteDecision",
    "RouteStep",
    "build_orchestration_agent",
    "format_chat_response",
]
