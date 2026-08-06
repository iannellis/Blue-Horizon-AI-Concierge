"""Routing schema and LangGraph state for the orchestration agent."""

from __future__ import annotations

from typing import Literal, cast

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

type RouteStep = Literal["info", "booking", "refuse", "error"]


class RouteDecision(BaseModel):
    """Structured router output.

    Attributes:
        step: The next step to take in answering the user's query.

    Notes:
        - This model is used with ChatOpenAI.with_structured_output() to request a
          typed response from the router.
        - The router prompt should constrain outputs to the RouteStep literal.

    """

    step: RouteStep = Field(
        ...,
        description="The next step to take in answering the user's query.",
    )


class ConversationState(MessagesState, total=False):
    """LangGraph state with persisted message history.

    Attributes:
        messages: Message history (provided by MessagesState).
        route: Router decision.

    """

    route: RouteStep


def _route_from_state(state: ConversationState) -> RouteStep:
    """Select the next node to execute based on state.

    Args:
        state: Current LangGraph state.

    Returns:
        Route step key for conditional edges.

    """
    return cast("RouteStep", state.get("route") or "error")
