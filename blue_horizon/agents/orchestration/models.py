"""Routing schema and LangGraph state for the orchestration agent."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from blue_horizon.config import MessagesConfig

type RouteStep = Literal["info", "booking", "refuse", "error"]
type TurnErrorCode = Literal["timeout", "internal", "unavailable"]


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
        turn_error: Why the current turn failed, or ``None`` if it has not.
            ``"timeout"`` when the router or a sub-agent exceeded its
            wall-clock cap, ``"unavailable"`` when the booking agent could not
            reach its database, ``"internal"`` for any other failure, including a
            turn that produced no reply. The router resets it at the start of
            every turn, so a value from an earlier turn never leaks forward.
            A failed turn is reported to the client as an ``error`` event, not
            as an apology written into history.

    """

    route: RouteStep
    turn_error: TurnErrorCode | None


def _route_from_state(state: ConversationState) -> RouteStep:
    """Select the next node to execute based on state.

    Args:
        state: Current LangGraph state.

    Returns:
        Route step key for conditional edges.

    """
    return cast("RouteStep", state.get("route") or "error")


def turn_error_message(messages: MessagesConfig, turn_error: str) -> str:
    """Return the guest-facing copy for a failed turn.

    Shared by the streaming ``error`` event and the JSON branch's failure
    body, so the two cannot drift apart.

    Args:
        messages: Configured orchestration message strings.
        turn_error: The failed turn's code.

    Returns:
        ``messages.database_unavailable`` for ``"unavailable"``, otherwise
        ``messages.error``.

    """
    if turn_error == "unavailable":
        return messages.database_unavailable
    return messages.error
