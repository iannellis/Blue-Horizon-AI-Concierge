"""Routing schema and LangGraph state for the orchestration agent."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal, TypedDict

# LangGraph resolves the state's annotations at runtime, so the message
# classes they name must be importable then, not only under TYPE_CHECKING.
from langchain_core.messages import AIMessage, BaseMessage  # noqa: TC002
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from blue_horizon.config import MessagesConfig

type RouteStep = Literal["info", "booking", "refuse"]
type PartStep = Literal["info", "booking"]
type TurnErrorCode = Literal["timeout", "internal", "unavailable", "quota"]
type FlaggedText = Literal["message", "reply", "injection"]


class RoutePart(BaseModel):
    """One piece of the guest's message and the step that should answer it.

    Attributes:
        step: The sub-agent that should answer it, or ``"refuse"``.
        query: That piece of the message, restated to stand on its own.

    Notes:
        - ``step`` is declared before ``query``. With the query first, the
          router refused injection attempts less often, sending some to a
          sub-agent instead.

    """

    step: RouteStep = Field(
        ...,
        description="The step that should answer this part.",
    )
    query: str = Field(
        ...,
        description=(
            "This part of the guest's latest message, restated so it can be "
            "understood without the rest of the message."
        ),
    )


class RouteDecision(BaseModel):
    """Structured router output.

    Attributes:
        parts: One entry per sub-agent the guest's message needs, in the
            order the guest asked. A single-intent message has one part.

    Notes:
        - This model is used with ChatOpenAI.with_structured_output() to request a
          typed response from the router.
        - The router prompt should constrain each part's step to the RouteStep
          literal.

    """

    parts: list[RoutePart] = Field(
        ...,
        description=(
            "One entry per agent the guest's message needs, in the order asked."
        ),
    )


class InjectionCheck(BaseModel):
    """Structured output of the injection check.

    Attributes:
        injection: Whether any part of the guest's latest message tries to
            manipulate the assistant instead of asking for hotel help.

    """

    injection: bool


class PartInput(TypedDict):
    """What the fan-out sends one sub-agent node for one part.

    Attributes:
        messages: The earlier history, then the part's own message: the
            guest's message itself for a single-part turn, otherwise the
            part's standalone query.
        index: The part's position in the guest's message.

    """

    messages: list[BaseMessage]
    index: int


class PartReply(TypedDict):
    """One sub-agent's result for one part of the turn.

    Attributes:
        index: The part's position in the guest's message.
        step: The sub-agent that answered it.
        replies: The sub-agent's final replies, without tool calls. Empty
            when the part failed.
        turn_error: Why the part failed, or ``None`` if it did not.

    """

    index: int
    step: PartStep
    replies: list[AIMessage]
    turn_error: TurnErrorCode | None


def _merge_part_replies(
    current: list[PartReply],
    update: list[PartReply] | None,
) -> list[PartReply]:
    """Reduce ``part_replies`` writes from parallel sub-agent nodes.

    Args:
        current: Replies collected so far.
        update: Replies from one node, or ``None`` to clear the channel.

    Returns:
        ``current`` followed by ``update``, or an empty list when ``update``
        is ``None``.

    """
    return [] if update is None else [*current, *update]


class ConversationState(MessagesState, total=False):
    """LangGraph state with persisted message history.

    Attributes:
        messages: Message history (provided by MessagesState).
        route: The router's decision: the parts' steps joined by ``"+"``
            in the order the guest asked (``"info"``, ``"booking"``,
            ``"booking+info"``, ``"info+booking"``), or ``"refuse"`` or
            ``"error"``.
        parts: The router's split of the guest's message, one part per
            sub-agent, refusals removed.
        part_replies: One `PartReply` per dispatched part. The router clears
            it before the fan-out and ``finalize`` clears it again, so a
            reply from an earlier turn never reaches the join.
        failed_parts: The steps whose part failed on a turn that did not
            fail as a whole, so the guest read the other part's reply.
            Reset at the start of every turn.
        turn_error: Why the current turn failed, or ``None`` if it has not.
            ``"timeout"`` when the router or a sub-agent exceeded its
            wall-clock cap, ``"unavailable"`` when the booking agent could not
            reach its database, ``"quota"`` when the model provider refused
            the request because the project's budget or credit is spent,
            ``"internal"`` for any other failure, including a
            turn that produced no reply. A turn with several parts fails as
            a whole only when every part failed. The first node of every
            turn resets it, so a value from an earlier turn never leaks
            forward. A failed turn is reported to the client as an ``error``
            event, not as an apology written into history.
        flagged: Which check flagged the current turn, or ``None`` if none
            did. ``"message"`` when moderation flagged the guest's message
            and the turn was refused before the router ran, ``"injection"``
            when the injection check flagged it and the whole turn was
            refused, ``"reply"`` when moderation flagged a sub-agent's reply
            and it was replaced with the refusal. Reset at the start of
            every turn.
        injection_error: Why the injection check failed, or ``None`` if it
            did not. Kept apart from ``turn_error`` because the check runs
            in the same step as the router, which writes ``turn_error``
            itself; ``injection_gate`` merges the two. Reset at the start
            of every turn.

    """

    route: str
    parts: list[RoutePart]
    part_replies: Annotated[list[PartReply], _merge_part_replies]
    failed_parts: list[PartStep]
    turn_error: TurnErrorCode | None
    flagged: FlaggedText | None
    injection_error: TurnErrorCode | None


def turn_error_message(messages: MessagesConfig, turn_error: str) -> str:
    """Return the guest-facing copy for a failed turn or a failed part.

    Shared by the streaming ``error`` event, the JSON branch's failure
    body, and the join node's copy for one failed part of a turn that
    otherwise succeeded, so they cannot drift apart.

    Args:
        messages: Configured orchestration message strings.
        turn_error: The failed turn's code.

    Returns:
        ``messages.database_unavailable`` for ``"unavailable"``,
        ``messages.quota_exhausted`` for ``"quota"``, otherwise
        ``messages.error``.

    """
    if turn_error == "unavailable":
        return messages.database_unavailable
    if turn_error == "quota":
        return messages.quota_exhausted
    return messages.error
