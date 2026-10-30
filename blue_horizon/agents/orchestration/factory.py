"""Factory for compiling the orchestration LangGraph."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Literal, cast

import openai
from langchain_core.exceptions import ModelConnectionError, ModelTimeoutError
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    filter_messages,
)
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from blue_horizon.agents.booking import database_unavailable_this_turn
from blue_horizon.agents.booking.write_ops import BookingUnavailableError
from blue_horizon.agents.orchestration.models import (
    ConversationState,
    RouteStep,
    _route_from_state,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from langchain_core.messages import BaseMessage
    from langchain_core.runnables import RunnableConfig
    from langgraph.graph.state import CompiledStateGraph

    from blue_horizon.agents.orchestration.models import TurnErrorCode
    from blue_horizon.agents.orchestration.resources import OrchestrationResources

logger = logging.getLogger(__name__)

# Exceptions that mean a network dependency could not be reached. `OSError`
# covers a failed DNS lookup or refused connect at the root of any client
# library's chain, including embeddings and Redis calls inside a sub-agent.
_NETWORK_ERRORS = (ModelConnectionError, ModelTimeoutError, OSError)

# Marks an OpenAI 429 that no retry will cure: a project or organization
# spend limit, or an exhausted prepaid balance. Current responses carry it as
# `type` with a `code` naming which limit, which is logged rather than
# matched. Older responses carried it as `code`.
_QUOTA_ERROR_MARKER = "insufficient_quota"

# Replaces a flagged guest message in history, so neither the router nor a
# sub-agent reads the flagged text on a later turn.
_WITHHELD_MESSAGE = "[Message withheld: flagged by moderation]"

type _AfterModeration = Literal["router", "refuse", "error"]


def build_orchestration_agent(  # noqa: C901, PLR0915
    *,
    resources: OrchestrationResources,
) -> CompiledStateGraph:
    """Compile and return the orchestration graph using initialized resources.

    Contract:
        - Assumes OrchestrationResources.startup_check() has completed successfully.
        - Does not perform I/O or network checks.
        - Returns a CompiledStateGraph that can be shared concurrently.

    Args:
        resources: Initialized orchestration resources.

    Returns:
        Compiled orchestration state graph.

    """
    cfg = resources.config

    def _make_dispatch_node(
        agent_name: str,
        get_agent: Callable[[], CompiledStateGraph],
        timeout_s: float,
        failure_check: Callable[[dict[str, Any]], TurnErrorCode | None] | None = None,
    ) -> Callable[..., Awaitable[dict[str, Any]]]:
        """Create a dispatch node for a sub-agent.

        Args:
            agent_name: Human-readable name for logging.
            get_agent: Callable returning the compiled sub-agent.
            timeout_s: Wall-clock timeout in seconds.
            failure_check: Optional inspection of the sub-agent's completed
                result, returning a ``turn_error`` code when its reply must be
                discarded even though the sub-agent returned normally.

        Returns:
            Async node function suitable for LangGraph.

        """

        async def _node(
            state: ConversationState,
            config: RunnableConfig,
        ) -> dict[str, Any]:
            """Dispatch a request to a sub-agent with a timeout.

            Args:
                state: Current conversation state.
                config: LangGraph runnable config.

            Returns:
                State patch from the sub-agent, or a ``turn_error`` on failure,
                including one ``failure_check`` reports. A failure writes no
                message: see `finalize_node` for why.

            """
            logger.info("Dispatching to %s agent", agent_name)
            try:
                result = await asyncio.wait_for(
                    get_agent().ainvoke(state, config=config),
                    timeout=timeout_s,
                )
            except TimeoutError:
                logger.warning(
                    "%s agent timed out after %s s",
                    agent_name,
                    timeout_s,
                )
                return {"turn_error": "timeout"}
            except Exception as exc:  # noqa: BLE001
                return {
                    "turn_error": _record_turn_failure(f"{agent_name} agent", exc),
                }
            patch = cast("dict[str, Any]", result)
            if failure_check is not None:
                turn_error = failure_check(patch)
                if turn_error is not None:
                    return {"turn_error": turn_error}
            return patch

        _node.__name__ = _node.__qualname__ = f"{agent_name}_node"
        return _node

    async def moderate_message_node(state: ConversationState) -> dict[str, Any]:
        """Check the guest's message before any model reads it.

        This is the first node of every turn, so it also clears
        ``turn_error`` and ``flagged``: an earlier turn's values, still held
        in the checkpoint, must not be reported again.

        Args:
            state: Current conversation state, ending with the guest's
                message.

        Returns:
            State patch with ``turn_error`` and ``flagged`` reset. A flagged
            message also sets ``flagged`` to ``"message"``, routes the turn
            to ``refuse``, and replaces the message in history with a
            placeholder carrying the same id. A failed check sets
            ``turn_error`` and the ``error`` route instead.

        """
        guest_message = state["messages"][-1]
        try:
            flagged = await resources.moderator.flagged(
                guest_message.text,
                source="guest message",
            )
        except Exception as exc:  # noqa: BLE001
            return {
                "route": "error",
                "turn_error": _record_turn_failure("Message moderation", exc),
                "flagged": None,
            }
        if not flagged:
            return {"turn_error": None, "flagged": None}
        return {
            "messages": [HumanMessage(content=_WITHHELD_MESSAGE, id=guest_message.id)],
            "route": "refuse",
            "turn_error": None,
            "flagged": "message",
        }

    async def router_node(state: ConversationState) -> dict[str, Any]:
        """Route the conversation to the appropriate sub-agent.

        Args:
            state: Current conversation state.

        Returns:
            State patch containing the chosen route, or the ``error`` route
            and a ``turn_error`` if the router failed.

        """
        messages = state["messages"]

        try:
            system_msg = SystemMessage(content=resources.get_system_prompt())
            decision = await asyncio.wait_for(
                resources.router.ainvoke([system_msg, *messages]),
                timeout=cfg.orchestration.router_timeout_s,
            )
        except TimeoutError:
            logger.warning(
                "Router timed out after %s s",
                cfg.orchestration.router_timeout_s,
            )
            return {"route": "error", "turn_error": "timeout"}
        except Exception as exc:  # noqa: BLE001
            return {"route": "error", "turn_error": _record_turn_failure("Router", exc)}

        step = cast("RouteStep", getattr(decision, "step", "error"))
        logger.info("Router decision: %s", step)
        return {"route": step}

    info_node = _make_dispatch_node(
        "info",
        resources.get_info_agent,
        cfg.orchestration.info_timeout_s,
    )
    booking_node = _make_dispatch_node(
        "booking",
        resources.get_booking_agent,
        cfg.orchestration.booking_timeout_s,
        failure_check=_booking_turn_failure,
    )

    async def moderate_reply_node(state: ConversationState) -> dict[str, Any]:
        """Check a sub-agent's reply before the guest sees it.

        Args:
            state: Current conversation state, after a sub-agent ran.

        Returns:
            An empty patch when the turn already failed, produced no reply
            text, or passed the check. A flagged reply is replaced in
            history with the configured refusal, and ``flagged`` is set to
            ``"reply"`` so the manager invalidates any proposal the turn
            made. A failed check sets ``turn_error``.

        """
        if state.get("turn_error") is not None:
            return {}
        replies = _turn_replies(state["messages"])
        text = "\n".join(reply.text for reply in replies)
        if not text.strip():
            # `finalize` records a turn with no reply as failed.
            return {}
        try:
            flagged = await resources.moderator.flagged(text, source="reply")
        except Exception as exc:  # noqa: BLE001
            return {"turn_error": _record_turn_failure("Reply moderation", exc)}
        if not flagged:
            return {}
        return {
            "messages": [
                # `add_messages` gave every message in the state an id.
                *(RemoveMessage(id=cast("str", reply.id)) for reply in replies),
                AIMessage(content=[{"type": "text", "text": cfg.messages.refusal}]),
            ],
            "flagged": "reply",
        }

    def refuse_node(state: ConversationState) -> dict[str, Any]:  # noqa: ARG001
        """Return an out-of-scope refusal response.

        Args:
            state: Unused conversation state.

        Returns:
            State patch with the refusal message.

        """
        logger.info("Refusing request")
        return {
            "messages": [
                AIMessage(content=[{"type": "text", "text": cfg.messages.refusal}]),
            ],
        }

    def error_node(state: ConversationState) -> dict[str, Any]:
        """Mark the turn failed without writing a reply.

        Reached when the message moderation check or the router failed, or
        the router chose the ``error`` step itself.
        No apology is written into history: the client is told about the
        failure through an ``error`` event instead (see `finalize_node`).

        Args:
            state: Current conversation state, read for a ``turn_error`` the
                router already recorded.

        Returns:
            State patch carrying ``turn_error``: ``"internal"`` unless the
            router recorded something more specific.

        """
        logger.info("Turn failed at routing")
        return {"turn_error": state.get("turn_error") or "internal"}

    def finalize_node(state: ConversationState) -> dict[str, Any]:
        """Prune intermediate tool chatter and drop a failed turn from history.

        This graph includes tool-using sub-agents. Their intermediate AI/tool
        messages are useful for execution but should not be retained or returned
        to the API client. What is kept is every HumanMessage together with the
        AIMessages that follow it without tool calls, preserving legitimate
        multi-message replies.

        A turn has failed when a node recorded ``turn_error``, or when it
        produced no final AIMessage at all, which is recorded as
        ``"internal"``. A failed turn is removed from history entirely rather
        than answered with an apology: an apology written into history reads,
        to the router and sub-agents on the next turn, as something the
        concierge said, and a guest resending the same text would otherwise
        leave the unanswered question in history twice.

        Only the most recent ``max_history_turns`` completed turns are kept.
        History is resent to the router and a sub-agent on every turn, so
        without a cap each message in a long conversation would cost more
        than the one before it.

        Args:
            state: Current conversation state.

        Returns:
            State patch that clears the messages channel and replaces it with
            the pruned history, plus ``turn_error`` when this turn failed and
            ``flagged`` when moderation flagged it. The manager reads both
            from this node's own output: ``turn_error`` to choose between an
            ``error`` and a ``done`` event, and ``flagged`` for its turn log
            and to invalidate the proposal of a flagged reply.

        """
        turns = _group_turns(
            filter_messages(state["messages"], exclude_tool_calls=True),
        )
        turn_error: TurnErrorCode | None = state.get("turn_error")
        if turns and (turn_error is not None or not turns[-1][1]):
            turn_error = turn_error or "internal"
            turns.pop()

        completed = [(human, replies) for human, replies in turns if replies]
        kept: list[BaseMessage] = []
        for human, replies in completed[-cfg.orchestration.max_history_turns :]:
            kept.extend([human, *replies])

        patch: dict[str, Any] = {
            "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *kept],
        }
        if turn_error is not None:
            patch["turn_error"] = turn_error
        flagged = state.get("flagged")
        if flagged is not None:
            patch["flagged"] = flagged
        return patch

    graph = StateGraph(ConversationState)
    graph.add_node("moderate_message", moderate_message_node)
    graph.add_node("router", router_node)
    graph.add_node("info", info_node)
    graph.add_node("booking", booking_node)
    graph.add_node("moderate_reply", moderate_reply_node)
    graph.add_node("refuse", refuse_node)
    graph.add_node("error", error_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "moderate_message")
    graph.add_conditional_edges(
        "moderate_message",
        _after_message_moderation,
        {"router": "router", "refuse": "refuse", "error": "error"},
    )
    graph.add_conditional_edges(
        "router",
        _route_from_state,
        {
            "info": "info",
            "booking": "booking",
            "refuse": "refuse",
            "error": "error",
        },
    )

    graph.add_edge("info", "moderate_reply")
    graph.add_edge("booking", "moderate_reply")
    graph.add_edge("moderate_reply", "finalize")
    graph.add_edge("refuse", "finalize")
    graph.add_edge("error", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=resources.checkpointer)


def _booking_turn_failure(result: dict[str, Any]) -> TurnErrorCode | None:
    """Fail a booking turn whose `search_rooms` could not reach the database.

    The booking agent returns normally after such an outage, with a reply the
    model wrote itself. That reply would reach the guest as an ordinary
    answer, free to invite a retry in its own words, and with no Send again
    button. Discarding it routes the outage through the failed-turn path
    instead, where both the copy and the resend belong to the application.

    Args:
        result: The booking agent's completed state.

    Returns:
        ``"unavailable"`` if a `search_rooms` call this turn found the database
        unreachable, otherwise ``None``.

    """
    if not database_unavailable_this_turn(result.get("messages", [])):
        return None
    logger.warning("booking agent could not reach the database; reply discarded")
    return "unavailable"


def _after_message_moderation(state: ConversationState) -> _AfterModeration:
    """Select the node after the message moderation check.

    Args:
        state: Current conversation state.

    Returns:
        ``"error"`` if the check failed, ``"refuse"`` if it flagged the
        message, otherwise ``"router"``.

    """
    if state.get("turn_error") is not None:
        return "error"
    if state.get("flagged") is not None:
        return "refuse"
    return "router"


def _record_turn_failure(source: str, exc: BaseException) -> TurnErrorCode:
    """Log a node failure and return the ``turn_error`` code it records.

    A booking tool that could not reach its database raises
    `BookingUnavailableError`, which records ``"unavailable"``: the same code
    a `search_rooms` outage records, so every booking-database outage reaches the
    guest the same way. An unreachable model provider or dependency raises
    through several layers of client library, each chaining the last, so its
    traceback runs to well over a hundred lines and says nothing the root
    cause does not. Both are logged as one warning line naming the root
    cause. A model call refused because the OpenAI budget or credit is spent
    records ``"quota"``, also as one warning line, since a resend would fail
    the same way and the guest must not be invited to make one. Anything
    else is a genuine defect and keeps its full traceback.

    Args:
        source: What failed, such as ``"Router"`` or ``"info agent"``.
        exc: The exception the node caught.

    Returns:
        ``"unavailable"`` for a booking database outage, ``"quota"`` for a
        spent model budget, otherwise ``"internal"``.

    """
    chain = _cause_chain(exc)
    if any(isinstance(link, BookingUnavailableError) for link in chain):
        logger.warning(
            "%s failed: booking database unreachable: %r (root cause: %r)",
            source,
            exc,
            chain[-1],
        )
        return "unavailable"
    quota_error = _quota_error(chain)
    if quota_error is not None:
        logger.warning(
            "%s failed: model quota exhausted: code=%s",
            source,
            quota_error.code,
            extra={"quota_code": quota_error.code},
        )
        return "quota"
    if any(isinstance(link, _NETWORK_ERRORS) for link in chain):
        logger.warning(
            "%s failed: network unreachable: %r (root cause: %r)",
            source,
            exc,
            chain[-1],
        )
        return "internal"
    logger.error("%s failed", source, exc_info=exc)
    return "internal"


def _cause_chain(exc: BaseException) -> list[BaseException]:
    """Return an exception followed by each exception it was raised from.

    Args:
        exc: The outermost exception.

    Returns:
        The chain from ``exc`` to its root cause, following ``__cause__``
        and then ``__context__``, stopping at any cycle.

    """
    chain: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and current not in chain:
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def _quota_error(chain: Sequence[BaseException]) -> openai.APIError | None:
    """Find the OpenAI error, if any, that reports a spent budget or credit.

    Matches `openai.APIError` rather than `RateLimitError` alone because
    `langchain_openai` re-raises its own subclass, and LlamaIndex embedding
    calls raise the SDK's class directly. Both keep `type` and `code` from
    the body.

    Args:
        chain: An exception followed by each exception it was raised from.

    Returns:
        The first link whose `type` or `code` is ``"insufficient_quota"``,
        or ``None``.

    """
    for link in chain:
        if isinstance(link, openai.APIError) and _QUOTA_ERROR_MARKER in (
            link.type,
            link.code,
        ):
            return link
    return None


def _turn_replies(messages: Sequence[BaseMessage]) -> list[AIMessage]:
    """Return the current turn's replies, without tool-calling messages.

    Args:
        messages: Conversation history, ending with the current turn.

    Returns:
        The AIMessages after the last HumanMessage that carry no tool calls,
        in order. Empty if there is no HumanMessage.

    """
    turns = _group_turns(filter_messages(messages, exclude_tool_calls=True))
    return turns[-1][1] if turns else []


def _group_turns(
    messages: Sequence[BaseMessage],
) -> list[tuple[HumanMessage, list[AIMessage]]]:
    """Group a message history into turns, one per HumanMessage.

    Messages before the first HumanMessage belong to no turn and are dropped,
    as are message types other than human and AI.

    Args:
        messages: History with tool-call messages already filtered out.

    Returns:
        One ``(human_message, replies)`` pair per turn, in order, where
        ``replies`` holds the AIMessages that followed that HumanMessage.

    """
    turns: list[tuple[HumanMessage, list[AIMessage]]] = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            turns.append((msg, []))
        elif isinstance(msg, AIMessage) and turns:
            turns[-1][1].append(msg)
    return turns
