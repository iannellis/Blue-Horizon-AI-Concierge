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
from langgraph.types import Send

from blue_horizon.agents.booking import database_unavailable_this_turn
from blue_horizon.agents.booking.write_ops import BookingUnavailableError
from blue_horizon.agents.orchestration.models import (
    ConversationState,
    PartInput,
    PartReply,
    RoutePart,
    turn_error_message,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from langchain_core.messages import BaseMessage
    from langchain_core.runnables import RunnableConfig
    from langgraph.graph.state import CompiledStateGraph

    from blue_horizon.agents.orchestration.models import (
        InjectionCheck,
        PartStep,
        RouteDecision,
        TurnErrorCode,
    )
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

# When every part of a turn failed, the turn reports the first of these codes
# any part recorded. `quota` comes first because its copy must never invite a
# resend that would fail the same way.
_TURN_ERROR_PRIORITY: tuple[TurnErrorCode, ...] = (
    "quota",
    "unavailable",
    "timeout",
    "internal",
)

type _AfterModeration = (
    list[Literal["router", "injection_check"]] | Literal["refuse", "error"]
)


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

    def _make_part_node(
        step: PartStep,
        get_agent: Callable[[], CompiledStateGraph],
        timeout_s: float,
        failure_check: Callable[[dict[str, Any]], TurnErrorCode | None] | None = None,
    ) -> Callable[..., Awaitable[dict[str, Any]]]:
        """Create the node that answers one part of a turn with a sub-agent.

        Args:
            step: The sub-agent's step name, also used for logging.
            get_agent: Callable returning the compiled sub-agent.
            timeout_s: Wall-clock timeout in seconds.
            failure_check: Optional inspection of the sub-agent's completed
                result, returning a ``turn_error`` code when its reply must be
                discarded even though the sub-agent returned normally.

        Returns:
            Async node function suitable for LangGraph, called through a
            `Send` from the fan-out with a `PartInput`.

        """

        async def _node(part: PartInput, config: RunnableConfig) -> dict[str, Any]:
            """Answer one part with a sub-agent, under a timeout.

            A failure is recorded on the part rather than on the turn, since
            whether the turn failed depends on the turn's other parts. See
            ``join_node``.

            Args:
                part: The part's messages and position.
                config: LangGraph runnable config.

            Returns:
                State patch adding this part's `PartReply` to
                ``part_replies``, carrying the sub-agent's replies or, on
                failure, the ``turn_error`` it recorded. A sub-agent that
                returned no reply records ``"internal"``.

            """
            logger.info("Dispatching to %s agent", step)
            replies: list[AIMessage] = []
            turn_error: TurnErrorCode | None = None
            try:
                result = await asyncio.wait_for(
                    get_agent().ainvoke({"messages": part["messages"]}, config=config),
                    timeout=timeout_s,
                )
            except TimeoutError:
                logger.warning("%s agent timed out after %s s", step, timeout_s)
                turn_error = "timeout"
            except Exception as exc:  # noqa: BLE001
                turn_error = _record_turn_failure(f"{step} agent", exc)
            else:
                patch = cast("dict[str, Any]", result)
                if failure_check is not None:
                    turn_error = failure_check(patch)
                if turn_error is None:
                    replies = _new_replies(patch.get("messages", []))
                    if not replies:
                        logger.warning("%s agent returned no reply", step)
                        turn_error = "internal"
            reply = PartReply(
                index=part["index"],
                step=step,
                replies=replies,
                turn_error=turn_error,
            )
            return {"part_replies": [reply]}

        _node.__name__ = _node.__qualname__ = f"{step}_node"
        return _node

    async def moderate_message_node(state: ConversationState) -> dict[str, Any]:
        """Check the guest's message before any model reads it.

        This is the first node of every turn, so it also clears
        ``turn_error``, ``flagged``, ``injection_error``, and
        ``failed_parts``: an earlier turn's values, still held in the
        checkpoint, must not be reported again.

        Args:
            state: Current conversation state, ending with the guest's
                message.

        Returns:
            State patch with ``turn_error``, ``flagged``, ``injection_error``,
            and ``failed_parts`` reset. A flagged message also sets
            ``flagged`` to ``"message"``,
            routes the turn to ``refuse``, and replaces the message in
            history with a placeholder carrying the same id. A failed check
            sets ``turn_error`` and the ``error`` route instead.

        """
        guest_message = state["messages"][-1]
        reset: dict[str, Any] = {
            "turn_error": None,
            "flagged": None,
            "injection_error": None,
            "failed_parts": [],
        }
        try:
            flagged = await resources.moderator.flagged(
                guest_message.text,
                source="guest message",
            )
        except Exception as exc:  # noqa: BLE001
            return {
                **reset,
                "route": "error",
                "turn_error": _record_turn_failure("Message moderation", exc),
            }
        if not flagged:
            return reset
        return {
            **reset,
            "messages": [HumanMessage(content=_WITHHELD_MESSAGE, id=guest_message.id)],
            "route": "refuse",
            "flagged": "message",
        }

    async def router_node(state: ConversationState) -> dict[str, Any]:
        """Split the guest's message into parts, one per sub-agent.

        Args:
            state: Current conversation state.

        Returns:
            State patch with the route and the parts, and ``part_replies``
            cleared for the fan-out. The route is ``"refuse"`` when every
            part was a refusal, and ``"error"``, with a ``turn_error``, when
            the router failed or returned no parts.

        """
        messages = state["messages"]

        try:
            system_msg = SystemMessage(content=resources.get_system_prompt())
            decision = cast(
                "RouteDecision",
                await asyncio.wait_for(
                    resources.router.ainvoke([system_msg, *messages]),
                    timeout=cfg.orchestration.router_timeout_s,
                ),
            )
        except TimeoutError:
            logger.warning(
                "Router timed out after %s s",
                cfg.orchestration.router_timeout_s,
            )
            return {"route": "error", "turn_error": "timeout"}
        except Exception as exc:  # noqa: BLE001
            return {"route": "error", "turn_error": _record_turn_failure("Router", exc)}

        if not decision.parts:
            logger.warning("Router returned no parts")
            return {"route": "error", "turn_error": "internal"}
        parts = _usable_parts(decision.parts)
        route = "+".join(part.step for part in parts) or "refuse"
        logger.info("Router decision: %s", route)
        return {"route": route, "parts": parts, "part_replies": None}

    async def injection_check_node(state: ConversationState) -> dict[str, Any]:
        """Ask whether the guest's message tries to manipulate the assistant.

        Runs in the same step as the router. The router alone sometimes
        answers a request for its instructions instead of refusing it,
        because it is busy splitting the message into parts. This check asks
        only the one question. Its verdict is applied by ``injection_gate``.

        Args:
            state: Current conversation state, ending with the guest's
                message.

        Returns:
            An empty patch when the message is clean, ``flagged`` set to
            ``"injection"`` when it is not, or ``injection_error`` when the
            check failed or timed out. It never writes ``turn_error`` or
            ``route``, which the router writes in the same step.

        """
        timeout_s = cfg.orchestration.injection_check_timeout_s
        check_input = [
            SystemMessage(content=resources.get_injection_check_prompt()),
            HumanMessage(content=_injection_check_input(state["messages"])),
        ]
        try:
            result = cast(
                "InjectionCheck",
                await asyncio.wait_for(
                    resources.injection_checker.ainvoke(check_input),
                    timeout=timeout_s,
                ),
            )
        except TimeoutError:
            logger.warning("Injection check timed out after %s s", timeout_s)
            return {"injection_error": "timeout"}
        except Exception as exc:  # noqa: BLE001
            return {"injection_error": _record_turn_failure("Injection check", exc)}
        if not result.injection:
            return {}
        logger.warning("Injection check flagged the guest message")
        return {"flagged": "injection"}

    info_node = _make_part_node(
        "info",
        resources.get_info_agent,
        cfg.orchestration.info_timeout_s,
    )
    booking_node = _make_part_node(
        "booking",
        resources.get_booking_agent,
        cfg.orchestration.booking_timeout_s,
        failure_check=_booking_turn_failure,
    )

    def join_node(state: ConversationState) -> dict[str, Any]:
        """Combine the parts' replies into the turn's reply.

        A single part's replies are kept as the sub-agent wrote them. Several
        parts' replies are joined into one message, in the order the guest
        asked, with no model call: a model rewriting them could misdescribe
        a proposal the guest is about to confirm. A part that failed is
        answered in place with the guest-facing copy for its ``turn_error``.

        Args:
            state: Current conversation state, after every part ran.

        Returns:
            State patch with the turn's reply and ``failed_parts`` naming any
            part that failed. When every part failed, no reply and a
            ``turn_error`` instead, chosen by `_combined_turn_error`.

        """
        part_replies = sorted(state.get("part_replies", []), key=lambda r: r["index"])
        failed = [reply for reply in part_replies if reply["turn_error"] is not None]
        if len(failed) == len(part_replies):
            logger.info("Every part of the turn failed")
            return {"turn_error": _combined_turn_error(failed)}
        if len(part_replies) == 1:
            return {"messages": part_replies[0]["replies"]}

        sections: list[str] = []
        for reply in part_replies:
            if reply["turn_error"] is None:
                sections.extend(message.text for message in reply["replies"])
            else:
                logger.info("The %s part of the turn failed", reply["step"])
                sections.append(turn_error_message(cfg.messages, reply["turn_error"]))
        joined = AIMessage(content=[{"type": "text", "text": "\n\n".join(sections)}])
        return {
            "messages": [joined],
            "failed_parts": [reply["step"] for reply in failed],
        }

    async def moderate_reply_node(state: ConversationState) -> dict[str, Any]:
        """Check the turn's reply before the guest sees it.

        Args:
            state: Current conversation state, after the join.

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

        Reached when the message moderation check, the router, or the
        injection check failed, or the router returned no parts.
        No apology is written into history: the client is told about the
        failure through an ``error`` event instead (see `finalize_node`).

        Args:
            state: Current conversation state, read for a ``turn_error`` an
                earlier node already recorded.

        Returns:
            State patch carrying ``turn_error``: ``"internal"`` unless an
            earlier node recorded something more specific.

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
            the pruned history, and clears ``part_replies``, plus
            ``turn_error`` when this turn failed, ``flagged`` when moderation
            flagged it, and ``failed_parts`` when one of its parts failed.
            The manager reads all three from this node's own output:
            ``turn_error`` to choose between an ``error`` and a ``done``
            event, and ``flagged`` and ``failed_parts`` for its turn log and
            to invalidate a proposal the guest never read about.

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
            "part_replies": None,
        }
        if turn_error is not None:
            patch["turn_error"] = turn_error
        flagged = state.get("flagged")
        if flagged is not None:
            patch["flagged"] = flagged
        failed_parts = state.get("failed_parts")
        if failed_parts:
            patch["failed_parts"] = failed_parts
        return patch

    graph = StateGraph(ConversationState)
    graph.add_node("moderate_message", moderate_message_node)
    graph.add_node("router", router_node)
    graph.add_node("injection_check", injection_check_node)
    graph.add_node("injection_gate", _injection_gate_node)
    graph.add_node("info", info_node)
    graph.add_node("booking", booking_node)
    graph.add_node("join", join_node)
    graph.add_node("moderate_reply", moderate_reply_node)
    graph.add_node("refuse", refuse_node)
    graph.add_node("error", error_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "moderate_message")
    graph.add_conditional_edges(
        "moderate_message",
        _after_message_moderation,
        ["router", "injection_check", "refuse", "error"],
    )
    # Waits for both branches, so the fan-out sees the check's verdict.
    graph.add_edge(["router", "injection_check"], "injection_gate")
    graph.add_conditional_edges(
        "injection_gate",
        _fan_out,
        ["info", "booking", "refuse", "error"],
    )

    graph.add_edge("info", "join")
    graph.add_edge("booking", "join")
    graph.add_edge("join", "moderate_reply")
    graph.add_edge("moderate_reply", "finalize")
    graph.add_edge("refuse", "finalize")
    graph.add_edge("error", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=resources.checkpointer)


def _booking_turn_failure(result: dict[str, Any]) -> TurnErrorCode | None:
    """Fail a booking part whose `search_rooms` could not reach the database.

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
        message, otherwise both ``"router"`` and ``"injection_check"``, which
        run in parallel.

    """
    if state.get("turn_error") is not None:
        return "error"
    if state.get("flagged") is not None:
        return "refuse"
    return ["router", "injection_check"]


def _injection_check_input(messages: Sequence[BaseMessage]) -> str:
    """Lay out the conversation as one block of text for the injection check.

    The check reads the earlier turns as context, labelled by speaker, and
    judges only the latest message. This is the layout it was measured
    with.

    Args:
        messages: The thread's history, ending with the guest's message.

    Returns:
        The earlier turns, one ``user:`` or ``assistant:`` line each, then
        the latest message under its own heading.

    """
    *earlier, latest = messages
    history = "\n".join(
        f"{'user' if isinstance(msg, HumanMessage) else 'assistant'}: {msg.text}"
        for msg in earlier
    )
    context = f"Earlier turns:\n{history}\n\n" if history else ""
    return f"{context}Latest guest message:\n{latest.text}"


def _injection_gate_node(state: ConversationState) -> dict[str, Any]:
    """Apply the injection check's verdict once the router has also finished.

    A flag refuses the whole turn, since the check cannot say which part of
    the message was the injection. It takes precedence over a router
    failure, which would only fail the turn. A failed check fails the turn
    rather than letting it through unchecked, as a failed moderation check
    does.

    Args:
        state: Current conversation state, after the router and the check.

    Returns:
        The ``refuse`` route with ``turn_error`` cleared when the check
        flagged the message, the ``error`` route with a ``turn_error`` when
        the check failed (the router's own code if it also failed),
        otherwise an empty patch that leaves the router's route in place.

    """
    if state.get("flagged") == "injection":
        return {"route": "refuse", "parts": [], "turn_error": None}
    injection_error = state.get("injection_error")
    if injection_error is not None:
        return {
            "route": "error",
            "turn_error": state.get("turn_error") or injection_error,
        }
    return {}


def _fan_out(state: ConversationState) -> list[Send] | Literal["refuse", "error"]:
    """Send each part of the turn to its sub-agent, or end the turn early.

    Every part gets the earlier history. A single-part turn's agent also
    gets the guest's message itself, so a turn with one intent reaches its
    agent exactly as it would without the split. With several parts, each
    agent gets its own part's standalone query in place of the guest's
    message instead, so it does not answer, or refuse, the other part.

    Args:
        state: Current conversation state, after the router and the
            injection gate.

    Returns:
        One `Send` per part, to the node named by its step, or
        ``"refuse"`` or ``"error"`` when the router or the injection gate
        chose that route.
        ``"error"`` also when there is no part to send, since an empty list
        of sends would end the graph before ``finalize``.

    """
    match state.get("route"):
        case "refuse" | "error" as route:
            return route
        case _:
            pass
    messages = state["messages"]
    parts = state.get("parts", [])
    if not parts:
        return "error"
    if len(parts) == 1:
        return [Send(parts[0].step, PartInput(messages=list(messages), index=0))]
    earlier = list(messages[:-1])
    return [
        Send(
            part.step,
            PartInput(
                messages=[*earlier, HumanMessage(content=part.query)],
                index=index,
            ),
        )
        for index, part in enumerate(parts)
    ]


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


def _usable_parts(parts: Sequence[RoutePart]) -> list[RoutePart]:
    """Drop refused parts and merge parts that share a step.

    The prompt asks for at most one part per sub-agent and never a refusal
    alongside other parts. This enforces both, so a router that splits one
    room request in two cannot dispatch the booking agent twice.

    Args:
        parts: The router's parts, in the order the guest asked.

    Returns:
        At most one part per sub-agent, in order of each step's first
        appearance, its query the queries of that step's parts joined by a
        space. Empty when every part was a refusal.

    """
    queries: dict[PartStep, list[str]] = {}
    for part in parts:
        if part.step != "refuse":
            queries.setdefault(part.step, []).append(part.query)
    return [
        RoutePart(query=" ".join(step_queries), step=step)
        for step, step_queries in queries.items()
    ]


def _new_replies(messages: Sequence[BaseMessage]) -> list[AIMessage]:
    """Return a sub-agent's replies for this part, without tool-calling messages.

    Args:
        messages: The sub-agent's completed messages: normally its input
            history followed by what it added.

    Returns:
        The AIMessages after the last HumanMessage that carry no tool calls,
        in order, or every such AIMessage when there is no HumanMessage.

    """
    start = 0
    for index, message in enumerate(messages):
        if isinstance(message, HumanMessage):
            start = index + 1
    return [
        message
        for message in messages[start:]
        if isinstance(message, AIMessage) and not message.tool_calls
    ]


def _combined_turn_error(failed: Sequence[PartReply]) -> TurnErrorCode:
    """Choose the ``turn_error`` for a turn whose every part failed.

    Args:
        failed: The failed parts' replies.

    Returns:
        The earliest code in `_TURN_ERROR_PRIORITY` that any part recorded,
        or ``"internal"`` when there is no failed part to choose from.

    """
    codes = {reply["turn_error"] for reply in failed}
    for code in _TURN_ERROR_PRIORITY:
        if code in codes:
            return code
    return "internal"


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
