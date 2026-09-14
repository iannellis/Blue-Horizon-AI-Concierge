"""Factory for compiling the orchestration LangGraph."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, cast

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
    ) -> Callable[..., Awaitable[dict[str, Any]]]:
        """Create a dispatch node for a sub-agent.

        Args:
            agent_name: Human-readable name for logging.
            get_agent: Callable returning the compiled sub-agent.
            timeout_s: Wall-clock timeout in seconds.

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
                State patch from the sub-agent, or a ``turn_error`` on failure.
                A failure writes no message: see `finalize_node` for why.

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
                _log_turn_failure(f"{agent_name} agent", exc)
                return {"turn_error": "internal"}
            return cast("dict[str, Any]", result)

        _node.__name__ = _node.__qualname__ = f"{agent_name}_node"
        return _node

    async def router_node(state: ConversationState) -> dict[str, Any]:
        """Route the conversation to the appropriate sub-agent.

        Args:
            state: Current conversation state.

        Returns:
            State patch containing the chosen route and this turn's reset
            ``turn_error``. The router is the first node of every turn, so
            clearing ``turn_error`` here keeps an earlier turn's failure,
            still held in the checkpoint, from being reported again.

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
            _log_turn_failure("Router", exc)
            return {"route": "error", "turn_error": "internal"}

        step = cast("RouteStep", getattr(decision, "step", "error"))
        logger.info("Router decision: %s", step)
        return {"route": step, "turn_error": None}

    info_node = _make_dispatch_node(
        "info",
        resources.get_info_agent,
        cfg.orchestration.info_timeout_s,
    )
    booking_node = _make_dispatch_node(
        "booking",
        resources.get_booking_agent,
        cfg.orchestration.booking_timeout_s,
    )

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

        Reached when the router failed, or chose the ``error`` step itself.
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

        Args:
            state: Current conversation state.

        Returns:
            State patch that clears the messages channel and replaces it with
            the pruned history, plus ``turn_error`` when this turn failed. The
            manager reads that value from this node's own output to choose
            between an ``error`` and a ``done`` event.

        """
        turns = _group_turns(
            filter_messages(state["messages"], exclude_tool_calls=True),
        )
        turn_error: TurnErrorCode | None = state.get("turn_error")
        if turns and (turn_error is not None or not turns[-1][1]):
            turn_error = turn_error or "internal"
            turns.pop()

        kept: list[BaseMessage] = []
        for human, replies in turns:
            if replies:
                kept.extend([human, *replies])

        patch: dict[str, Any] = {
            "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *kept],
        }
        if turn_error is not None:
            patch["turn_error"] = turn_error
        return patch

    graph = StateGraph(ConversationState)
    graph.add_node("router", router_node)
    graph.add_node("info", info_node)
    graph.add_node("booking", booking_node)
    graph.add_node("refuse", refuse_node)
    graph.add_node("error", error_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "router")
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

    graph.add_edge("info", "finalize")
    graph.add_edge("booking", "finalize")
    graph.add_edge("refuse", "finalize")
    graph.add_edge("error", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=resources.checkpointer)


def _log_turn_failure(source: str, exc: BaseException) -> None:
    """Log a node failure, without a traceback when the network is the cause.

    An unreachable model provider or dependency raises through several layers
    of client library, each chaining the last, so its traceback runs to well
    over a hundred lines and says nothing the root cause does not. Such a
    failure is logged as one warning line naming the root cause. Anything
    else is a genuine defect and keeps its full traceback.

    Args:
        source: What failed, such as ``"Router"`` or ``"info agent"``.
        exc: The exception the node caught.

    """
    chain = _cause_chain(exc)
    if any(isinstance(link, _NETWORK_ERRORS) for link in chain):
        logger.warning(
            "%s failed: network unreachable: %r (root cause: %r)",
            source,
            exc,
            chain[-1],
        )
        return
    logger.error("%s failed", source, exc_info=exc)


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
