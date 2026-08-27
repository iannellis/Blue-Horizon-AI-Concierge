"""Factory for compiling the orchestration LangGraph."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, cast

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
    from collections.abc import Awaitable, Callable

    from langchain_core.messages import BaseMessage
    from langchain_core.runnables import RunnableConfig
    from langgraph.graph.state import CompiledStateGraph

    from blue_horizon.agents.orchestration.resources import OrchestrationResources

logger = logging.getLogger(__name__)


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
                State patch from the sub-agent, or an error message on failure.

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
                return {
                    "messages": [
                        AIMessage(
                            content=[{"type": "text", "text": cfg.messages.error}],
                        ),
                    ],
                }
            except Exception:
                logger.exception("%s agent failed", agent_name)
                return {
                    "messages": [
                        AIMessage(
                            content=[{"type": "text", "text": cfg.messages.error}],
                        ),
                    ],
                }
            return cast("dict[str, Any]", result)

        _node.__name__ = _node.__qualname__ = f"{agent_name}_node"
        return _node

    async def router_node(state: ConversationState) -> dict[str, Any]:
        """Route the conversation to the appropriate sub-agent.

        Args:
            state: Current conversation state.

        Returns:
            State patch containing the chosen route.

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
            return {"route": "error"}
        except Exception:
            logger.exception("Router failed")
            return {"route": "error"}

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

    def error_node(state: ConversationState) -> dict[str, Any]:  # noqa: ARG001
        """Return a user-friendly error response.

        Args:
            state: Unused conversation state.

        Returns:
            State patch with the error message.

        """
        logger.info("Returning error message")
        return {
            "messages": [
                AIMessage(content=[{"type": "text", "text": cfg.messages.error}]),
            ],
        }

    def finalize_node(state: ConversationState) -> dict[str, Any]:
        """Prune intermediate tool chatter and keep user+final assistant messages.

        This graph includes tool-using sub-agents. Their intermediate AI/tool
        messages are useful for execution but should not be retained or returned
        to the API client.

        This implementation keeps:
            - Every HumanMessage
            - All AIMessage objects following that HumanMessage that do NOT
              contain tool calls (to preserve legitimate multi-message replies)

        If a user message has no corresponding final AIMessage, an error message
        is appended for that turn.

        Args:
            state: Current conversation state.

        Returns:
            State patch that clears the messages channel and replaces it with
            the pruned history.

        """
        messages = state["messages"]
        filtered_messages = filter_messages(messages, exclude_tool_calls=True)

        kept: list[BaseMessage] = []
        current_user: HumanMessage | None = None
        current_assistants: list[AIMessage] = []

        def _flush_turn() -> None:
            """Append current turn to kept, ensuring a final assistant message.

            Ensures that a stored human message without a system response gets
            an error message inserted to prevent system confusion.

            """
            nonlocal current_user, current_assistants
            if current_user is None:
                return
            kept.append(current_user)
            if current_assistants:
                kept.extend(current_assistants)
            else:
                kept.append(
                    AIMessage(
                        content=[{"type": "text", "text": cfg.messages.error}],
                    ),
                )
            current_user = None
            current_assistants = []

        for msg in filtered_messages:
            if isinstance(msg, HumanMessage):
                _flush_turn()
                current_user = msg
                current_assistants = []
                continue

            if current_user is None:
                continue

            if isinstance(msg, AIMessage):
                current_assistants.append(msg)

        _flush_turn()

        return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *kept]}

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
