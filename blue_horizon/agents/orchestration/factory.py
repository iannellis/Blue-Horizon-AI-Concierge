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
from langchain_core.runnables import RunnableLambda
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


class OrchestrationAgentFactory:
    """Compile the orchestration LangGraph using initialized resources.

    Contract:
        - Assumes OrchestrationResources.startup_check() has completed successfully.
        - Does not perform I/O or network checks.
        - Returns a CompiledStateGraph that can be shared concurrently.

    """

    __slots__ = ("_resources",)

    _resources: OrchestrationResources

    def __init__(self, *, resources: OrchestrationResources) -> None:
        """Initialize the factory.

        Args:
            resources: Initialized orchestration resources.

        """
        self._resources = resources

    def build(self) -> CompiledStateGraph:  # noqa: C901, PLR0915
        """Compile and return the orchestration graph.

        Returns:
            Compiled orchestration state graph.

        """
        resources = self._resources
        cfg = resources.get_config()

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
                except asyncio.TimeoutError:  # noqa: UP041
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

            async def _invoke() -> RouteStep:
                """Invoke the router runnable with the system prompt prepended.

                Returns:
                    RouteStep returned by the router runnable.

                Raises:
                    Exception: Propagates any exception raised by the underlying
                        router runnable.

                """
                decision = await resources.get_router().ainvoke(
                    [
                        SystemMessage(content=resources.get_system_prompt()),
                        *messages,
                    ],
                )
                return cast("RouteStep", getattr(decision, "step", "error"))

            try:
                step = await asyncio.wait_for(
                    _invoke(),
                    timeout=cfg.orchestration.router_timeout_s,
                )
            except asyncio.TimeoutError:  # noqa: UP041
                logger.warning(
                    "Router timed out after %s s",
                    cfg.orchestration.router_timeout_s,
                )
                return {"route": "error"}
            except Exception:
                logger.exception("Router failed")
                return {"route": "error"}

            logger.info("Router decision: %s", step)
            return {"route": step}

        info_node = _make_dispatch_node(
            "info",
            resources.get_info_agent,
            cfg.orchestration.info_timeout_s,
        )
        rooms_node = _make_dispatch_node(
            "rooms",
            resources.get_rooms_agent,
            cfg.orchestration.rooms_timeout_s,
        )

        def refuse_node(_: ConversationState) -> dict[str, Any]:
            """Return an out-of-scope refusal response.

            Args:
                _: Unused conversation state.

            Returns:
                State patch with the refusal message.

            """
            logger.info("Refusing request")
            return {
                "messages": [
                    AIMessage(content=[{"type": "text", "text": cfg.messages.refusal}]),
                ],
            }

        def error_node(_: ConversationState) -> dict[str, Any]:
            """Return a user-friendly error response.

            Args:
                _: Unused conversation state.

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
        graph.add_node("router", RunnableLambda(router_node))
        graph.add_node("info", RunnableLambda(info_node))
        graph.add_node("rooms", RunnableLambda(rooms_node))
        graph.add_node("refuse", RunnableLambda(refuse_node))
        graph.add_node("error", RunnableLambda(error_node))
        graph.add_node("finalize", RunnableLambda(finalize_node))

        graph.add_edge(START, "router")
        graph.add_conditional_edges(
            "router",
            _route_from_state,
            {"info": "info", "rooms": "rooms", "refuse": "refuse", "error": "error"},
        )

        graph.add_edge("info", "finalize")
        graph.add_edge("rooms", "finalize")
        graph.add_edge("refuse", "finalize")
        graph.add_edge("error", "finalize")
        graph.add_edge("finalize", END)

        return graph.compile(checkpointer=resources.get_checkpointer())
