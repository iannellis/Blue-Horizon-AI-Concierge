"""Factory for building the information RAG agent DAG."""

from __future__ import annotations

import heapq
import logging
from typing import TYPE_CHECKING, Any, cast

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from blue_horizon.agents.information.models import (
    InfoState,
    ParsedQuery,
    ParsedState,
    RetrievalItem,
    Source,
)
from blue_horizon.agents.information.retrieval import build_filters

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langgraph.graph.state import CompiledStateGraph

    from blue_horizon.agents.information.resources import InfoRagResources
    from blue_horizon.config import InfoRagConfig

logger = logging.getLogger(__name__)


def _latest_user_text(messages: list[BaseMessage]) -> str:
    """Return the most recent user message content as a string.

    Args:
        messages: Ordered conversation messages to scan from newest to oldest.

    Returns:
        Most recent user message content, or an empty string if none exists.

    """
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def _build_context_block(items: list[RetrievalItem]) -> str:
    """Build a plain-text context block from hydrated items.

    Args:
        items: Hydrated retrieval items to serialize into a context block.

    Returns:
        Context block string with one section per item.

    """
    lines = [
        "\n".join(
            [
                f"source={item.source}",
                f"score={item.score}",
                f"text={item.text}",
                f"metadata={item.metadata}",
            ],
        )
        for item in items
    ]
    return "\n\n".join(lines)


class InfoAgentFactory:
    """Factory for constructing the info retrieval DAG.

    This factory wires together:
        - configuration
        - shared async retrieval resources
        - prompt templates

    The resulting agent performs a single retrieval round in a fixed DAG.

    """

    __slots__ = ("_config", "_resources")

    _resources: InfoRagResources
    _config: InfoRagConfig

    def __init__(
        self,
        *,
        resources: InfoRagResources,
        config: InfoRagConfig,
    ) -> None:
        """Initialize the agent factory.

        Args:
            resources: Shared async resources (Redis client, indexes, retrievers).
            config: Parsed TOML configuration.

        """
        self._resources = resources
        self._config = config

    def build(self) -> CompiledStateGraph:  # noqa: C901, PLR0915
        """Build and return a compiled agent graph.

        Returns:
            CompiledStateGraph: A compiled LangGraph DAG for info retrieval.

        """
        resources = self._resources
        top_k = resources.top_k

        llm_cfg = self._config.llm
        parser_llm = ChatOpenAI(
            model=str(llm_cfg.model),
            temperature=0.0,
            timeout=float(llm_cfg.timeout_s),
            max_retries=int(llm_cfg.max_retries),
            reasoning={"effort": llm_cfg.reasoning_effort},
        ).with_structured_output(ParsedQuery, method="function_calling")
        responder_llm = ChatOpenAI(
            model=str(llm_cfg.model),
            temperature=float(llm_cfg.temperature),
            timeout=float(llm_cfg.timeout_s),
            max_retries=int(llm_cfg.max_retries),
            reasoning={"effort": llm_cfg.reasoning_effort},
        )

        def _log_node_failure(node_name: str, exc: Exception) -> None:
            """Log a node failure with traceback.

            Args:
                node_name: Name of the failing node.
                exc: Exception that was raised.

            """
            logger.exception("Info DAG node %s failed: %s", node_name, exc)

        def _make_catalog_query_node(
            source: Source,
            state_key: str,
        ) -> Callable[..., Awaitable[dict[str, Any]]]:
            """Create a query node for a filtered catalog source.

            Args:
                source: Which catalog source to query (AMENITIES or SERVICES).
                state_key: State dict key for the results (e.g. "amenities_results").

            Returns:
                Async node function suitable for LangGraph.

            """
            from blue_horizon.agents.exceptions import OperationalError  # noqa: PLC0415

            tool_name = f"query_{source.value}"

            async def _node(state: ParsedState) -> dict[str, Any]:
                """Query a filtered catalog source and return results.

                Args:
                    state: Parsed state containing the query and constraints.

                Returns:
                    State patch with results for the given source.

                """
                parsed = state["parsed"]
                filters = build_filters(
                    booking_required=parsed.booking_required,
                    min_price=parsed.min_price,
                    max_price=parsed.max_price,
                    min_notice_hours=parsed.min_notice_hours,
                    max_notice_hours=parsed.max_notice_hours,
                    min_duration_minutes=parsed.min_duration_minutes,
                    max_duration_minutes=parsed.max_duration_minutes,
                )
                try:
                    results = await resources.retrieve_filtered_catalog_items(
                        source=source,
                        query=parsed.query,
                        filters=filters,
                    )
                except OperationalError as exc:
                    logger.warning("%s operational failure: %s", tool_name, exc)
                    return {state_key: []}
                except Exception as exc:  # noqa: BLE001
                    _log_node_failure(tool_name, exc)
                    return {state_key: []}
                return {state_key: results}

            _node.__name__ = _node.__qualname__ = tool_name + "_node"
            _node.__doc__ = (
                f"Retrieve {source.value} for the parsed query and constraints."
            )
            return _node

        async def parse_node(state: InfoState) -> dict[str, Any]:
            """Parse the user request into a structured query and constraints.

            Args:
                state: Current info state with conversation messages.

            Returns:
                State patch with the parsed query.

            """
            try:
                parsed = cast(
                    "ParsedQuery",
                    await parser_llm.ainvoke(
                        [
                            SystemMessage(
                                content=(
                                    "Extract a single query string plus any "
                                    "constraints (booking, price, notice, duration) "
                                    "from the full conversation. Prefer the latest "
                                    "user request.\n\n"
                                    "Note: Service and amenity descriptions often "
                                    "include duration information. Do not extract "
                                    "these as user duration constraints unless the "
                                    "user explicitly requests a specific duration."
                                ),
                            ),
                            *state["messages"],
                        ],
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                _log_node_failure("parse", exc)
                parsed = ParsedQuery(query=_latest_user_text(state["messages"]))

            return {"parsed": parsed}

        async def query_faq_node(state: ParsedState) -> dict[str, Any]:
            """Retrieve FAQ entries for the parsed query.

            Args:
                state: Parsed state containing the query text.

            Returns:
                State patch with ``faq_results`` populated.

            """
            from blue_horizon.agents.exceptions import OperationalError  # noqa: PLC0415

            query = state["parsed"].query
            try:
                results = await resources.retrieve_faq(query)
            except OperationalError as exc:
                logger.warning("query_faq operational failure: %s", exc)
                return {"faq_results": []}
            except Exception as exc:  # noqa: BLE001
                _log_node_failure("query_faq", exc)
                return {"faq_results": []}
            return {"faq_results": results}

        query_amenities_node = _make_catalog_query_node(
            Source.AMENITIES,
            "amenities_results",
        )
        query_services_node = _make_catalog_query_node(
            Source.SERVICES,
            "services_results",
        )

        def rerank_node(state: InfoState) -> dict[str, Any]:
            """Select the top-k results across FAQ, amenities, and services.

            Args:
                state: Current state with retrieval results.

            Returns:
                State patch with ``top_results`` populated.

            """
            all_results = [
                *state.get("faq_results", []),
                *state.get("amenities_results", []),
                *state.get("services_results", []),
            ]
            ranked = heapq.nlargest(top_k, all_results, key=lambda x: x.score)
            return {"top_results": ranked}

        async def respond_node(state: InfoState) -> dict[str, Any]:
            """Generate the final response using retrieved context.

            Args:
                state: Current state with reranked results.

            Returns:
                State patch with the LLM response appended to messages.

            """
            top_items = state.get("top_results", [])
            context_block = _build_context_block(top_items)
            system_prompt = resources.get_system_prompt()
            try:
                response = await responder_llm.ainvoke(
                    [
                        SystemMessage(
                            content=(f"{system_prompt}{context_block}"),
                        ),
                        *state["messages"],
                    ],
                )
            except Exception as exc:
                _log_node_failure("respond", exc)
                raise
            return {"messages": [response]}

        graph = StateGraph(InfoState)
        graph.add_node("parse", parse_node)
        graph.add_node("query_faq", query_faq_node)
        graph.add_node("query_amenities", query_amenities_node)
        graph.add_node("query_services", query_services_node)
        graph.add_node("rerank", rerank_node)
        graph.add_node("respond", respond_node)

        graph.add_edge(START, "parse")
        graph.add_edge("parse", "query_faq")
        graph.add_edge("parse", "query_amenities")
        graph.add_edge("parse", "query_services")
        graph.add_edge("query_faq", "rerank")
        graph.add_edge("query_amenities", "rerank")
        graph.add_edge("query_services", "rerank")
        graph.add_edge("rerank", "respond")
        graph.add_edge("respond", END)

        return graph.compile()
