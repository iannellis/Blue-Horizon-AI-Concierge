"""Factory for building the information RAG agent DAG."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, cast

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from blue_horizon.agents._llm import build_chat_model
from blue_horizon.agents.exceptions import OperationalError
from blue_horizon.agents.information.models import (
    InfoState,
    ParsedQuery,
    ParsedState,
    RetrievalItem,
    Source,
)
from blue_horizon.agents.information.retrieval import build_filters

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from langgraph.graph.state import CompiledStateGraph

    from blue_horizon.agents.information.resources import InfoRagResources
    from blue_horizon.config import InfoRagConfig

logger = logging.getLogger(__name__)


def _latest_user_text(messages: Sequence[BaseMessage]) -> str:
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
    """Build a plain-text context block from retrieved items.

    Args:
        items: Retrieval items to serialize into a context block.

    Returns:
        Context block string with one section per item.

    """
    return "\n\n".join(
        "\n".join([
            f"source={item.source}",
            f"score={item.score}",
            f"text={item.text}",
            f"metadata={item.metadata}",
        ])
        for item in items
    )


def _best_by_text(batches: list[list[RetrievalItem]]) -> list[RetrievalItem]:
    """Merge batched retrieval results, keeping the highest-scoring item per text.

    When the same text appears across multiple query batches, the item with the
    highest score is retained and duplicates are discarded.

    Args:
        batches: List of per-query retrieval result lists.

    Returns:
        Deduplicated list of retrieval items, one per unique text.

    """
    best: dict[str, RetrievalItem] = {}
    for batch in batches:
        for item in batch:
            if item.text not in best or item.score > best[item.text].score:
                best[item.text] = item
    return list(best.values())


def build_info_agent(  # noqa: C901, PLR0915
    *,
    resources: InfoRagResources,
    config: InfoRagConfig,
) -> CompiledStateGraph:
    """Build and return a compiled info retrieval agent graph.

    Wires together configuration, shared async retrieval resources, and
    prompt templates. The resulting agent performs a single retrieval round
    in a fixed DAG.

    Args:
        resources: Shared async resources (Redis client, indexes, retrievers).
        config: Parsed TOML configuration.

    Returns:
        CompiledStateGraph: A compiled LangGraph DAG for info retrieval.

    """
    max_context_items = config.retrieval.max_context_items

    llm_cfg = config.llm
    parser_llm = build_chat_model(llm_cfg).with_structured_output(
        ParsedQuery, method="function_calling",
    )
    responder_llm = build_chat_model(llm_cfg)

    def _log_node_failure(node_name: str, exc: Exception) -> None:
        """Log a node failure with traceback.

        Args:
            node_name: Name of the failing node.
            exc: Exception that was raised.

        """
        logger.exception("Info DAG node %s failed: %s", node_name, exc)

    def _make_retrieval_node(
        source: Source,
        state_key: str,
    ) -> Callable[..., Awaitable[dict[str, Any]]]:
        """Create a retrieval node for one information source.

        FAQ has no metadata filters; amenities and services build filters
        from the parsed query's constraints before retrieving.

        Args:
            source: Which source to query (FAQ, AMENITIES, or SERVICES).
            state_key: State dict key for the results (e.g. "faq_results").

        Returns:
            Async node function suitable for LangGraph.

        """
        tool_name = f"query_{source.value}"

        async def _node(state: ParsedState) -> dict[str, Any]:
            """Retrieve `source` for all parsed queries, deduplicated by text.

            Each query string is retrieved independently; results are merged,
            keeping the highest-scoring item when the same text is returned
            by multiple queries.

            Args:
                state: Parsed state containing the query strings and constraints.

            Returns:
                State patch with deduplicated results for the given source.

            """
            parsed = state["parsed"]
            if source is Source.FAQ:
                pending = [resources.retrieve_faq(q) for q in parsed.queries]
            else:
                filters = build_filters(
                    booking_required=parsed.booking_required,
                    min_price=parsed.min_price,
                    max_price=parsed.max_price,
                    max_notice_hours=parsed.max_notice_hours,
                    min_duration_minutes=parsed.min_duration_minutes,
                    max_duration_minutes=parsed.max_duration_minutes,
                )
                pending = [
                    resources.retrieve_filtered_catalog_items(
                        source=source,
                        query=q,
                        filters=filters,
                    )
                    for q in parsed.queries
                ]
            try:
                batches = cast(
                    "list[list[RetrievalItem]]",
                    await asyncio.gather(*pending),
                )
            except OperationalError as exc:
                logger.warning("%s operational failure: %s", tool_name, exc)
                return {state_key: []}
            except Exception as exc:  # noqa: BLE001
                _log_node_failure(tool_name, exc)
                return {state_key: []}
            return {state_key: _best_by_text(batches)}

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
                        SystemMessage(content=resources.get_parser_prompt()),
                        *state["messages"],
                    ],
                ),
            )
        except Exception as exc:  # noqa: BLE001
            _log_node_failure("parse", exc)
            parsed = ParsedQuery(queries=[_latest_user_text(state["messages"])])

        return {"parsed": parsed}

    query_faq_node = _make_retrieval_node(Source.FAQ, "faq_results")
    query_amenities_node = _make_retrieval_node(Source.AMENITIES, "amenities_results")
    query_services_node = _make_retrieval_node(Source.SERVICES, "services_results")

    def merge_results_node(state: InfoState) -> dict[str, Any]:
        """Merge results from all sources and sort by score descending.

        All deduplicated results from FAQ, amenities, and services are passed
        to the LLM in score order. The LLM is instructed via the system prompt
        to present at most top_k cards, selecting the most relevant ones.

        Args:
            state: Current state with retrieval results.

        Returns:
            State patch with ``top_results`` populated, sorted by score.

        """
        all_results = [
            *state.get("faq_results", []),
            *state.get("amenities_results", []),
            *state.get("services_results", []),
        ]
        ranked = sorted(all_results, key=lambda x: x.score, reverse=True)
        return {"top_results": ranked[:max_context_items]}

    async def respond_node(state: InfoState) -> dict[str, Any]:
        """Generate the final response using retrieved context.

        Args:
            state: Current state with merged results.

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
    graph.add_node("merge", merge_results_node)
    graph.add_node("respond", respond_node)

    graph.add_edge(START, "parse")
    graph.add_edge("parse", "query_faq")
    graph.add_edge("parse", "query_amenities")
    graph.add_edge("parse", "query_services")
    graph.add_edge("query_faq", "merge")
    graph.add_edge("query_amenities", "merge")
    graph.add_edge("query_services", "merge")
    graph.add_edge("merge", "respond")
    graph.add_edge("respond", END)

    return graph.compile()
