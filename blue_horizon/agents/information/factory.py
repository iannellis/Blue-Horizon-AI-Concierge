"""Factory for building the information RAG agent DAG."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, cast

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

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
        max_context_items = self._config.retrieval.max_context_items

        llm_cfg = self._config.llm
        parser_llm = ChatOpenAI(
            model=llm_cfg.model,
            temperature=0.0,
            timeout=llm_cfg.timeout_s,
            max_retries=llm_cfg.max_retries,
            reasoning={"effort": llm_cfg.reasoning_effort},
        ).with_structured_output(ParsedQuery, method="function_calling")
        responder_llm = ChatOpenAI(
            model=llm_cfg.model,
            temperature=llm_cfg.temperature,
            timeout=llm_cfg.timeout_s,
            max_retries=llm_cfg.max_retries,
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
                """Query a filtered catalog source for all parsed queries.

                Each query string is retrieved independently against the same
                filter set. Results are merged and deduplicated by text,
                keeping the highest-scoring item for each unique text.

                Args:
                    state: Parsed state containing the query strings and constraints.

                Returns:
                    State patch with deduplicated results for the given source.

                """
                parsed = state["parsed"]
                filters = build_filters(
                    booking_required=parsed.booking_required,
                    min_price=parsed.min_price,
                    max_price=parsed.max_price,
                    max_notice_hours=parsed.max_notice_hours,
                    min_duration_minutes=parsed.min_duration_minutes,
                    max_duration_minutes=parsed.max_duration_minutes,
                )
                try:
                    batches = cast(
                        "list[list[RetrievalItem]]",
                        await asyncio.gather(
                            *[
                                resources.retrieve_filtered_catalog_items(
                                    source=source,
                                    query=q,
                                    filters=filters,
                                )
                                for q in parsed.queries
                            ],
                        ),
                    )
                except OperationalError as exc:
                    logger.warning("%s operational failure: %s", tool_name, exc)
                    return {state_key: []}
                except Exception as exc:  # noqa: BLE001
                    _log_node_failure(tool_name, exc)
                    return {state_key: []}
                best: dict[str, RetrievalItem] = {}
                for batch in batches:
                    for item in batch:
                        if item.text not in best or item.score > best[item.text].score:
                            best[item.text] = item
                return {state_key: list(best.values())}

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
                                    "Decompose the user's request into one or more "
                                    "short, dense search strings plus any constraints "
                                    "(booking, price, max notice, duration). "
                                    "Prefer the "
                                    "latest user request.\n\n"
                                    "Rules for search strings:\n"
                                    "- Each string must be a tight noun phrase: core "
                                    "concepts only, no context ('so I can\u2026'), "
                                    "reasoning, or filler.\n"
                                    "- If the user asks about multiple distinct "
                                    "topics, produce one string per topic.\n"
                                    "- Single-topic requests produce exactly one "
                                    "string.\n\n"
                                    "Duration rules — always evaluate the WHOLE "
                                    "request, never each item separately:\n"
                                    "Step 1: count distinct services/activities "
                                    "in the request.\n"
                                    "Step 2:\n"
                                    "- Exactly ONE, with explicit duration: "
                                    "min = max = that value — EXCEPT when the "
                                    "single item is a quick snack/bite (words like "
                                    "'bite', 'snack'), in which case the duration "
                                    "is a time cap: max = that value, min = null.\n"
                                    "  '10-min bite' \u2192 "
                                    "min=null, max=10\n"
                                    "  '60-min massage' \u2192 min=60, max=60\n"
                                    "  '30-min breakfast' \u2192 min=30, max=30\n"
                                    "- TWO OR MORE, ALL with explicit durations: "
                                    "min = smallest, max = largest "
                                    "(all item types count).\n"
                                    "  '30-min massage + 60-min yoga' \u2192 "
                                    "min=30, max=60\n"
                                    "  '180-min movie + 30-min in-room breakfast' "
                                    "\u2192 min=30, max=180\n"
                                    "- TWO OR MORE, ANY without explicit duration: "
                                    "min=null, max=null (no estimation).\n"
                                    "  'breakfast + 60-min workout' \u2192 "
                                    "min=null, max=null\n"
                                    "  '30-min massage + quick bite' \u2192 "
                                    "min=null, max=null\n"
                                    "Key: a numeric prefix (e.g. '30-min', "
                                    "'60-minute') on ANY item — snack, meal, "
                                    "activity — counts as an explicit duration. "
                                    "Only bare item names without a numeric "
                                    "prefix lack an explicit duration.\n\n"
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
                parsed = ParsedQuery(queries=[_latest_user_text(state["messages"])])

            return {"parsed": parsed}

        async def query_faq_node(state: ParsedState) -> dict[str, Any]:
            """Retrieve FAQ entries for all parsed queries, deduplicated by text.

            Each query string is retrieved independently and results are merged,
            keeping the highest-scoring item when the same text is returned by
            multiple queries.

            Args:
                state: Parsed state containing the query strings.

            Returns:
                State patch with ``faq_results`` populated.

            """
            queries = state["parsed"].queries
            try:
                batches = cast(
                    "list[list[RetrievalItem]]",
                    await asyncio.gather(*[resources.retrieve_faq(q) for q in queries]),
                )
            except OperationalError as exc:
                logger.warning("query_faq operational failure: %s", exc)
                return {"faq_results": []}
            except Exception as exc:  # noqa: BLE001
                _log_node_failure("query_faq", exc)
                return {"faq_results": []}
            best: dict[str, RetrievalItem] = {}
            # The same item may appear in results from two different queries
            for batch in batches:
                for item in batch:
                    if item.text not in best or item.score > best[item.text].score:
                        best[item.text] = item
            return {"faq_results": list(best.values())}

        query_amenities_node = _make_catalog_query_node(
            Source.AMENITIES,
            "amenities_results",
        )
        query_services_node = _make_catalog_query_node(
            Source.SERVICES,
            "services_results",
        )

        def rerank_node(state: InfoState) -> dict[str, Any]:
            """Sort all retrieved results by score descending.

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
