"""Operational wrapper for the orchestration agent.

Handles background initialization with retry/backoff, readiness tracking,
and memory-aware request invocation.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import TYPE_CHECKING, Any, cast

from langchain_core.messages import AIMessage, HumanMessage
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_never,
    wait_exponential,
)

from blue_horizon.agents.exceptions import OperationalError
from blue_horizon.agents.orchestration.factory import OrchestrationAgentFactory
from blue_horizon.agents.orchestration.formatting import format_chat_response
from blue_horizon.agents.orchestration.resources import OrchestrationResources

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from langchain_core.runnables import RunnableConfig
    from langgraph.graph.state import CompiledStateGraph
    from tenacity import RetryCallState

    from blue_horizon.agents.orchestration.models import ConversationState

# Maps LangGraph node names to (stage_key, human-readable label) pairs.
# ``stage_key`` is used to deduplicate events so that multiple nodes sharing
# the same conceptual stage (e.g. the three query_* nodes) only emit one event.
_NODE_TO_STAGE: dict[str, tuple[str, str]] = {
    "router": ("routing", "Routing your request\u2026"),
    "info": ("search", "Searching hotel information\u2026"),
    "rooms": ("rooms", "Processing your room request\u2026"),
    "parse": ("parse", "Understanding your request\u2026"),
    "query_faq": ("search", "Searching hotel information\u2026"),
    "query_amenities": ("search", "Searching hotel information\u2026"),
    "query_services": ("search", "Searching hotel information\u2026"),
    "rerank": ("rerank", "Ranking results\u2026"),
    "respond": ("respond", "Generating response\u2026"),
}

logger = logging.getLogger(__name__)


class OrchestrationManager:
    """Operational wrapper around resources + compiled graph.

    Responsibilities:
        - Run startup in a background task with retry/backoff.
        - Expose readiness and a user-friendly unavailable message.
        - Provide a memory-aware ainvoke() helper.

    Threading/concurrency:
        - The compiled graph and sub-agents are shared across requests.
        - The init loop is guarded by a lock to avoid concurrent initialization.

    """

    __slots__ = (
        "_agent",
        "_factory",
        "_init_task",
        "_llm_semaphore",
        "_lock",
        "_resources",
        "_stop_event",
    )

    _resources: OrchestrationResources
    _factory: OrchestrationAgentFactory
    _agent: CompiledStateGraph | None
    _init_task: asyncio.Task[None] | None
    _llm_semaphore: asyncio.Semaphore
    _lock: asyncio.Lock
    _stop_event: asyncio.Event

    def __init__(self, *, pgsql_db_url: str | None = None) -> None:
        """Initialize the orchestration manager.

        Args:
            pgsql_db_url: Optional database URL override forwarded to the rooms
                SQL agent.  When set, the rooms agent uses this URL instead of
                the ``PGSQL_DB_URL`` application setting.  Use this in test
                harnesses that operate against a separate evaluation database so
                that the agent's writes are visible to the reconciliation pool.

        """
        self._resources = OrchestrationResources(pgsql_db_url=pgsql_db_url)
        self._factory = OrchestrationAgentFactory(resources=self._resources)

        llm_concurrency = self._resources.get_config().orchestration.llm_concurrency
        self._llm_semaphore = asyncio.Semaphore(llm_concurrency)
        self._agent = None
        self._init_task = None
        self._lock = asyncio.Lock()
        self._stop_event = asyncio.Event()

    @property
    def is_ready(self) -> bool:
        """Return whether the orchestration agent is ready.

        Returns:
            True if the compiled orchestration graph has been built.

        """
        return self._agent is not None

    async def start(self) -> None:
        """Start background initialization with retries."""
        if self._init_task is not None:
            return
        self._stop_event.clear()
        self._init_task = asyncio.create_task(
            self._init_loop(),
            name="orchestration-init",
        )

    async def stop(self) -> None:
        """Stop background initialization and close resources."""
        self._stop_event.set()

        if self._init_task is not None:
            self._init_task.cancel()
            try:
                await self._init_task
            except asyncio.CancelledError:
                pass
            finally:
                self._init_task = None

        try:
            await self._resources.aclose()
        except Exception:  # noqa: BLE001
            logger.warning("Failed to close orchestration resources", exc_info=True)

    async def ainvoke(
        self,
        *,
        thread_id: str,
        user_text: str,
        callbacks: list[Any] | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Invoke the orchestration agent with MemorySaver-backed history.

        Behavior:
            - If the orchestration agent is not ready, returns an assistant message
              with the configured "unavailable" text.
            - Otherwise, sends only the new user message. The MemorySaver
              checkpointer loads prior history for the given thread_id and
              persists the updated history after the run.

        Args:
            thread_id: Conversation identifier. Same id => shared history.
            user_text: Latest user message.
            callbacks: LangChain/LangGraph callbacks (e.g., LangSmith tracing hooks)
                to attach to the run.
            tags: Optional LangSmith tags associated with the run.
            metadata: Optional additional metadata to persist with the run (also used
                by LangSmith tracing).

        Returns:
            Final state patch from the orchestration agent.

        """
        if self._agent is None:
            msg = self.get_unavailable_message()
            return {"messages": [AIMessage(content=[{"type": "text", "text": msg}])]}

        state: ConversationState = {"messages": [HumanMessage(content=user_text)]}

        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        if callbacks is not None:
            config["callbacks"] = callbacks
        if tags is not None:
            config["tags"] = tags
        if metadata is not None:
            config["metadata"] = metadata

        async with self._llm_semaphore:
            return cast(
                "dict[str, Any]",
                await self._agent.ainvoke(
                    state,
                    config=config,
                ),
            )

    async def ainvoke_stream(
        self,
        *,
        thread_id: str,
        user_text: str,
    ) -> AsyncGenerator[dict[str, Any]]:
        """Stream stage events followed by the final assistant response.

        Yields stage-progress events as the graph executes each node, then a
        single ``done`` event once the graph has finished.

        Stage events have the form ``{"type": "stage", "label": str}``.
        The done event has the form ``{"type": "done", "response": str}``.

        Multiple graph nodes that share the same conceptual stage (e.g.
        ``query_faq``, ``query_amenities``, ``query_services``) are
        deduplicated so that only one event is emitted per logical stage.

        Args:
            thread_id: Conversation identifier. Same id => shared history.
            user_text: Latest user message.

        Yields:
            Stage dicts ``{"type": "stage", "label": str}`` as each graph
            node starts, followed by a done dict
            ``{"type": "done", "response": str}`` when the graph completes.

        """
        if self._agent is None:
            yield {"type": "done", "response": self.get_unavailable_message()}
            return

        state: ConversationState = {"messages": [HumanMessage(content=user_text)]}
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        emitted_stages: set[str] = set()
        async with self._llm_semaphore:
            async for event in self._agent.astream_events(
                state,
                config=config,
                version="v2",
            ):
                if event.get("event") != "on_chain_start":
                    continue
                node_name: str = event.get("metadata", {}).get("langgraph_node", "")
                if not node_name or node_name not in _NODE_TO_STAGE:
                    continue
                stage_key, label = _NODE_TO_STAGE[node_name]
                if stage_key not in emitted_stages:
                    emitted_stages.add(stage_key)
                    yield {"type": "stage", "label": label}

        final_state = await self._agent.aget_state(config)
        response_dict = format_chat_response(
            cast("dict[str, Any]", final_state.values),
        )
        ai_messages = [m for m in response_dict["messages"] if m["type"] == "ai"]
        response_text = (
            ai_messages[-1]["content"] if ai_messages else "No response received."
        )
        yield {"type": "done", "response": response_text}

    def get_unavailable_message(self) -> str:
        """Return the configured "unavailable" message.

        Returns:
            User-facing message to return when the system is not initialized.

        """
        return self._resources.get_config().messages.unavailable

    async def _init_loop(self) -> None:
        """Background loop that initializes and retries on failure.

        The loop:
            - Attempts initialization if the agent is not ready.
            - On success, waits for a stop signal.
            - On failure, logs, resets, and retries with exponential backoff.
            - Backoff sleeps are interruptible: a stop signal exits immediately.

        """
        cfg = self._resources.get_config().orchestration

        async def _interruptible_sleep(wait: float) -> None:
            """Sleep for *wait* seconds or until the stop event fires.

            Args:
                wait: Maximum seconds to sleep.

            Raises:
                asyncio.CancelledError: If the stop event fires during sleep.

            """
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._stop_event.wait(), timeout=wait)
            if self._stop_event.is_set():
                raise asyncio.CancelledError

        def _before_sleep(retry_state: RetryCallState) -> None:
            """Reset state and log the failure before the next retry.

            Args:
                retry_state: Tenacity retry call state carrying the last outcome.

            """
            self._resources.reset_runtime_state()
            self._agent = None
            exc = retry_state.outcome.exception() if retry_state.outcome else None
            if isinstance(exc, OperationalError):
                logger.warning(
                    "Initialization failed (operational): %s",
                    repr(exc),
                    exc_info=exc,
                )
            else:
                logger.error("Initialization failed", exc_info=exc)

        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type(Exception),
            stop=stop_never,
            wait=wait_exponential(
                multiplier=cfg.init_retry_base_s,
                max=cfg.init_retry_max_s,
            ),
            before_sleep=_before_sleep,
            sleep=_interruptible_sleep,
        ):
            with attempt:
                if self._stop_event.is_set():
                    return
                async with self._lock:
                    if self._agent is None:
                        logger.info("Initializing orchestration resources...")
                        await self._resources.startup_check()
                        self._agent = self._factory.build()
                        logger.info("Orchestration agent ready")

        await self._stop_event.wait()
