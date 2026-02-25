"""Operational wrapper for the orchestration agent.

Handles background initialization with retry/backoff, readiness tracking,
and memory-aware request invocation.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, cast

from langchain_core.messages import AIMessage, HumanMessage

from blue_horizon.agents.exceptions import OperationalError
from blue_horizon.agents.orchestration.factory import OrchestrationAgentFactory
from blue_horizon.agents.orchestration.resources import OrchestrationResources

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig
    from langgraph.graph.state import CompiledStateGraph

    from blue_horizon.agents.orchestration.models import ConversationState

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

        """
        cfg = self._resources.get_config().orchestration
        backoff = cfg.init_retry_base_s

        while not self._stop_event.is_set():
            try:
                async with self._lock:
                    if self._agent is None:
                        logger.info("Initializing orchestration resources...")
                        await self._resources.startup_check()
                        self._agent = self._factory.build()
                        logger.info("Orchestration agent ready")
                        backoff = cfg.init_retry_base_s

                await self._stop_event.wait()

            except asyncio.CancelledError:
                raise
            except OperationalError as exc:
                logger.warning(
                    "Initialization failed (operational): %s",
                    repr(exc),
                    exc_info=True,
                )
                backoff = await self._reset_and_backoff(
                    backoff=backoff,
                    max_backoff=cfg.init_retry_max_s,
                )
            except Exception:
                logger.exception("Initialization failed")
                backoff = await self._reset_and_backoff(
                    backoff=backoff,
                    max_backoff=cfg.init_retry_max_s,
                )

    async def _reset_and_backoff(self, *, backoff: float, max_backoff: float) -> float:
        """Reset runtime state and sleep for the current backoff.

        Args:
            backoff: Current backoff duration in seconds.
            max_backoff: Maximum backoff duration in seconds.

        Returns:
            Next backoff duration in seconds.

        """
        self._resources.reset_runtime_state()
        self._agent = None

        stopped = await self._sleep_or_stop(timeout_s=backoff)
        if stopped:
            return backoff

        return min(backoff * 2.0, max_backoff)

    async def _sleep_or_stop(self, *, timeout_s: float) -> bool:
        """Wait for either a stop signal or a timeout.

        Args:
            timeout_s: Maximum wait duration in seconds.

        Returns:
            True if a stop signal was received, otherwise False.

        """
        if self._stop_event.is_set():
            return True

        try:
            await asyncio.wait_for(self._stop_event.wait(), timeout=timeout_s)
        except asyncio.TimeoutError:  # noqa: UP041
            return False
        else:
            return True
