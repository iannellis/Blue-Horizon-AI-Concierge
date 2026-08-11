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

from blue_horizon.agents.exceptions import OperationalError, ThreadCustomerMismatchError
from blue_horizon.agents.orchestration.factory import OrchestrationAgentFactory
from blue_horizon.agents.orchestration.formatting import format_chat_response
from blue_horizon.agents.orchestration.resources import OrchestrationResources

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from langchain_core.runnables import RunnableConfig
    from langgraph.graph.state import CompiledStateGraph
    from tenacity import RetryCallState

    from blue_horizon.agents.booking.resources import BookingSqlResources
    from blue_horizon.agents.orchestration.models import ConversationState

# Maps LangGraph node names to (stage_key, human-readable label) pairs.
# ``stage_key`` is used to deduplicate events so that multiple nodes sharing
# the same conceptual stage (e.g. the three query_* nodes) only emit one event.
_NODE_TO_STAGE: dict[str, tuple[str, str]] = {
    "router": ("routing", "Routing your request\u2026"),
    "info": ("search", "Searching hotel information\u2026"),
    "booking": ("booking", "Processing your room request\u2026"),
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
        "_thread_customers",
    )

    _resources: OrchestrationResources
    _factory: OrchestrationAgentFactory
    _agent: CompiledStateGraph | None
    _init_task: asyncio.Task[None] | None
    _llm_semaphore: asyncio.Semaphore
    _lock: asyncio.Lock
    _stop_event: asyncio.Event
    _thread_customers: dict[str, int]

    def __init__(
        self,
        *,
        pgsql_rw_db_url: str | None = None,
        pgsql_ro_db_url: str | None = None,
    ) -> None:
        """Initialize the orchestration manager.

        Args:
            pgsql_rw_db_url: Optional read-write database URL override forwarded
                to the booking SQL agent (`bh_agent_rw`).  When set, the
                booking agent uses this URL instead of the ``PGSQL_RW_DB_URL``
                application setting.  Use this in test harnesses that operate
                against a separate evaluation database so that the agent's
                writes are visible to the reconciliation pool.
            pgsql_ro_db_url: Optional read-only database URL override
                forwarded to the booking SQL agent (`bh_agent_ro`), used
                exclusively by `run_sql`.  When overriding `pgsql_rw_db_url` for
                a test harness, this must be overridden alongside it and
                point at the same database.

        """
        self._resources = OrchestrationResources(
            pgsql_rw_db_url=pgsql_rw_db_url,
            pgsql_ro_db_url=pgsql_ro_db_url,
        )
        self._factory = OrchestrationAgentFactory(resources=self._resources)

        llm_concurrency = self._resources.get_config().orchestration.llm_concurrency
        self._llm_semaphore = asyncio.Semaphore(llm_concurrency)
        self._agent = None
        self._init_task = None
        self._lock = asyncio.Lock()
        self._stop_event = asyncio.Event()
        self._thread_customers = {}

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

    def bind_thread_customer(self, *, thread_id: str, customer_id: int) -> None:
        """Bind a thread to whichever guest first uses it; reject mismatches.

        Args:
            thread_id: Conversation identifier.
            customer_id: Guest identity presented for this request.

        Raises:
            ThreadCustomerMismatchError: If `thread_id` is already bound to a
                different `customer_id`.

        """
        bound = self._thread_customers.get(thread_id)
        if bound is None:
            self._thread_customers[thread_id] = customer_id
        elif bound != customer_id:
            msg = f"thread_id {thread_id!r} is already bound to a different guest."
            raise ThreadCustomerMismatchError(msg)

    def get_booking_resources(self) -> BookingSqlResources:
        """Expose booking resources for the confirm/dismiss/bookings endpoints.

        Returns:
            BookingSqlResources: Shared proposal store and write pool.

        """
        return self._resources.get_booking_resources()

    async def append_assistant_message(self, *, thread_id: str, text: str) -> None:
        """Append an app-authored assistant message to a thread's history.

        Used by the confirm/dismiss endpoints so the agent's next turn knows
        what happened -- the model is never the one reporting it.

        Args:
            thread_id: Conversation thread to update.
            text: App-authored message text (e.g., a confirmation receipt).

        """
        if self._agent is None:
            return
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        await self._agent.aupdate_state(
            config,
            {"messages": [AIMessage(content=[{"type": "text", "text": text}])]},
        )

    def clear_conversation_state(self) -> None:
        """Clear all in-process conversation state after a database reset.

        Clears the thread/customer registry in addition to delegating to
        `OrchestrationResources.clear_conversation_state()` (proposals and
        checkpointer). See that method's docstring for why this is not the
        same thing as the failed-init retry loop's `reset_runtime_state()`.

        """
        self._thread_customers.clear()
        self._resources.clear_conversation_state()

    async def ainvoke(  # noqa: PLR0913
        self,
        *,
        thread_id: str,
        user_text: str,
        customer_id: int,
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
            - Any proposal still pending for this thread is invalidated before
              the turn runs: a dialog must never be confirmable once the
              conversation has moved past it.

        Args:
            thread_id: Conversation identifier. Same id => shared history.
            user_text: Latest user message.
            customer_id: Server-resolved guest identity, injected into tool
                calls via `config["configurable"]["customer_id"]` and never
                exposed to the model directly.
            callbacks: LangChain/LangGraph callbacks (e.g., LangSmith tracing hooks)
                to attach to the run.
            tags: Optional LangSmith tags associated with the run.
            metadata: Optional additional metadata to persist with the run (also used
                by LangSmith tracing).

        Returns:
            Final state patch from the orchestration agent.

        Raises:
            ThreadCustomerMismatchError: If `thread_id` is already bound to a
                different `customer_id`.

        """
        self.bind_thread_customer(thread_id=thread_id, customer_id=customer_id)

        if self._agent is None:
            msg = self.get_unavailable_message()
            return {"messages": [AIMessage(content=[{"type": "text", "text": msg}])]}

        self.get_booking_resources().proposals.invalidate_thread(thread_id)

        state: ConversationState = {"messages": [HumanMessage(content=user_text)]}

        config: RunnableConfig = {
            "configurable": {"thread_id": thread_id, "customer_id": customer_id},
        }
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
        customer_id: int,
    ) -> AsyncGenerator[dict[str, Any]]:
        """Stream stage events, an optional proposal, then the final response.

        Yields stage-progress events as the graph executes each node, an
        optional ``proposal`` event if the turn created one, then a single
        ``done`` event once the graph has finished.

        Stage events have the form ``{"type": "stage", "label": str}``.
        The proposal event has the form ``{"type": "proposal", "proposal_id":
        str, "action": str, "summary": dict}``. The done event has the form
        ``{"type": "done", "response": str}``.

        Multiple graph nodes that share the same conceptual stage (e.g.
        ``query_faq``, ``query_amenities``, ``query_services``) are
        deduplicated so that only one event is emitted per logical stage.

        Args:
            thread_id: Conversation identifier. Same id => shared history.
            user_text: Latest user message.
            customer_id: Server-resolved guest identity, injected into tool
                calls and never exposed to the model directly.

        Yields:
            Stage, proposal, and done event dicts, in that order.

        Raises:
            ThreadCustomerMismatchError: If `thread_id` is already bound to a
                different `customer_id`.

        """
        self.bind_thread_customer(thread_id=thread_id, customer_id=customer_id)

        if self._agent is None:
            yield {"type": "done", "response": self.get_unavailable_message()}
            return

        booking_resources = self.get_booking_resources()
        booking_resources.proposals.invalidate_thread(thread_id)

        state: ConversationState = {"messages": [HumanMessage(content=user_text)]}
        config: RunnableConfig = {
            "configurable": {"thread_id": thread_id, "customer_id": customer_id},
        }

        emitted_stages: set[str] = set()
        finalize_output: dict[str, Any] | None = None
        async with self._llm_semaphore:
            async for event in self._agent.astream_events(
                state,
                config=config,
                version="v2",
            ):
                node_name: str = event.get("metadata", {}).get("langgraph_node", "")
                event_type = event.get("event")

                if event_type == "on_chain_start" and node_name in _NODE_TO_STAGE:
                    stage_key, label = _NODE_TO_STAGE[node_name]
                    if stage_key not in emitted_stages:
                        emitted_stages.add(stage_key)
                        yield {"type": "stage", "label": label}

                elif event_type == "on_chain_end" and node_name == "finalize":
                    # Captured directly from this run's own event stream
                    # rather than a post-hoc aget_state() re-read, which
                    # would return whatever the checkpoint currently holds
                    # and could hand two concurrent requests on one
                    # thread_id each other's reply.
                    finalize_output = cast("dict[str, Any]", event["data"]["output"])

        response_dict = format_chat_response(finalize_output or {})

        proposal = booking_resources.proposals.get_pending_for_thread(thread_id)
        if proposal is not None:
            yield {
                "type": "proposal",
                "proposal_id": proposal.proposal_id,
                "action": proposal.action,
                "summary": proposal.summary,
            }

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
