"""Operational wrapper for the orchestration agent.

Handles background initialization with retry/backoff, readiness tracking,
and memory-aware request invocation.
"""

from __future__ import annotations

import asyncio
import contextlib
import enum
import logging
from typing import TYPE_CHECKING, Any, cast

from langchain_core.messages import AIMessage, HumanMessage
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_never,
    wait_exponential,
)

from blue_horizon.agents._lifecycle import require
from blue_horizon.agents.exceptions import (
    ConfigurationError,
    OperationalError,
    ThreadCustomerMismatchError,
)
from blue_horizon.agents.orchestration.factory import build_orchestration_agent
from blue_horizon.agents.orchestration.formatting import format_chat_response
from blue_horizon.agents.orchestration.models import turn_error_message
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
    "merge": ("merge", "Ranking results\u2026"),
    "respond": ("respond", "Generating response\u2026"),
}

logger = logging.getLogger(__name__)


class Readiness(enum.Enum):
    """Orchestration agent readiness, as seen by `/v1/health` and `/v1/chat`.

    Attributes:
        READY: The compiled agent is built; requests are served normally.
        STARTING: No agent yet, and the last attempt (if any) failed for a
            reason classified transient (`OperationalError`, or anything
            unclassified -- see `OrchestrationManager._before_sleep`). The
            init loop keeps retrying; the guest sees a "waking up" message
            with no instruction, since the system is expected to recover on
            its own.
        FAILED: No agent yet, and the last attempt failed for a reason
            classified permanent (`ConfigurationError`). The init loop still
            keeps retrying -- an externally-fixed dependency (e.g. a
            corrected database role) should still recover without an
            operator restart -- but the guest is told this will not resolve
            on its own, with no retry affordance.

    """

    READY = "ready"
    STARTING = "starting"
    FAILED = "failed"


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
        "_init_task",
        "_llm_semaphore",
        "_lock",
        "_readiness",
        "_resources",
        "_stop_event",
        "_thread_customers",
    )

    _resources: OrchestrationResources
    _agent: CompiledStateGraph | None
    _readiness: Readiness
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

        llm_concurrency = self._resources.config.orchestration.llm_concurrency
        self._llm_semaphore = asyncio.Semaphore(llm_concurrency)
        self._agent = None
        self._readiness = Readiness.STARTING
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
        return self._readiness is Readiness.READY

    @property
    def readiness(self) -> Readiness:
        """Return the current readiness state.

        Callers that only need a yes/no answer should use `is_ready`
        instead; this is for callers that need to distinguish `STARTING`
        from `FAILED`, such as the `/v1/chat` readiness gate deciding what
        to tell the guest.

        Returns:
            Readiness: `READY`, `STARTING`, or `FAILED`.

        """
        return self._readiness

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
        except Exception:
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
            logger.warning(
                "Refused thread_id=%s to customer_id=%s: bound to customer_id=%s",
                thread_id,
                customer_id,
                bound,
            )
            msg =f"thread_id {thread_id!r} is already bound to a different guest."
            raise ThreadCustomerMismatchError(msg)

    def get_booking_resources(self) -> BookingSqlResources:
        """Expose booking resources for the confirm/dismiss/bookings endpoints.

        Returns:
            BookingSqlResources: Shared proposal store and write pool.

        """
        return self._resources.booking_resources

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
            - Sends only the new user message. The MemorySaver checkpointer
              loads prior history for the given thread_id and persists the
              updated history after the run.
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
            Final state from the orchestration agent. ``turn_error`` is set
            when the turn failed, in which case the failed turn is absent
            from ``messages`` and any proposal it created has been
            invalidated. Callers must check it rather than reading the last
            AI message, which would then belong to an earlier turn.

        Raises:
            ThreadCustomerMismatchError: If `thread_id` is already bound to a
                different `customer_id`.
            RuntimeError: If the orchestration agent is not ready. Callers
                are expected to check `is_ready` (or `readiness`) first --
                `api.app.chat`'s readiness gate is what actually stands
                between a not-ready manager and this method in production,
                so reaching this from there would be a bug in that gate, not
                an expected outcome to render as a chat reply.

        """
        self.bind_thread_customer(thread_id=thread_id, customer_id=customer_id)
        agent = require(self._agent, "Orchestration agent")

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
            result = cast(
                "dict[str, Any]",
                await agent.ainvoke(
                    state,
                    config=config,
                ),
            )
        if result.get("turn_error") is not None:
            # The turn is reported as failed, so nothing it proposed may
            # remain confirmable.
            self.get_booking_resources().proposals.invalidate_thread(thread_id)
        return result

    async def ainvoke_stream(
        self,
        *,
        thread_id: str,
        user_text: str,
        customer_id: int,
    ) -> AsyncGenerator[dict[str, Any]]:
        """Stream stage events, an optional proposal, then the final response.

        Yields stage-progress events as the graph executes each node, then
        either an optional ``proposal`` event followed by a single ``done``
        event, or, if the turn failed, a single ``error`` event and nothing
        else.

        Stage events have the form ``{"type": "stage", "label": str}``.
        The proposal event has the form ``{"type": "proposal", "proposal_id":
        str, "action": str, "summary": dict}``. The done event has the form
        ``{"type": "done", "response": str}``. The error event has the form
        ``{"type": "error", "code": str, "message": str}``, where ``code`` is
        a ``TurnErrorCode``.
        A failed turn yields no proposal, and any proposal it created is
        invalidated, since the guest is told the request did not complete.

        Multiple graph nodes that share the same conceptual stage (e.g.
        ``query_faq``, ``query_amenities``, ``query_services``) are
        deduplicated so that only one event is emitted per logical stage.

        Args:
            thread_id: Conversation identifier. Same id => shared history.
            user_text: Latest user message.
            customer_id: Server-resolved guest identity, injected into tool
                calls and never exposed to the model directly.

        Yields:
            Stage events, then proposal and done event dicts, or one error
            event.

        Raises:
            ThreadCustomerMismatchError: If `thread_id` is already bound to a
                different `customer_id`.
            RuntimeError: If the orchestration agent is not ready. Callers
                are expected to check `is_ready` (or `readiness`) first --
                `api.app.chat`'s readiness gate is what actually stands
                between a not-ready manager and this method in production.

        """
        self.bind_thread_customer(thread_id=thread_id, customer_id=customer_id)
        agent = require(self._agent, "Orchestration agent")

        booking_resources = self.get_booking_resources()
        booking_resources.proposals.invalidate_thread(thread_id)

        state: ConversationState = {"messages": [HumanMessage(content=user_text)]}
        config: RunnableConfig = {
            "configurable": {"thread_id": thread_id, "customer_id": customer_id},
        }

        emitted_stages: set[str] = set()
        finalize_output: dict[str, Any] | None = None
        async with self._llm_semaphore:
            async for event in agent.astream_events(
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
                    finalize_output = cast(
                        "dict[str, Any]", event["data"].get("output"),
                    )

        turn_error = (finalize_output or {}).get("turn_error")
        if turn_error is not None:
            booking_resources.proposals.invalidate_thread(thread_id)
            yield {
                "type": "error",
                "code": turn_error,
                "message": turn_error_message(
                    self._resources.config.messages, turn_error,
                ),
            }
            return

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

    def get_readiness_message(self) -> str:
        """Return guest-facing copy for the current non-ready readiness state.

        Only meaningful while `is_ready` is False; callers own deciding
        whether to show it at all (e.g. the `/v1/chat` readiness gate).

        Returns:
            str: The configured "still starting" message while `readiness`
            is `STARTING`, or the configured "will not resolve on its own"
            message while it is `FAILED`.

        """
        if self._readiness is Readiness.FAILED:
            return self._resources.config.messages.failed
        return self._resources.config.messages.unavailable

    async def _init_loop(self) -> None:
        """Background loop that initializes and retries on failure.

        The loop:
            - Attempts initialization if the agent is not ready.
            - On success, waits for a stop signal.
            - On failure, logs, resets, and retries with exponential backoff.
            - Backoff sleeps are interruptible: a stop signal exits immediately.

        """
        cfg = self._resources.config.orchestration
        # Identity of the previous failure, so a dependency that stays down
        # logs its traceback once rather than on every retry.
        last_failure: (
            tuple[type[BaseException], str, type[BaseException] | None] | None
        ) = None

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
            """Reset state, classify the failure, and log before the next retry.

            Classification drives `readiness`: `ConfigurationError` means
            retrying will not help, so the guest is told this will not
            resolve on its own (`FAILED`); everything else -- including an
            exception type this function does not recognise -- defaults to
            `STARTING`, so a misclassification degrades to bad copy rather
            than a guest being told a transient outage is permanent. The
            loop itself keeps retrying either way (`stop_never`): an
            externally-fixed dependency should still recover without an
            operator restart.

            The traceback is logged only when the failure differs from the
            previous one (by exception type, message, and cause type); a
            repeat logs one line with the attempt number.

            Args:
                retry_state: Tenacity retry call state carrying the last outcome.

            """
            nonlocal last_failure
            self._resources.reset_runtime_state()
            self._agent = None
            exc = retry_state.outcome.exception() if retry_state.outcome else None
            failure = None if exc is None else (
                type(exc),
                str(exc),
                None if exc.__cause__ is None else type(exc.__cause__),
            )
            traceback = None if failure == last_failure else exc
            last_failure = failure
            attempt = retry_state.attempt_number
            if isinstance(exc, ConfigurationError):
                self._readiness = Readiness.FAILED
                logger.error(
                    "Initialization failed (permanent, attempt %d): %r",
                    attempt,
                    exc,
                    exc_info=traceback,
                )
            elif isinstance(exc, OperationalError):
                self._readiness = Readiness.STARTING
                logger.warning(
                    "Initialization failed (operational, attempt %d): %r",
                    attempt,
                    exc,
                    exc_info=traceback,
                )
            else:
                self._readiness = Readiness.STARTING
                logger.error(
                    "Initialization failed (unclassified, attempt %d): %r",
                    attempt,
                    exc,
                    exc_info=traceback,
                )

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
                        self._agent = build_orchestration_agent(
                            resources=self._resources,
                        )
                        self._readiness = Readiness.READY
                        logger.info("Orchestration agent ready")

        await self._stop_event.wait()
