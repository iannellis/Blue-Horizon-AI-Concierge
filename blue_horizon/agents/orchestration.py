"""LangGraph orchestration for the Blue-Horizon FastAPI app.

This module defines a LangGraph orchestration layer that routes user requests to
one of two compiled sub-agents:

- Information RAG agent (hotel policies, amenities, services, general info)
- Rooms SQL agent (availability, booking/modify/cancel flows)

The orchestration graph is compiled once and shared across concurrent requests.
Initialization runs asynchronously with retry/backoff so the FastAPI process can
start even if dependencies (e.g., Redis/Postgres) are temporarily unavailable.

Memory
  Uses LangGraph MemorySaver checkpointer to persist message history in-process.
  Each conversation is keyed by a thread_id. Invoke via
  OrchestrationManager.ainvoke(thread_id=..., user_text=...) and the user's
  message and final assistant response are appended to the stored history.

Architecture
- OrchestrationResources: owns long-lived dependencies + startup/close.
- OrchestrationAgentFactory: compiles the LangGraph using initialized resources.
- OrchestrationManager: operational wrapper that retries initialization and
  exposes readiness + a memory-aware invocation API.

"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Literal, cast

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    filter_messages,
)
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from pydantic import BaseModel, Field

from blue_horizon.agents.exceptions import OperationalError
from blue_horizon.agents.information import (
    InfoAgentFactory,
    InfoRagResources,
)
from blue_horizon.agents.prompt_utils import load_packaged_text
from blue_horizon.agents.rooms import (
    RoomsAgentFactory,
    RoomsSqlResources,
)
from blue_horizon.config import (
    InfoRagConfig,
    OrchestrationConfig,
    RoomsSqlConfig,
    load_app_config,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)


# ============================
# Routing schema + State
# ============================


type RouteStep = Literal["info", "rooms", "refuse", "error"]


class RouteDecision(BaseModel):
    """Structured router output.

    Attributes:
        step: The next step to take in answering the user's query.

    Notes:
        - This model is used with ChatOpenAI.with_structured_output() to request a
          typed response from the router.
        - The router prompt should constrain outputs to the RouteStep literal.

    """

    step: RouteStep = Field(
        ...,
        description="The next step to take in answering the user's query.",
    )


class ConversationState(MessagesState, total=False):
    """LangGraph state with persisted message history.

    Attributes:
        messages: Message history (provided by MessagesState).
        route: Router decision.

    """

    route: RouteStep


def _route_from_state(state: ConversationState) -> RouteStep:
    """Select the next node to execute based on state.

    Args:
        state: Current LangGraph state.

    Returns:
        Route step key for conditional edges.

    """
    return cast("RouteStep", state.get("route") or "error")


# ============================
# Resources (mirrors rooms_sql_agent.py pattern)
# ============================


class OrchestrationResources:
    """Long-lived dependencies for orchestration.

    Responsibilities:
        - Load orchestration configuration.
        - Resolve and load the orchestration system prompt.
        - Construct and hold the router LLM runnable.
        - Construct and hold sub-resources (InfoRagResources, RoomsSqlResources).
        - Build and hold the compiled sub-agents for info + rooms.
        - Provide a process-lifetime MemorySaver checkpointer for message history.

    Lifecycle:
        - Instantiate (cheap): reads TOML + resolves paths, creates lightweight clients.
        - startup_check() (expensive): performs connectivity checks and builds agents.
        - aclose(): closes underlying resource pools/clients.

    Notes:
        - If startup_check() raises, callers may retry.
        - reset_runtime_state() clears partially-initialized runtime fields.

    """

    __slots__ = (
        "_checkpointer",
        "_config",
        "_info_agent",
        "_info_config",
        "_info_resources",
        "_rooms_agent",
        "_rooms_config",
        "_rooms_resources",
        "_router",
        "_system_prompt",
        "_system_prompt_resource",
    )

    _config: OrchestrationConfig
    _system_prompt_resource: str
    _system_prompt: str | None

    _info_config: InfoRagConfig
    _info_resources: InfoRagResources
    _info_agent: CompiledStateGraph | None

    _rooms_config: RoomsSqlConfig
    _rooms_resources: RoomsSqlResources
    _rooms_agent: CompiledStateGraph | None

    _router: Runnable[list[BaseMessage], RouteDecision]
    _checkpointer: MemorySaver

    def __init__(self) -> None:
        """Initialize orchestration resources.

        This constructor performs only lightweight work: it reads the orchestration
        TOML, resolves the orchestration prompt path, constructs resource objects,
        and builds the router runnable.

        Heavy work (connectivity checks, agent compilation) occurs in startup_check().

        Raises:
            RuntimeError: If configuration or prompt resolution fails.

        """
        app_config = load_app_config()
        self._config = app_config.orchestration

        prompts_folder = self._config.prompts.folder.strip("/")
        if prompts_folder:
            self._system_prompt_resource = (
                f"{prompts_folder}/{self._config.prompts.orchestration_prompt_filename}"
            )
        else:
            self._system_prompt_resource = (
                self._config.prompts.orchestration_prompt_filename
            )
        self._system_prompt = None

        self._info_config = app_config.info
        self._info_resources = InfoRagResources(
            redis_url=app_config.redis_url,
            config=self._info_config,
        )
        self._info_agent = None

        self._rooms_config = app_config.rooms
        self._rooms_resources = RoomsSqlResources(
            pgsql_db_url=app_config.pgsql_db_url,
            config=self._rooms_config,
        )
        self._rooms_agent = None

        llm_cfg = self._config.llm
        llm = ChatOpenAI(
            model=llm_cfg.model,
            temperature=llm_cfg.temperature,
            reasoning={"effort": llm_cfg.reasoning_effort},
            timeout=llm_cfg.timeout_s,
            max_retries=llm_cfg.max_retries,
        )
        self._router = cast(
            "Runnable[list[BaseMessage], RouteDecision]",
            llm.with_structured_output(RouteDecision),
        )

        self._checkpointer = MemorySaver()

    async def startup_check(self) -> None:
        """Validate dependencies and build sub-agents.

        This method should run once during FastAPI startup (or inside the
        OrchestrationManager retry loop). It loads the orchestration prompt, runs
        each sub-resource startup check, and builds the compiled info/rooms agents.

        Raises:
            OperationalError: If startup fails for a reason that should be treated
                as transient and retried (e.g., dependency outage).

        """
        try:
            self._system_prompt = load_packaged_text(self._system_prompt_resource)

            await self._info_resources.startup_check()
            self._info_agent = InfoAgentFactory(
                resources=self._info_resources,
                config=self._info_config,
            ).build()

            await self._rooms_resources.startup_check()
            self._rooms_agent = RoomsAgentFactory(
                config=self._rooms_config,
                resources=self._rooms_resources,
            ).build()

        except OperationalError:
            raise
        except Exception as exc:
            msg = "Orchestration resources failed during startup"
            raise OperationalError(msg) from exc

    async def aclose(self) -> None:
        """Close underlying resource pools/clients.

        This should be called during FastAPI shutdown.

        """
        await self._rooms_resources.aclose()
        await self._info_resources.aclose()

    def reset_runtime_state(self) -> None:
        """Clear runtime fields after a failed initialization attempt.

        This enables the manager to retry startup cleanly without re-instantiating
        the resources object.

        """
        self._system_prompt = None
        self._info_agent = None
        self._rooms_agent = None

    def get_config(self) -> OrchestrationConfig:
        """Return the loaded orchestration configuration.

        Returns:
            Loaded orchestration configuration.

        """
        return self._config

    def get_system_prompt(self) -> str:
        """Return the orchestration system prompt text.

        Returns:
            System prompt content.

        Raises:
            RuntimeError: If startup_check() has not been called.

        """
        if self._system_prompt is None:
            msg = "System prompt not loaded. Call startup_check() first."
            raise RuntimeError(msg)
        return self._system_prompt

    def get_router(self) -> Runnable[list[BaseMessage], RouteDecision]:
        """Return the router runnable.

        Returns:
            Runnable that maps a list of messages to a RouteDecision.

        """
        return self._router

    def get_checkpointer(self) -> MemorySaver:
        """Return the MemorySaver checkpointer.

        Returns:
            In-memory checkpointer for message history.

        """
        return self._checkpointer

    def get_info_agent(self) -> CompiledStateGraph:
        """Return the compiled info agent.

        Returns:
            Compiled info agent.

        Raises:
            RuntimeError: If startup_check() has not been called.

        """
        if self._info_agent is None:
            msg = "Info agent not initialized. Call startup_check() first."
            raise RuntimeError(msg)
        return self._info_agent

    def get_rooms_agent(self) -> CompiledStateGraph:
        """Return the compiled rooms agent.

        Returns:
            Compiled rooms agent.

        Raises:
            RuntimeError: If startup_check() has not been called.

        """
        if self._rooms_agent is None:
            msg = "Rooms agent not initialized. Call startup_check() first."
            raise RuntimeError(msg)
        return self._rooms_agent


# ============================
# Factory (compiles graph using initialized resources)
# ============================


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

    def build(self, *, prune_tool_messages: bool = True) -> CompiledStateGraph:  # noqa: C901, PLR0915
        """Compile and return the orchestration graph.

        Returns:
            Compiled orchestration state graph.

        Args:
            prune_tool_messages: Whether to remove tool messages from the returned
                message history via the finalize node.

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

            async def _invoke() -> RouteDecision:
                """Invoke the router runnable.

                This helper exists to keep the router call self-contained for
                asyncio.wait_for(). It applies the orchestration system prompt as a
                SystemMessage and then appends the current conversation messages.

                Returns:
                    RouteDecision returned by the router runnable.

                Raises:
                    Exception: Propagates any exception raised by the underlying
                        router runnable (network errors, timeouts at the client
                        layer, schema/parse errors, etc.).

                """
                return await resources.get_router().ainvoke(
                    [
                        SystemMessage(content=resources.get_system_prompt()),
                        *messages,
                    ],
                )

            try:
                decision = await asyncio.wait_for(
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

            step: RouteStep = getattr(decision, "step", "error")
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
            """Return an out-of-scope refusal response."""
            logger.info("Refusing request")
            return {
                "messages": [
                    AIMessage(content=[{"type": "text", "text": cfg.messages.refusal}]),
                ],
            }

        def error_node(_: ConversationState) -> dict[str, Any]:
            """Return a user-friendly error response."""
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

                Real purpose is to make sure that, in the case of a system crash and
                the last human message is stored without a system response, a response
                to that human message is inserted to prevent system confusion.
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
        if prune_tool_messages:
            graph.add_node("finalize", RunnableLambda(finalize_node))

        graph.add_edge(START, "router")
        graph.add_conditional_edges(
            "router",
            _route_from_state,
            {"info": "info", "rooms": "rooms", "refuse": "refuse", "error": "error"},
        )

        if prune_tool_messages:
            graph.add_edge("info", "finalize")
            graph.add_edge("rooms", "finalize")
            graph.add_edge("refuse", "finalize")
            graph.add_edge("error", "finalize")
            graph.add_edge("finalize", END)
        else:
            graph.add_edge("info", END)
            graph.add_edge("rooms", END)
            graph.add_edge("refuse", END)
            graph.add_edge("error", END)

        return graph.compile(checkpointer=resources.get_checkpointer())


# ============================
# Manager for FastAPI startup + retry + memory-aware invocation
# ============================


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
        "_prune_tool_messages",
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

    def __init__(self, *, prune_tool_messages: bool = True) -> None:
        """Initialize the orchestration manager.

        Args:
            prune_tool_messages: Whether to remove tool messages from message
                history via the finalize node.

        """
        self._resources = OrchestrationResources()
        self._factory = OrchestrationAgentFactory(resources=self._resources)

        llm_concurrency = self._resources.get_config().orchestration.llm_concurrency
        self._llm_semaphore = asyncio.Semaphore(llm_concurrency)
        self._agent = None
        self._init_task = None
        self._lock = asyncio.Lock()
        self._stop_event = asyncio.Event()
        self._prune_tool_messages = prune_tool_messages

    @property
    def is_ready(self) -> bool:
        """Return whether the orchestration agent is ready.

        Returns:
            True if the compiled orchestration graph has been built.

        """
        return self._agent is not None

    def get_unavailable_message(self) -> str:
        """Return the configured "unavailable" message.

        Returns:
            User-facing message to return when the system is not initialized.

        """
        return self._resources.get_config().messages.unavailable

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
            callbacks: LangChain/LangGraph callbacks (e.g., LangSmith tracing hooks) to
                attach to the run.
            tags: Optional LangSmith tags associated with the run.
            metadata: Optional additional metadata to persist with the run (also used by
                LangSmith tracing).

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
                        self._agent = self._factory.build(
                            prune_tool_messages=self._prune_tool_messages,
                        )
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


# -----------------------------
# FastAPI integration example
# -----------------------------


"""In your app module (e.g. main.py), you'd do something like:

from contextlib import asynccontextmanager
from fastapi import FastAPI

orchestration = OrchestrationManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await orchestration.start()
    yield
    await orchestration.stop()

app = FastAPI(lifespan=lifespan)

@app.post("/chat")
async def chat(payload: dict):
    # You must provide a stable conversation id.
    thread_id = payload["thread_id"]
    user_text = payload["text"]
    return await orchestration.ainvoke(thread_id=thread_id, user_text=user_text)
"""
