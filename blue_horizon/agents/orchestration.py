"""Orchestration graph for the Blue-Horizon FastAPI app.

This module provides a LangGraph orchestration layer that routes user requests to
one of two compiled sub-agents:

- An information RAG agent (policies, amenities, services, etc.).
- A rooms SQL agent (availability, booking/modify/cancel flows).

The orchestration graph is compiled once and shared across requests. Startup is
handled asynchronously with retry/backoff semantics so the FastAPI application
can start even if dependencies (e.g., Redis/Postgres) are temporarily unavailable.

Architecture (mirrors rooms_sql_agent.py house style)
- OrchestrationResources: owns long-lived dependencies + startup/close.
- OrchestrationAgentFactory: compiles the LangGraph using initialized resources.
- OrchestrationManager: operational wrapper that retries initialization and
  exposes readiness.

Configuration
- Loaded from orchestration_config.toml into typed dataclasses.
- Secrets (Redis/DB URLs, API keys) come from environment variables.

"""

import asyncio
import logging
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, cast

from langchain.messages import AIMessage, SystemMessage
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from blue_horizon.agents.information import (
    InfoAgentFactory,
    InfoRagResources,
    get_redis_url,
    load_info_config,
)
from blue_horizon.agents.rooms_sql import (
    RoomsAgentFactory,
    RoomsSqlResources,
    get_pgsql_db_url,
    load_rooms_sql_config,
)

logger = logging.getLogger(__name__)


class OperationalError(RuntimeError):
    """Operational (expected) error.

    Use this exception for failures that can occur during normal operation
    (e.g., transient connectivity issues, dependency outages).

    Callers should generally log the error and return a safe, user-friendly
    response or retry later rather than crashing the process.

    """


# ============================
# Settings (loaded from TOML config)
# ============================


@dataclass(frozen=True, slots=True)
class LlmConfig:
    """Chat model configuration.

    Attributes:
        model: Model identifier passed to ChatOpenAI.
        temperature: Sampling temperature.
        reasoning_effort: Reasoning effort hint passed as reasoning={"effort": ...}.
        timeout_s: Per-request network timeout for the LLM client.
        max_retries: Provider/client retry count for transient failures.

    """

    model: str
    temperature: float
    reasoning_effort: str
    timeout_s: float
    max_retries: int


@dataclass(frozen=True, slots=True)
class OrchestrationRuntimeConfig:
    """Orchestration behavior configuration.

    Attributes:
        init_retry_base_s: Initial backoff delay after a failed init attempt.
        init_retry_max_s: Maximum backoff delay between init attempts.
        router_timeout_s: Wall-clock cap for the router node execution.

    """

    init_retry_base_s: float
    init_retry_max_s: float
    router_timeout_s: float


@dataclass(frozen=True, slots=True)
class PromptsConfig:
    """Prompt file configuration.

    Attributes:
        folder: Folder (relative to this module) containing prompt templates.
        orchestration_prompt_filename: Router system prompt filename.

    """

    folder: str
    orchestration_prompt_filename: str


@dataclass(frozen=True, slots=True)
class MessagesConfig:
    """User-facing message configuration.

    Attributes:
        refusal: Message returned when the router explicitly chooses "refuse".
        error: Message returned when an internal error occurs.
        unavailable: Message returned while the system is initializing.

    """

    refusal: str
    error: str
    unavailable: str


@dataclass(frozen=True, slots=True)
class OrchestrationConfig:
    """Top-level orchestration configuration loaded from TOML.

    Attributes:
        llm: LLM client configuration.
        orchestration: Orchestration runtime settings.
        prompts: Prompt locations and filenames.
        messages: User-facing message strings.

    """

    llm: LlmConfig
    orchestration: OrchestrationRuntimeConfig
    prompts: PromptsConfig
    messages: MessagesConfig


def load_orchestration_config(config_path: Path) -> OrchestrationConfig:
    """Load orchestration configuration from a TOML file.

    Args:
        config_path: Path to the TOML configuration file.

    Returns:
        Parsed orchestration configuration.

    Raises:
        RuntimeError: If the file is missing, unreadable, invalid TOML, or
            required keys are missing.

    """
    path = config_path.expanduser().resolve()
    if not path.exists() or not path.is_file():
        msg = f"Config file not found: {path}"
        raise RuntimeError(msg)

    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        msg = f"Failed to read config file: {path}"
        raise RuntimeError(msg) from exc
    except tomllib.TOMLDecodeError as exc:
        msg = f"Invalid TOML in config file: {path}"
        raise RuntimeError(msg) from exc

    try:
        llm = data["llm"]
        orchestration = data["orchestration"]
        prompts = data["prompts"]
        messages = data["messages"]

        return OrchestrationConfig(
            llm=LlmConfig(
                model=str(llm["model"]),
                temperature=float(llm["temperature"]),
                reasoning_effort=str(llm["reasoning_effort"]),
                timeout_s=float(llm["timeout_s"]),
                max_retries=int(llm["max_retries"]),
            ),
            orchestration=OrchestrationRuntimeConfig(
                init_retry_base_s=float(orchestration["init_retry_base_s"]),
                init_retry_max_s=float(orchestration["init_retry_max_s"]),
                router_timeout_s=float(orchestration["router_timeout_s"]),
            ),
            prompts=PromptsConfig(
                folder=str(prompts["folder"]),
                orchestration_prompt_filename=str(
                    prompts["orchestration_prompt_filename"],
                ),
            ),
            messages=MessagesConfig(
                refusal=str(messages["refusal"]),
                error=str(messages["error"]),
                unavailable=str(messages["unavailable"]),
            ),
        )

    except KeyError as exc:
        msg = f"Missing required config key: {exc}"
        raise RuntimeError(msg) from exc
    except (TypeError, ValueError) as exc:
        msg = "Invalid config value type"
        raise RuntimeError(msg) from exc


# ============================
# Prompt loading
# ============================


def resolve_prompts_dir(*, prompts_folder: str) -> Path:
    """Resolve the directory containing prompt templates.

    Args:
        prompts_folder: Folder name relative to this module.

    Returns:
        Resolved absolute path to the prompts directory.

    Raises:
        RuntimeError: If the folder does not exist or is not a directory.

    """
    prompts_dir = (Path(__file__).parent / prompts_folder).resolve()
    if not prompts_dir.exists() or not prompts_dir.is_dir():
        msg = f"Prompts folder not found: {prompts_dir}"
        raise RuntimeError(msg)
    return prompts_dir


def resolve_prompt_path(*, prompts_dir: Path, filename: str) -> Path:
    """Resolve a prompt file within the prompts directory.

    Args:
        prompts_dir: Directory containing prompt templates.
        filename: Prompt filename within prompts_dir.

    Returns:
        Resolved absolute path to the prompt file.

    Raises:
        RuntimeError: If the file does not exist or is not a file.

    """
    candidate = (prompts_dir / filename).resolve()
    if not candidate.exists() or not candidate.is_file():
        msg = f"Prompt file not found: {candidate}"
        raise RuntimeError(msg)
    return candidate


@lru_cache(maxsize=5)
def load_prompt_text(path: Path) -> str:
    """Load and cache prompt text.

    Args:
        path: Path to the prompt file.

    Returns:
        The prompt file contents.

    Raises:
        RuntimeError: If the file cannot be read.

    """
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        msg = f"Failed to read prompt template at: {path}"
        raise RuntimeError(msg) from exc


# ============================
# Paths
# ============================


def _config_dir() -> Path:
    """Resolve the base configuration directory.

    This is one level above the package/module directory.

    Returns:
        Absolute path to the base configuration directory.

    """
    return Path(__file__).resolve().parent.parent


def _orchestration_config_path() -> Path:
    """Get the orchestration TOML configuration path.

    Returns:
        Absolute path to orchestration_config.toml.

    """
    return (_config_dir() / "orchestration_config.toml").resolve()


def _info_config_path() -> Path:
    """Get the info-agent TOML configuration path.

    Returns:
        Absolute path to info_config.toml.

    """
    return (_config_dir() / "info_config.toml").resolve()


def _rooms_sql_config_path() -> Path:
    """Get the rooms SQL agent TOML configuration path.

    Returns:
        Absolute path to rooms_sql_config.toml.

    """
    return (_config_dir() / "rooms_sql_config.toml").resolve()


# ============================
# Routing schema + State
# ============================


type RouteStep = Literal["info", "rooms", "refuse", "error"]


class RouteDecision(BaseModel):
    """Structured router output.

    Attributes:
        step: The next step selected by the router.

    """

    step: RouteStep = Field(
        ...,
        description="The next step to take in answering the user's query.",
    )


class ConversationState(MessagesState, total=False):
    """LangGraph state for orchestration.

    This extends LangGraph's MessagesState with a single optional key `route`
    produced by the router node.

    Attributes:
        route: Router-selected step (optional).

    """

    route: RouteStep


def _route_from_state(state: ConversationState) -> RouteStep:
    """Extract the routing decision from state.

    Args:
        state: Current orchestration state.

    Returns:
        The route step. Defaults to "error" if missing.

    """
    return cast("RouteStep", state.get("route") or "error")


# ============================
# Resources (mirrors rooms_sql_agent.py pattern)
# ============================


class OrchestrationResources:
    """Long-lived dependencies for orchestration.

    This object owns:
    - OrchestrationConfig loaded from TOML.
    - Prompt paths and cached prompt content.
    - Sub-resources for info/rooms agents.
    - Built sub-agents and router runnable.

    Call startup_check() once before using getters.

    """

    __slots__ = (
        "_config",
        "_info_agent",
        "_info_config",
        "_info_resources",
        "_rooms_agent",
        "_rooms_config",
        "_rooms_resources",
        "_router",
        "_system_prompt",
        "_system_prompt_path",
    )

    _config: OrchestrationConfig
    _system_prompt_path: Path
    _info_config: Any
    _info_resources: InfoRagResources
    _info_agent: CompiledStateGraph | None
    _rooms_config: Any
    _rooms_resources: RoomsSqlResources
    _rooms_agent: CompiledStateGraph | None
    _router: Runnable[list[BaseMessage], RouteDecision]
    _system_prompt: str | None

    def __init__(self) -> None:
        """Initialize resources containers and load static configuration."""
        self._config = load_orchestration_config(_orchestration_config_path())

        prompts_dir = resolve_prompts_dir(prompts_folder=self._config.prompts.folder)
        self._system_prompt_path = resolve_prompt_path(
            prompts_dir=prompts_dir,
            filename=self._config.prompts.orchestration_prompt_filename,
        )

        self._info_config = load_info_config(_info_config_path())
        self._info_resources = InfoRagResources(
            redis_url=get_redis_url(),
            config=self._info_config,
        )
        self._info_agent: CompiledStateGraph | None = None

        self._rooms_config = load_rooms_sql_config(_rooms_sql_config_path())
        self._rooms_resources = RoomsSqlResources(
            pgsql_db_url=get_pgsql_db_url(),
            config=self._rooms_config,
        )
        self._rooms_agent: CompiledStateGraph | None = None

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

        self._system_prompt: str | None = None

    async def startup_check(self) -> None:
        """Initialize and validate dependencies.

        This should be called once at startup. If it raises, the caller may
        retry later.

        Raises:
            OperationalError: If resources cannot be initialized.

        """
        try:
            self._system_prompt = load_prompt_text(self._system_prompt_path)

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
        """Close underlying resources.

        This should be called during app shutdown.

        """
        await self._rooms_resources.aclose()
        await self._info_resources.aclose()

    def reset_runtime_state(self) -> None:
        """Clear runtime-built objects after an initialization failure."""
        self._system_prompt = None
        self._info_agent = None
        self._rooms_agent = None

    def get_config(self) -> OrchestrationConfig:
        """Return the loaded orchestration configuration."""
        return self._config

    def get_system_prompt(self) -> str:
        """Return the router system prompt.

        Returns:
            Router system prompt text.

        Raises:
            RuntimeError: If startup_check() has not been called.

        """
        if self._system_prompt is None:
            msg = "System prompt not loaded. Call startup_check() first."
            raise RuntimeError(msg)
        return self._system_prompt

    def get_router(self) -> Runnable[list[BaseMessage], RouteDecision]:
        """Return the router runnable."""
        return self._router

    def get_info_agent(self) -> CompiledStateGraph:
        """Return the compiled info agent.

        Returns:
            CompiledStateGraph for the info agent.

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
            CompiledStateGraph for the rooms agent.

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
    """Compile the orchestration LangGraph from initialized resources."""

    __slots__ = ("_resources",)

    _resources: OrchestrationResources

    def __init__(self, *, resources: OrchestrationResources) -> None:
        """Create the factory.

        Args:
            resources: Initialized resources container.

        """
        self._resources = resources

    def build(self) -> CompiledStateGraph:
        """Build and compile the orchestration graph.

        Returns:
            Compiled orchestration graph.

        """
        resources = self._resources
        cfg = resources.get_config()

        async def router_node(state: ConversationState) -> dict[str, Any]:
            """Route the request to the appropriate sub-agent.

            Args:
                state: Current orchestration state.

            Returns:
                State patch containing the selected route.

            """
            messages = state["messages"]

            async def _invoke() -> RouteDecision:
                """Invoke the router LLM with the system prompt and message history.

                Returns:
                    Parsed RouteDecision produced by the router runnable.

                Raises:
                    Exception: Propagates any exception raised by the underlying LLM
                        call.

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
            except Exception:
                logger.exception("Router failed")
                return {"route": "error"}

            step: RouteStep = getattr(decision, "step", "error")
            logger.info("Router decision: %s", step)
            return {"route": step}

        async def info_node(state: ConversationState) -> dict[str, Any]:
            """Dispatch to the info agent.

            Args:
                state: Current orchestration state.

            Returns:
                Info agent result (state patch).

            """
            logger.info("Dispatching to info agent")
            return cast(
                "dict[str, Any]", await resources.get_info_agent().ainvoke(state),
            )

        async def rooms_node(state: ConversationState) -> dict[str, Any]:
            """Dispatch to the rooms agent.

            Args:
                state: Current orchestration state.

            Returns:
                Rooms agent result (state patch).

            """
            logger.info("Dispatching to rooms agent")
            return cast(
                "dict[str, Any]",
                await resources.get_rooms_agent().ainvoke(state),
            )

        def refuse_node(_: ConversationState) -> dict[str, Any]:
            """Return the configured refusal message.

            Args:
                _: Current orchestration state (unused).

            Returns:
                State patch containing a refusal AIMessage.

            """
            logger.info("Refusing request")
            return {"messages": [AIMessage(content=cfg.messages.refusal)]}

        def error_node(_: ConversationState) -> dict[str, Any]:
            """Return the configured internal error message.

            Args:
                _: Current orchestration state (unused).

            Returns:
                State patch containing an error AIMessage.

            """
            logger.info("Returning error message")
            return {"messages": [AIMessage(content=cfg.messages.error)]}

        graph = StateGraph(ConversationState)
        graph.add_node("router", RunnableLambda(router_node))
        graph.add_node("info", RunnableLambda(info_node))
        graph.add_node("rooms", RunnableLambda(rooms_node))
        graph.add_node("refuse", RunnableLambda(refuse_node))
        graph.add_node("error", RunnableLambda(error_node))

        graph.add_edge(START, "router")
        graph.add_conditional_edges(
            "router",
            _route_from_state,
            {"info": "info", "rooms": "rooms", "refuse": "refuse", "error": "error"},
        )

        graph.add_edge("info", END)
        graph.add_edge("rooms", END)
        graph.add_edge("refuse", END)
        graph.add_edge("error", END)

        return graph.compile()


# ============================
# Manager for FastAPI startup + retry
# ============================


class OrchestrationManager:
    """Operational wrapper around resources + compiled graph.

    This class is intended to be instantiated once and used by the FastAPI
    lifespan to start initialization in the background.

    Requests can call get_agent() to obtain the compiled graph if available.
    If unavailable, callers can return get_unavailable_message().

    """

    __slots__ = ("_agent", "_factory", "_init_task", "_lock", "_resources")

    _resources: OrchestrationResources
    _factory: OrchestrationAgentFactory
    _agent: CompiledStateGraph | None
    _init_task: asyncio.Task[None] | None
    _lock: asyncio.Lock

    def __init__(self) -> None:
        """Initialize the manager."""
        self._resources = OrchestrationResources()
        self._factory = OrchestrationAgentFactory(resources=self._resources)

        self._agent: CompiledStateGraph | None = None
        self._init_task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()

    @property
    def is_ready(self) -> bool:
        """Return whether the compiled orchestration agent is ready.

        Returns:
            True if the orchestration graph has been compiled and is available.

        """
        return self._agent is not None

    def get_agent(self) -> CompiledStateGraph | None:
        """Return the compiled orchestration agent if ready.

        Returns:
            The compiled agent, or None if initialization has not completed.

        """
        return self._agent

    def get_unavailable_message(self) -> str:
        """Return the configured user-facing unavailable message.

        Returns:
            The user-facing message returned while the system is initializing.

        """
        return self._resources.get_config().messages.unavailable

    async def start(self) -> None:
        """Start background initialization.

        This method is idempotent.

        """
        if self._init_task is not None:
            return
        self._init_task = asyncio.create_task(
            self._init_loop(),
            name="orchestration-init",
        )

    async def stop(self) -> None:
        """Stop background initialization and close resources."""
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

    async def _reset_and_backoff(self, *, backoff: float, max_backoff: float) -> float:
        """Reset runtime state and sleep with exponential backoff.

        Args:
            backoff: Current backoff duration.
            max_backoff: Maximum backoff duration.

        Returns:
            Updated backoff duration (capped).

        """
        self._resources.reset_runtime_state()
        self._agent = None
        await asyncio.sleep(backoff)
        return min(backoff * 2.0, max_backoff)

    async def _init_loop(self) -> None:
        """Initialize resources and compile the orchestration graph.

        This method runs as a background task. It attempts to initialize
        dependencies via OrchestrationResources.startup_check(), compiles the
        orchestration graph once, and then sleeps until cancelled.

        If initialization fails, it resets runtime state and retries with
        exponential backoff.

        Raises:
            asyncio.CancelledError: If the background task is cancelled.

        """
        cfg = self._resources.get_config().orchestration
        backoff = cfg.init_retry_base_s

        while True:
            try:
                async with self._lock:
                    if self._agent is None:
                        logger.info("Initializing orchestration resources...")
                        await self._resources.startup_check()
                        self._agent = self._factory.build()
                        logger.info("Orchestration agent ready")

                # Once ready, just sleep until cancelled.
                await asyncio.sleep(3600)

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


""" In your app module (e.g. main.py), you'd do something like:

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
    agent = orchestration.get_agent()
    if agent is None:
        msg = orchestration.get_unavailable_message()
        return {"messages": [{"role": "assistant", "content": msg}]}
    return await agent.ainvoke(payload)

@app.get("/health/orchestration")
async def orchestration_health():
    return {"ready": orchestration.is_ready}"""
