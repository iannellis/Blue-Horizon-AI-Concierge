"""Long-lived dependencies for the orchestration agent."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from blue_horizon.agents.exceptions import OperationalError
from blue_horizon.agents.information import InfoAgentFactory, InfoRagResources
from blue_horizon.agents.orchestration.models import RouteDecision
from blue_horizon.agents.prompt_utils import load_packaged_text
from blue_horizon.agents.rooms import RoomsAgentFactory, RoomsSqlResources
from blue_horizon.config import load_app_config

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage
    from langchain_core.runnables import Runnable
    from langgraph.graph.state import CompiledStateGraph

    from blue_horizon.config import InfoRagConfig, OrchestrationConfig, RoomsSqlConfig

logger = logging.getLogger(__name__)


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
            llm.with_structured_output(RouteDecision, method="function_calling"),
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

    def get_info_resources(self) -> InfoRagResources:
        """Return the info RAG resources instance.

        Returns:
            InfoRagResources: Shared retrieval resources.

        """
        return self._info_resources

    def get_rooms_resources(self) -> RoomsSqlResources:
        """Return the rooms SQL resources instance.

        Returns:
            RoomsSqlResources: Shared database resources.

        """
        return self._rooms_resources
