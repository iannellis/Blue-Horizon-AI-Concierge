"""Long-lived dependencies for the orchestration agent."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

from langgraph.checkpoint.memory import MemorySaver

from blue_horizon.agents._lifecycle import require
from blue_horizon.agents._llm import build_chat_model
from blue_horizon.agents.booking import BookingSqlResources, build_booking_agent
from blue_horizon.agents.exceptions import OperationalError
from blue_horizon.agents.information import InfoRagResources, build_info_agent
from blue_horizon.agents.orchestration.models import RouteDecision
from blue_horizon.agents.prompt_utils import load_packaged_text, prompt_resource_path
from blue_horizon.config import load_app_config

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage
    from langchain_core.runnables import Runnable
    from langgraph.graph.state import CompiledStateGraph

    from blue_horizon.config import (
        BookingSqlConfig,
        InfoRagConfig,
        OrchestrationConfig,
    )

logger = logging.getLogger(__name__)


class OrchestrationResources:
    """Long-lived dependencies for orchestration.

    Responsibilities:
        - Load orchestration configuration.
        - Resolve and load the orchestration system prompt.
        - Construct and hold the router LLM runnable.
        - Construct and hold sub-resources (InfoRagResources, BookingSqlResources).
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

    config: OrchestrationConfig
    _system_prompt_resource: str
    _system_prompt: str | None

    _info_config: InfoRagConfig
    info_resources: InfoRagResources
    _info_agent: CompiledStateGraph | None

    _booking_config: BookingSqlConfig
    booking_resources: BookingSqlResources
    _booking_agent: CompiledStateGraph | None

    router: Runnable[list[BaseMessage], RouteDecision]
    checkpointer: MemorySaver

    def __init__(
        self,
        *,
        pgsql_rw_db_url: str | None = None,
        pgsql_ro_db_url: str | None = None,
    ) -> None:
        """Initialize orchestration resources.

        This constructor performs only lightweight work: it reads the orchestration
        TOML, resolves the orchestration prompt path, constructs resource objects,
        and builds the router runnable.

        Heavy work (connectivity checks, agent compilation) occurs in startup_check().

        Args:
            pgsql_rw_db_url: Optional read-write database URL override for the
                booking SQL agent (`bh_agent_rw`).  When provided, takes
                precedence over the ``PGSQL_RW_DB_URL`` value from the
                application configuration.  Pass this when the caller (e.g.,
                the eval/stress harnesses) operates against a separate
                database and needs the agent to write to that same database
                so that reconciliation queries see the changes.
            pgsql_ro_db_url: Optional read-only database URL override for the
                booking SQL agent (`bh_agent_ro`), used exclusively by
                `run_sql`.  When provided, takes precedence over the
                ``PGSQL_RO_DB_URL`` application configuration value.  Must be
                overridden together with `pgsql_rw_db_url`: the two must point
                at the same database, just different roles.

        Raises:
            RuntimeError: If configuration or prompt resolution fails.

        """
        app_config = load_app_config()
        self.config = app_config.orchestration

        self._system_prompt_resource = prompt_resource_path(
            self.config.prompts.folder,
            self.config.prompts.orchestration_prompt_filename,
        )
        self._system_prompt = None

        self._info_config = app_config.info
        self.info_resources = InfoRagResources(
            redis_url=app_config.redis_url,
            config=self._info_config,
        )
        self._info_agent = None

        self._booking_config = app_config.booking
        self.booking_resources = BookingSqlResources(
            pgsql_rw_db_url=pgsql_rw_db_url or app_config.pgsql_rw_db_url,
            pgsql_ro_db_url=pgsql_ro_db_url or app_config.pgsql_ro_db_url,
            config=self._booking_config,
        )
        self._booking_agent = None

        llm = build_chat_model(self.config.llm)
        self.router = cast(
            "Runnable[list[BaseMessage], RouteDecision]",
            llm.with_structured_output(RouteDecision, method="function_calling"),
        )

        self.checkpointer = MemorySaver()

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

            await self.info_resources.startup_check()
            self._info_agent = build_info_agent(
                resources=self.info_resources,
                config=self._info_config,
            )

            await self.booking_resources.startup_check()
            self._booking_agent = build_booking_agent(
                config=self._booking_config,
                resources=self.booking_resources,
            )

        except OperationalError:
            raise
        except Exception as exc:
            msg = "Orchestration resources failed during startup"
            raise OperationalError(msg) from exc

    async def aclose(self) -> None:
        """Close underlying resource pools/clients.

        This should be called during FastAPI shutdown.

        """
        await self.booking_resources.aclose()
        await self.info_resources.aclose()

    def reset_runtime_state(self) -> None:
        """Clear runtime fields after a failed initialization attempt.

        This enables the manager to retry startup cleanly without re-instantiating
        the resources object.

        """
        self._system_prompt = None
        self._info_agent = None
        self._booking_agent = None

    def get_system_prompt(self) -> str:
        """Return the orchestration system prompt text.

        Returns:
            System prompt content.

        Raises:
            RuntimeError: If startup_check() has not been called.

        """
        return require(self._system_prompt, "System prompt")

    def get_info_agent(self) -> CompiledStateGraph:
        """Return the compiled info agent.

        Returns:
            Compiled info agent.

        Raises:
            RuntimeError: If startup_check() has not been called.

        """
        return require(self._info_agent, "Info agent")

    def get_booking_agent(self) -> CompiledStateGraph:
        """Return the compiled booking agent.

        Returns:
            Compiled booking agent.

        Raises:
            RuntimeError: If startup_check() has not been called.

        """
        return require(self._booking_agent, "Booking agent")
