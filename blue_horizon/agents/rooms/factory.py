"""Factory for building the rooms SQL agent."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

    from blue_horizon.agents.rooms.resources import RoomsSqlResources
    from blue_horizon.config import RoomsSqlConfig


class RoomsAgentFactory:
    """Build the rooms agent using initialized resources.

    Attributes:
        config: Parsed configuration.
        resources: Initialized resources instance.

    """

    __slots__ = ("config", "resources")

    config: RoomsSqlConfig
    resources: RoomsSqlResources

    def __init__(self, *, config: RoomsSqlConfig, resources: RoomsSqlResources) -> None:
        """Construct the rooms agent factory.

        Args:
            config: Parsed configuration loaded from TOML.
            resources: Initialized resources instance.

        """
        self.config = config
        self.resources = resources

    def build(self) -> CompiledStateGraph:
        """Build and return a compiled rooms agent.

        Returns:
            Compiled LangGraph agent.

        Raises:
            RuntimeError: If resources were not initialized.

        """
        system_prompt = self.resources.get_system_prompt()

        llm = ChatOpenAI(
            model=self.config.llm.model,
            reasoning={"effort": self.config.llm.reasoning_effort},
            timeout=self.config.llm.timeout_s,
            max_retries=self.config.llm.max_retries,
        )

        @tool(parse_docstring=True)
        async def run_sql(query: str) -> dict[str, Any]:
            """Execute a single SQL statement and return rows.

            Args:
                query: One SQL statement (no semicolons). May be SELECT or DML. If DML
                    uses RETURNING, returned rows are provided.

            Returns:
                Dict with keys:
                  - status: str ("ok" or "error")
                  - rows: list[dict[str, Any]]
                  - truncated: bool
                  - rowcount: int
                  - error: str (only present on failure)

            """
            return await self.resources.execute_sql(query)

        return create_agent(
            model=llm,
            tools=[run_sql],
            system_prompt=system_prompt,
        )
