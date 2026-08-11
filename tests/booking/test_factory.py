"""Tests for `BookingAgentFactory.build()`.

These tests are deliberately DB-independent: `build()` never touches either
connection pool, it only decorates and schema-builds the `@tool` closures and
compiles the LangGraph agent around them. That schema-building step is where
two real bugs lived undetected until manual verification caught them --
`RunnableConfig` imported only under `TYPE_CHECKING` (raises `NameError` the
first time a tool with a `config: RunnableConfig` parameter is built) and a
missing `Args:` entry in a Google-style docstring (raises `ValueError: Found
invalid Google-Style docstring.`) -- so simply calling `build()` without it
raising is itself the regression test for both. The additional assertions
below pin down exactly what a correct build looks like, so a future variant
of either bug (e.g. on a newly added tool) fails loudly instead of silently
shipping a broken tool schema.
"""
# ruff: noqa: S101

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from blue_horizon.agents.booking.factory import BookingAgentFactory
from blue_horizon.config import BookingSqlConfig

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

# Every tool the factory is expected to expose to the model, and which of
# them carry a server-injected `config: RunnableConfig` parameter that must
# never appear in the model-facing schema.
_EXPECTED_TOOL_NAMES = frozenset({
    "run_sql",
    "list_my_bookings",
    "propose_booking",
    "propose_cancellation",
    "propose_modification",
})
_TOOLS_WITH_INJECTED_CONFIG = frozenset({
    "list_my_bookings",
    "propose_booking",
    "propose_cancellation",
    "propose_modification",
})

_BOOKING_CONFIG_DICT: dict[str, Any] = {
    "llm": {
        "model": "gpt-5-mini",
        "reasoning_effort": "low",
        "timeout_s": 20.0,
        "max_retries": 2,
    },
    "prompts": {
        "folder": "system_prompts",
        "system_prompt_filename": "rooms_sql_prompt.txt",
    },
    "agent": {"top_k": 4},
    "db": {
        "pool": {"min_size": 0, "max_size": 10, "timeout_s": 10.0, "max_idle_s": 240.0},
        "guardrails": {"max_rows": 50, "allow_only_hotel_tables": True},
        "retry": {"max_transient_retries": 1, "transient_retry_backoff_s": 0.15},
    },
    "proposals": {"ttl_s": 1800.0},
}


class _StubBookingSqlResources:
    """Stand-in for `BookingSqlResources` that never opens a DB connection.

    `build()` only calls `get_system_prompt()` directly; every other
    resource (`execute_sql`, `write_pool`, `proposals`) is captured by
    closures that run inside a tool body, never during the build itself, so
    this stub does not need to implement them for these tests.

    """

    def get_system_prompt(self) -> str:
        """Return a fixed stand-in system prompt.

        Returns:
            A short fixed prompt string.

        """
        return "You are a test booking assistant."


@pytest.fixture
def booking_config() -> BookingSqlConfig:
    """Build a minimal, valid `BookingSqlConfig` for factory tests.

    Returns:
        Parsed `BookingSqlConfig`.

    """
    return BookingSqlConfig.model_validate(_BOOKING_CONFIG_DICT)


@pytest.fixture(autouse=True)
def _dummy_openai_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure `ChatOpenAI` construction never depends on a real API key.

    `build()` constructs a `ChatOpenAI` client but never calls it, so a
    syntactically valid dummy key is sufficient.

    Args:
        monkeypatch: Pytest fixture used to set the environment variable.

    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy-key")


def _build_graph(booking_config: BookingSqlConfig) -> CompiledStateGraph:
    """Build the booking agent graph against the stub resources.

    Args:
        booking_config: Parsed booking configuration.

    Returns:
        The compiled LangGraph agent.

    """
    factory = BookingAgentFactory(
        config=booking_config, resources=_StubBookingSqlResources(),
    )
    return factory.build()


def _tools_by_name(graph: CompiledStateGraph) -> dict[str, Any]:
    """Extract the built `@tool` objects, keyed by name, from a compiled graph.

    Args:
        graph: A graph produced by `BookingAgentFactory.build()`.

    Returns:
        Mapping of tool name to the LangChain tool object.

    """
    return graph.nodes["tools"].bound.tools_by_name


def test_build_succeeds_and_exposes_expected_tools(
    booking_config: BookingSqlConfig,
) -> None:
    """`build()` compiles without error and exposes exactly the five tools.

    This is the regression test for both startup-blocking bugs: either one
    raises inside `build()`, before this assertion is ever reached.

    Args:
        booking_config: Parsed booking configuration fixture.

    """
    graph = _build_graph(booking_config)
    assert set(_tools_by_name(graph)) == _EXPECTED_TOOL_NAMES


def test_tool_schemas_exclude_server_injected_config(
    booking_config: BookingSqlConfig,
) -> None:
    """`config: RunnableConfig` never appears in a tool's model-facing schema.

    Args:
        booking_config: Parsed booking configuration fixture.

    """
    tools = _tools_by_name(_build_graph(booking_config))
    for name in _TOOLS_WITH_INJECTED_CONFIG:
        schema_properties = tools[name].args_schema.model_json_schema()["properties"]
        assert "config" not in schema_properties


def test_run_sql_schema_has_only_query(booking_config: BookingSqlConfig) -> None:
    """`run_sql`, the one tool without injected config, exposes only `query`.

    Args:
        booking_config: Parsed booking configuration fixture.

    """
    tools = _tools_by_name(_build_graph(booking_config))
    schema_properties = tools["run_sql"].args_schema.model_json_schema()["properties"]
    assert set(schema_properties) == {"query"}


def test_every_tool_has_a_non_empty_description(
    booking_config: BookingSqlConfig,
) -> None:
    """Every tool built by the factory carries a non-empty model-facing description.

    `parse_docstring=True` derives this from each tool's Google-style
    docstring; an empty or missing description would mean the docstring
    failed to parse in a way that did not raise.

    Args:
        booking_config: Parsed booking configuration fixture.

    """
    tools = _tools_by_name(_build_graph(booking_config))
    for name, tool_obj in tools.items():
        assert tool_obj.description, f"{name} has no description"
