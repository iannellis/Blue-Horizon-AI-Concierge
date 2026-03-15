"""Tests for orchestration routing models.

_route_from_state is a pure function operating on dict-like state.
"""

# ruff: noqa: S101

import pytest
from pydantic import ValidationError

from blue_horizon.agents.orchestration.models import RouteDecision, _route_from_state

# ---------------------------------------------------------------------------
# _route_from_state
# ---------------------------------------------------------------------------


class TestRouteFromState:
    """_route_from_state extracts the route key or falls back to 'error'."""

    @pytest.mark.parametrize("route", ["info", "rooms", "refuse", "error"])
    def test_valid_route_returned(self, route: str) -> None:
        """All valid RouteStep values are passed through unchanged."""
        state = {"messages": [], "route": route}
        assert _route_from_state(state) == route  # type: ignore[arg-type]

    def test_none_route_defaults_to_error(self) -> None:
        """A None route value falls back to 'error'."""
        state = {"messages": [], "route": None}
        assert _route_from_state(state) == "error"  # type: ignore[arg-type]

    def test_missing_route_key_defaults_to_error(self) -> None:
        """A state dict without a 'route' key falls back to 'error'."""
        state = {"messages": []}
        assert _route_from_state(state) == "error"  # type: ignore[arg-type]

    def test_empty_string_route_defaults_to_error(self) -> None:
        """An empty-string route is falsy and falls back to 'error'."""
        state = {"messages": [], "route": ""}
        assert _route_from_state(state) == "error"  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# RouteDecision
# ---------------------------------------------------------------------------


class TestRouteDecision:
    """RouteDecision validates its step field."""

    @pytest.mark.parametrize("step", ["info", "rooms", "refuse", "error"])
    def test_valid_steps_accepted(self, step: str) -> None:
        """All four valid steps parse without error."""
        decision = RouteDecision(step=step)  # type: ignore[arg-type]
        assert decision.step == step

    def test_invalid_step_raises(self) -> None:
        """An unknown step string is rejected by Pydantic validation."""
        with pytest.raises(ValidationError):
            RouteDecision(step="unknown")  # type: ignore[arg-type]
