"""Tests for orchestration routing models.

_merge_part_replies is a pure reducer, so it is tested on plain lists.
"""

# ruff: noqa: S101

import pytest
from pydantic import ValidationError

from blue_horizon.agents.orchestration.models import (
    PartReply,
    RouteDecision,
    RoutePart,
    _merge_part_replies,
)

# ---------------------------------------------------------------------------
# _merge_part_replies
# ---------------------------------------------------------------------------


def _reply(index: int) -> PartReply:
    """Build a successful part reply with no messages.

    Args:
        index: The part's position in the guest's message.

    Returns:
        PartReply: A reply for the info step.

    """
    return PartReply(index=index, step="info", replies=[], turn_error=None)


class TestMergePartReplies:
    """_merge_part_replies appends parallel writes and clears on None."""

    def test_update_is_appended(self) -> None:
        """A node's replies are added after those already collected."""
        first, second = _reply(0), _reply(1)
        assert _merge_part_replies([first], [second]) == [first, second]

    def test_none_clears_the_channel(self) -> None:
        """None empties the channel, so an earlier turn's replies are dropped."""
        assert _merge_part_replies([_reply(0)], None) == []


# ---------------------------------------------------------------------------
# RouteDecision
# ---------------------------------------------------------------------------


class TestRouteDecision:
    """RouteDecision validates each part's step."""

    @pytest.mark.parametrize("step", ["info", "booking", "refuse"])
    def test_valid_steps_accepted(self, step: str) -> None:
        """All three valid steps parse without error."""
        decision = RouteDecision(
            parts=[RoutePart(query="Hello", step=step)],  # type: ignore[arg-type]
        )
        assert decision.parts[0].step == step

    @pytest.mark.parametrize("step", ["unknown", "error"])
    def test_invalid_step_raises(self, step: str) -> None:
        """An unknown step, or the internal error route, is rejected."""
        with pytest.raises(ValidationError):
            RoutePart(query="Hello", step=step)  # type: ignore[arg-type]

    def test_query_is_required(self) -> None:
        """A part without its standalone query is rejected."""
        with pytest.raises(ValidationError):
            RoutePart(step="info")  # type: ignore[call-arg]
