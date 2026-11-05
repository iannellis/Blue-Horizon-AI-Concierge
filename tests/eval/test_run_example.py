"""Tests for what `run_example` records for each turn."""
# ruff: noqa: S101

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage

from eval.langsmith_target import target

if TYPE_CHECKING:
    import pytest

_REFUSAL = "I'm sorry, I cannot help with that query."


def test_turn_outputs_carry_graph_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """`flagged`, `turn_error`, and the state's route reach each turn output.

    Args:
        monkeypatch: Pytest fixture used to swap in a fake manager.

    """
    states: list[dict[str, Any]] = [
        {
            "messages": [AIMessage(content=_REFUSAL)],
            "flagged": "message",
            "route": "refuse",
        },
        # A failed turn adds no message, so the last AI message is turn 1's.
        {
            "messages": [AIMessage(content=_REFUSAL)],
            "route": "info",
            "turn_error": "timeout",
        },
    ]
    manager = MagicMock()
    manager.ainvoke = AsyncMock(side_effect=states)
    monkeypatch.setattr(
        target, "ensure_orchestration_ready", AsyncMock(return_value=manager),
    )
    example = {
        "case_id": "case_x",
        "turns": [{"user": "threat"}, {"user": "What time is checkout?"}],
        "tags": ["no_auto_confirm"],
    }

    result = asyncio.run(target.run_example(example, cfg=MagicMock()))

    first, second = result["turn_outputs"]
    assert first["flagged"] == "message"
    assert first["route_pred"] == "refuse"
    assert first["assistant_text"] == _REFUSAL
    assert first["turn_error"] is None
    assert second["flagged"] is None
    assert second["route_pred"] == "info"
    assert not second["assistant_text"]
    assert second["turn_error"] == "timeout"
