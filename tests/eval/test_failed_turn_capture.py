"""Tests that the eval and stress harnesses read a failed turn structurally.

A failed turn is dropped from the thread's history, so the last AI message in
an orchestration result belongs to an earlier turn. These pin that neither
harness credits a failed turn with that earlier reply.
"""

# ruff: noqa: S101
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage, HumanMessage

from eval.langsmith_target._text_utils import _extract_assistant_text_from_result
from eval.stress.workload import _invoke_orchestration

if TYPE_CHECKING:
    import pytest


class TestExtractAssistantText:
    """`_extract_assistant_text_from_result` honours `turn_error`."""

    def test_failed_turn_yields_empty_text_not_an_earlier_reply(self) -> None:
        """An earlier turn's reply is not returned for a failed turn."""
        result = {
            "messages": [
                HumanMessage(content="First question"),
                AIMessage(content="Earlier reply."),
            ],
            "turn_error": "timeout",
        }
        assert _extract_assistant_text_from_result(result) == ""

    def test_completed_turn_yields_its_reply(self) -> None:
        """A completed turn still yields its own reply."""
        result = {
            "messages": [HumanMessage(content="hi"), AIMessage(content="Hello.")],
            "turn_error": None,
        }
        assert _extract_assistant_text_from_result(result) == "Hello."


class TestInvokeOrchestrationFailedTurn:
    """The stress harness turns a failed turn into an error, structurally."""

    def test_failed_turn_is_an_error_and_skips_auto_confirm(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A failed turn yields error text and never auto-confirms."""
        auto_confirm = AsyncMock()
        monkeypatch.setattr(
            "eval.stress.workload.auto_confirm_pending_proposal", auto_confirm,
        )
        orchestration = MagicMock()
        orchestration.ainvoke = AsyncMock(
            return_value={"messages": [], "turn_error": "internal"},
        )

        assistant_text, err_text = asyncio.run(
            _invoke_orchestration(
                orchestration=orchestration,
                callback=MagicMock(),
                thread_id="t1",
                customer_id=1,
                prompt="Book room 204",
                tags=[],
                metadata={},
            ),
        )

        assert assistant_text == ""
        assert err_text == "turn_error: internal"
        auto_confirm.assert_not_called()
