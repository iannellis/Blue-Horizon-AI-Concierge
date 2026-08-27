"""Tests for the judge-adjudication path in eval/evaluators/_booking.py.

Covers _judge_claims_success()'s structured-output success and failure paths
using a monkeypatched _call_judge_llm_structured (no LangSmith, no network,
no database). The rest of _booking.py (DB invariants, tool-outcome scoring)
is covered elsewhere/by db_integration tests and is out of scope here.
"""

# ruff: noqa: S101

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from eval.evaluators import _booking
from eval.evaluators._booking import ClaimsSuccessPayload, _judge_claims_success

if TYPE_CHECKING:
    import pytest


class TestJudgeClaimsSuccess:
    """_judge_claims_success() adjudicates a phrase-matched assistant message."""

    def test_true_verdict_returns_claim_and_rationale(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A genuine success claim returns (True, rationale) from the payload."""

        async def _fake_structured(
            *,
            prompt: str,
            model: str,
            response_model: type[ClaimsSuccessPayload],
        ) -> ClaimsSuccessPayload:
            _ = prompt, model, response_model
            return ClaimsSuccessPayload(
                claims_success=True,
                rationale="States the room is booked with a confirmation number.",
            )

        monkeypatch.setattr(_booking, "_call_judge_llm_structured", _fake_structured)

        claims_success, rationale = asyncio.run(
            _judge_claims_success("Your room is booked! Confirmation: BH000123", "m"),
        )

        assert claims_success is True
        assert "confirmation number" in rationale

    def test_false_verdict_for_a_pending_description(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A message describing a pending proposal is not a success claim."""

        async def _fake_structured(
            *,
            prompt: str,
            model: str,
            response_model: type[ClaimsSuccessPayload],
        ) -> ClaimsSuccessPayload:
            _ = prompt, model, response_model
            return ClaimsSuccessPayload(
                claims_success=False,
                rationale="Describes a pending proposal awaiting confirmation.",
            )

        monkeypatch.setattr(_booking, "_call_judge_llm_structured", _fake_structured)

        claims_success, _rationale = asyncio.run(
            _judge_claims_success(
                "I can propose this booking for your review.", "m",
            ),
        )

        assert claims_success is False

    def test_judge_failure_flags_conservatively(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Any judge failure (unavailable model, invalid payload, ...) flags True.

        An unbacked claim is exactly the failure this metric exists to
        catch, so an inconclusive adjudication must not be waved through --
        matching the pre-structured-output behavior for a parse failure.
        """

        async def _fake_structured(
            *,
            prompt: str,
            model: str,
            response_model: type[ClaimsSuccessPayload],
        ) -> ClaimsSuccessPayload:
            _ = prompt, model, response_model
            msg = "judge call failed"
            raise RuntimeError(msg)

        monkeypatch.setattr(_booking, "_call_judge_llm_structured", _fake_structured)

        claims_success, rationale = asyncio.run(
            _judge_claims_success("some text", "m"),
        )

        assert claims_success is True
        assert "flagging conservatively" in rationale
