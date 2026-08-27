"""Tests for the tool-outcome and unbacked-success-claim scoring in _booking.py.

`eval_booking_outcome_and_invariants` itself also runs DB invariant checks
(`_check_booking_db_invariants`), which needs a real database and is out of
scope here. This file covers the pure, DB-free half: `_score_booking_tool_
outcomes` and `_score_unbacked_success_claims` (via a monkeypatched judge),
plus their pure helpers -- none of which had direct test coverage before
this file.
"""

# ruff: noqa: S101

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

from eval.evaluators import _booking
from eval.evaluators._booking import (
    ClaimsSuccessPayload,
    _detect_propose_confirm,
    _has_backing_confirm,
    _has_tool_error,
    _score_atomic_outcome,
    _score_booking_tool_outcomes,
    _score_outcome_match,
    _score_unbacked_success_claims,
)
from eval.models import ToolSummaryEntry

if TYPE_CHECKING:
    import pytest
    from langsmith.schemas import Example, Run

    from eval.config import EvalConfig


def _make_run(turn_outputs: list[dict[str, Any]]) -> Run:
    """Build a minimal run-like object for testing.

    Args:
        turn_outputs: List of per-turn output dicts.

    Returns:
        Run: A `SimpleNamespace` with an `outputs` attribute, cast to `Run`.

    """
    return cast("Run", SimpleNamespace(outputs={"turn_outputs": turn_outputs}))


def _make_example(turns: list[dict[str, Any]]) -> Example:
    """Build a minimal example-like object for testing.

    Args:
        turns: List of turn dicts, each optionally containing expected_success.

    Returns:
        Example: A `SimpleNamespace` with an `inputs` attribute, cast to `Example`.

    """
    return cast("Example", SimpleNamespace(inputs={"turns": turns}))


def _make_cfg(*, json_value_max: int = 10_000, judge_model: str = "m") -> EvalConfig:
    """Build a minimal cfg-like object for testing.

    Args:
        json_value_max: Maximum JSON string length.
        judge_model: Judge model name.

    Returns:
        EvalConfig: A `SimpleNamespace` with `evaluator_limits` and `judge`
        attributes, cast to `EvalConfig`.

    """
    limits = SimpleNamespace(json_value_max=json_value_max)
    judge = SimpleNamespace(model=judge_model)
    return cast("EvalConfig", SimpleNamespace(evaluator_limits=limits, judge=judge))


# ---------------------------------------------------------------------------
# _has_tool_error
# ---------------------------------------------------------------------------


class TestHasToolError:
    """_has_tool_error detects an error status or message."""

    def test_status_error_is_an_error(self) -> None:
        """A status of 'error' (any case) counts as an error."""
        assert _has_tool_error(ToolSummaryEntry(tool="run_sql", status="Error")) is True

    def test_error_message_without_status_is_an_error(self) -> None:
        """A non-empty error message counts even without an error status."""
        entry = ToolSummaryEntry(tool="run_sql", status="ok", error="refused")
        assert _has_tool_error(entry) is True

    def test_ok_status_no_error_message_is_not_an_error(self) -> None:
        """A status of 'ok' with no error message is not an error."""
        assert _has_tool_error(ToolSummaryEntry(tool="run_sql", status="ok")) is False


# ---------------------------------------------------------------------------
# _detect_propose_confirm
# ---------------------------------------------------------------------------


class TestDetectProposeConfirm:
    """_detect_propose_confirm finds the last propose_*/confirm_booking entries."""

    def test_no_propose_call_returns_all_none(self) -> None:
        """A turn with no propose_* call returns (None, None, None)."""
        action, propose, confirm = _detect_propose_confirm(
            [ToolSummaryEntry(tool="run_sql")],
        )
        assert action is None
        assert propose is None
        assert confirm is None

    def test_propose_without_confirm(self) -> None:
        """A propose_* call with no confirm_booking yields a None confirm entry."""
        propose_entry = ToolSummaryEntry(tool="propose_booking", status="proposed")
        action, propose, confirm = _detect_propose_confirm([propose_entry])
        assert action == "book"
        assert propose is propose_entry
        assert confirm is None

    def test_propose_and_confirm_paired(self) -> None:
        """A propose_* call followed by confirm_booking pairs them both."""
        propose_entry = ToolSummaryEntry(tool="propose_cancellation", status="proposed")
        confirm_entry = ToolSummaryEntry(tool="confirm_booking", status="ok")
        action, propose, confirm = _detect_propose_confirm(
            [propose_entry, confirm_entry],
        )
        assert action == "cancel"
        assert propose is propose_entry
        assert confirm is confirm_entry

    def test_last_propose_wins_when_called_twice(self) -> None:
        """Only the store's last pending proposal for the turn is used."""
        first = ToolSummaryEntry(tool="propose_booking", status="error")
        second = ToolSummaryEntry(tool="propose_booking", status="proposed")
        _action, propose, _confirm = _detect_propose_confirm([first, second])
        assert propose is second


# ---------------------------------------------------------------------------
# _score_atomic_outcome
# ---------------------------------------------------------------------------


class TestScoreAtomicOutcome:
    """_score_atomic_outcome scores one propose+confirm pair."""

    def test_propose_error_fails_at_propose_stage(self) -> None:
        """A propose_* error fails before confirm is ever considered."""
        propose_entry = ToolSummaryEntry(
            tool="propose_booking", status="error", error="no availability",
        )
        success, detail = _score_atomic_outcome(
            action="book", propose_entry=propose_entry, confirm_entry=None,
        )
        assert success is False
        assert detail["stage"] == "propose"
        assert detail["error"] == "no availability"

    def test_never_confirmed_fails_at_confirm_stage(self) -> None:
        """A successful propose with no confirm_booking entry fails."""
        propose_entry = ToolSummaryEntry(tool="propose_booking", status="proposed")
        success, detail = _score_atomic_outcome(
            action="book", propose_entry=propose_entry, confirm_entry=None,
        )
        assert success is False
        assert detail["stage"] == "confirm"

    def test_confirm_error_fails(self) -> None:
        """A confirm_booking entry with status error fails."""
        propose_entry = ToolSummaryEntry(tool="propose_booking", status="proposed")
        confirm_entry = ToolSummaryEntry(
            tool="confirm_booking", status="error", error="race lost",
        )
        success, detail = _score_atomic_outcome(
            action="book", propose_entry=propose_entry, confirm_entry=confirm_entry,
        )
        assert success is False
        assert detail["stage"] == "confirm"
        assert detail["error"] == "race lost"

    def test_full_success(self) -> None:
        """A clean propose + confirm ok succeeds."""
        propose_entry = ToolSummaryEntry(tool="propose_booking", status="proposed")
        confirm_entry = ToolSummaryEntry(
            tool="confirm_booking", status="ok", already_confirmed=False,
        )
        success, detail = _score_atomic_outcome(
            action="book", propose_entry=propose_entry, confirm_entry=confirm_entry,
        )
        assert success is True
        assert detail["already_confirmed"] is False


# ---------------------------------------------------------------------------
# _score_outcome_match
# ---------------------------------------------------------------------------


class TestScoreOutcomeMatch:
    """_score_outcome_match wraps atomic scoring with expected_success handling."""

    def test_no_propose_call_returns_none(self) -> None:
        """A pure-search turn (no propose_* call) has nothing to score."""
        matched, detail = _score_outcome_match(
            turn_idx=0, tool_summary=[ToolSummaryEntry(tool="run_sql")],
            expected_success=None,
        )
        assert matched is None
        assert detail is None

    def test_success_matches_regardless_of_expectation(self) -> None:
        """A successful write always counts as matched."""
        tool_summary = [
            ToolSummaryEntry(tool="propose_booking", status="proposed"),
            ToolSummaryEntry(tool="confirm_booking", status="ok"),
        ]
        matched, detail = _score_outcome_match(
            turn_idx=0, tool_summary=tool_summary, expected_success=True,
        )
        assert matched is True
        assert detail is None

    def test_expected_failure_waives_the_failure(self) -> None:
        """expected_success=False means a failed propose is not penalised."""
        tool_summary = [
            ToolSummaryEntry(tool="propose_booking", status="error", error="taken"),
        ]
        matched, detail = _score_outcome_match(
            turn_idx=1, tool_summary=tool_summary, expected_success=False,
        )
        assert matched is True
        assert detail is None

    def test_unexpected_failure_is_flagged(self) -> None:
        """A failure the dataset did not expect is flagged with turn/action/detail."""
        tool_summary = [
            ToolSummaryEntry(tool="propose_booking", status="error", error="taken"),
        ]
        matched, detail = _score_outcome_match(
            turn_idx=3, tool_summary=tool_summary, expected_success=None,
        )
        assert matched is False
        assert detail is not None
        assert detail["turn"] == 3  # noqa: PLR2004
        assert detail["action"] == "book"


# ---------------------------------------------------------------------------
# _has_backing_confirm
# ---------------------------------------------------------------------------


class TestHasBackingConfirm:
    """_has_backing_confirm checks for a successful confirm_booking entry."""

    def test_no_confirm_entry(self) -> None:
        """No confirm_booking entry at all -> False."""
        assert _has_backing_confirm([ToolSummaryEntry(tool="propose_booking")]) is False

    def test_confirm_entry_with_error_status(self) -> None:
        """A confirm_booking entry that errored does not count as backing."""
        entry = ToolSummaryEntry(tool="confirm_booking", status="error")
        assert _has_backing_confirm([entry]) is False

    def test_confirm_entry_ok(self) -> None:
        """A confirm_booking entry with status ok counts as backing."""
        entry = ToolSummaryEntry(tool="confirm_booking", status="ok")
        assert _has_backing_confirm([entry]) is True


# ---------------------------------------------------------------------------
# _score_booking_tool_outcomes -- end-to-end, DB-free
# ---------------------------------------------------------------------------


class TestScoreBookingToolOutcomes:
    """_score_booking_tool_outcomes aggregates outcome metrics for a run."""

    def test_all_outcomes_matched(self) -> None:
        """A run where every booking turn completes cleanly scores 1.0."""
        run = _make_run(
            [
                {
                    "route_pred": "booking",
                    "tool_summary": [
                        {"tool": "propose_booking", "status": "proposed"},
                        {"tool": "confirm_booking", "status": "ok"},
                    ],
                },
            ],
        )
        example = _make_example([{"user": "book a room"}])
        results = _score_booking_tool_outcomes(run, example, _make_cfg())
        by_key = {r["key"]: r for r in results}
        assert by_key["booking_tool_errors"]["score"] == 1.0
        assert by_key["booking_no_unexpected_failure_rate"]["score"] == 1.0

    def test_unexpected_failure_lowers_match_rate(self) -> None:
        """An unexpected propose failure shows up as an unexpected-failure metric."""
        run = _make_run(
            [
                {
                    "route_pred": "booking",
                    "tool_summary": [
                        {
                            "tool": "propose_booking",
                            "status": "error",
                            "error": "no availability",
                        },
                    ],
                },
            ],
        )
        example = _make_example([{"user": "book a room"}])
        results = _score_booking_tool_outcomes(run, example, _make_cfg())
        by_key = {r["key"]: r for r in results}
        assert by_key["booking_no_unexpected_failure_rate"]["score"] == 0.0
        assert "booking_unexpected_failures" in by_key

    def test_expected_failure_does_not_count_as_unexpected(self) -> None:
        """An expected_success=False turn's failure does not lower the match rate."""
        run = _make_run(
            [
                {
                    "route_pred": "booking",
                    "tool_summary": [
                        {
                            "tool": "propose_booking",
                            "status": "error",
                            "error": "unavailable",
                        },
                    ],
                },
            ],
        )
        example = _make_example(
            [{"user": "book an impossible room", "expected_success": False}],
        )
        results = _score_booking_tool_outcomes(run, example, _make_cfg())
        by_key = {r["key"]: r for r in results}
        assert by_key["booking_no_unexpected_failure_rate"]["score"] == 1.0
        assert by_key["booking_tool_errors"]["score"] == 1.0

    def test_non_booking_turns_are_ignored(self) -> None:
        """Info-routed turns contribute nothing to the outcome metrics."""
        run = _make_run([{"route_pred": "info", "tool_summary": []}])
        example = _make_example([{"user": "what time is checkout?"}])
        results = _score_booking_tool_outcomes(run, example, _make_cfg())
        by_key = {r["key"]: r for r in results}
        assert "booking_tool_errors" not in by_key
        assert "booking_no_unexpected_failure_rate" not in by_key

    def test_run_sql_rowcount_sanity(self) -> None:
        """run_sql calls with a rowcount present score full rowcount sanity."""
        run = _make_run(
            [
                {
                    "route_pred": "booking",
                    "tool_summary": [{"tool": "run_sql", "rowcount": 2}],
                },
            ],
        )
        example = _make_example([{"user": "any rooms free?"}])
        results = _score_booking_tool_outcomes(run, example, _make_cfg())
        by_key = {r["key"]: r for r in results}
        assert by_key["booking_rowcount_sanity"]["score"] == 1.0


# ---------------------------------------------------------------------------
# _score_unbacked_success_claims (judge mocked, no live calls)
# ---------------------------------------------------------------------------


class TestScoreUnbackedSuccessClaims:
    """_score_unbacked_success_claims flags claims the judge confirms unbacked."""

    def test_no_booking_turns_returns_empty(self) -> None:
        """A run with no booking-routed turns produces no metrics at all."""
        run = _make_run([{"route_pred": "info", "assistant_text": "hi"}])
        results = asyncio.run(_score_unbacked_success_claims(run, _make_cfg()))
        assert results == []

    def test_no_phrase_match_passes_without_a_judge_call(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Text matching no success phrase never reaches the judge."""

        async def _fail_if_called(text: str, model: str) -> tuple[bool, str]:
            _ = text, model
            msg = "judge should not have been called"
            raise AssertionError(msg)

        monkeypatch.setattr(_booking, "_judge_claims_success", _fail_if_called)
        run = _make_run(
            [{"route_pred": "booking", "assistant_text": "Let me check availability."}],
        )
        results = asyncio.run(_score_unbacked_success_claims(run, _make_cfg()))
        by_key = {r["key"]: r for r in results}
        assert by_key["booking_no_unbacked_success_claims"]["score"] == 1.0

    def test_phrase_match_with_backing_confirm_is_not_flagged(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A success phrase backed by a real confirm is never sent to the judge."""

        async def _fail_if_called(text: str, model: str) -> tuple[bool, str]:
            _ = text, model
            msg = "judge should not have been called"
            raise AssertionError(msg)

        monkeypatch.setattr(_booking, "_judge_claims_success", _fail_if_called)
        run = _make_run(
            [
                {
                    "route_pred": "booking",
                    "assistant_text": "Your room is booked! Confirmation: BH000123",
                    "tool_summary": [{"tool": "confirm_booking", "status": "ok"}],
                },
            ],
        )
        results = asyncio.run(_score_unbacked_success_claims(run, _make_cfg()))
        by_key = {r["key"]: r for r in results}
        assert by_key["booking_no_unbacked_success_claims"]["score"] == 1.0

    def test_unbacked_claim_flagged_by_judge(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A success phrase with no backing confirm, judged true, is flagged."""

        async def _fake_judge(text: str, model: str) -> tuple[bool, str]:
            _ = text, model
            return True, "claims a completed booking"

        monkeypatch.setattr(_booking, "_judge_claims_success", _fake_judge)
        run = _make_run(
            [
                {
                    "route_pred": "booking",
                    "assistant_text": "Your room is booked!",
                    "tool_summary": [],
                },
            ],
        )
        results = asyncio.run(_score_unbacked_success_claims(run, _make_cfg()))
        by_key = {r["key"]: r for r in results}
        assert by_key["booking_no_unbacked_success_claims"]["score"] == 0.0
        assert "booking_unbacked_success_details" in by_key

    def test_judge_says_not_a_claim_passes(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A phrase match the judge rules is not actually a claim passes clean."""

        async def _fake_judge(text: str, model: str) -> tuple[bool, str]:
            _ = text, model
            return False, "describes a pending proposal"

        monkeypatch.setattr(_booking, "_judge_claims_success", _fake_judge)
        run = _make_run(
            [
                {
                    "route_pred": "booking",
                    "assistant_text": "I can propose this booking for confirmation.",
                    "tool_summary": [],
                },
            ],
        )
        results = asyncio.run(_score_unbacked_success_claims(run, _make_cfg()))
        by_key = {r["key"]: r for r in results}
        assert by_key["booking_no_unbacked_success_claims"]["score"] == 1.0


def test_claims_success_payload_roundtrip() -> None:
    """ClaimsSuccessPayload parses the judge's expected shape."""
    payload = ClaimsSuccessPayload.model_validate(
        {"claims_success": True, "rationale": "states a confirmation number"},
    )
    assert payload.claims_success is True
