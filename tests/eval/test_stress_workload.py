"""Tests for structured outcome classification in eval/stress/workload.py."""

# ruff: noqa: S101

from eval.stress.workload import _classify_outcome


def _propose_entry(tool: str, *, status: str = "proposed") -> dict[str, object]:
    """Build a minimal `propose_*` tool-summary entry.

    Args:
        tool: One of `propose_booking`, `propose_cancellation`,
            `propose_modification`.
        status: Tool status string (`"proposed"` or `"error"`).

    Returns:
        `propose_*` summary dictionary.

    """
    entry: dict[str, object] = {"tool": tool, "status": status}
    if status == "error":
        entry["error"] = "Room is not available for every night requested."
    else:
        entry["proposal_id"] = "abc123"
    return entry


def _confirm_entry(*, status: str = "ok") -> dict[str, object]:
    """Build a minimal `confirm_booking` tool-summary entry.

    Args:
        status: Tool status string (`"ok"` or `"error"`).

    Returns:
        `confirm_booking` summary dictionary.

    """
    entry: dict[str, object] = {
        "tool": "confirm_booking",
        "status": status,
        "action": "book",
    }
    if status == "error":
        entry["error"] = "That request has expired, shall I check those dates again?"
    else:
        entry["already_confirmed"] = False
        entry["result"] = {"booking_id": 1, "confirmation_number": "BH000001"}
    return entry


class TestClassifyOutcome:
    """_classify_outcome() prefers propose/confirm outcomes over assistant phrasing."""

    def test_confirmed_proposal_beats_conflict_sounding_text(self) -> None:
        """A successful propose+confirm pair wins over misleading assistant wording."""
        outcome = _classify_outcome(
            op_type="BOOK",
            assistant_text="That room is unavailable.",
            err_text=None,
            tool_summary=[
                _propose_entry("propose_booking"),
                _confirm_entry(),
            ],
        )

        assert outcome == "success"

    def test_propose_error_is_a_conflict(self) -> None:
        """A propose_* refusal (nights unavailable) is classified as a conflict."""
        outcome = _classify_outcome(
            op_type="BOOK",
            assistant_text="Booked successfully.",
            err_text=None,
            tool_summary=[
                _propose_entry("propose_booking", status="error"),
            ],
        )

        assert outcome == "conflict"

    def test_confirm_error_is_a_conflict(self) -> None:
        """A confirm_booking failure (lost a race) is classified as a conflict."""
        outcome = _classify_outcome(
            op_type="MODIFY",
            assistant_text="I could not modify it.",
            err_text=None,
            tool_summary=[
                _propose_entry("propose_modification"),
                _confirm_entry(status="error"),
            ],
        )

        assert outcome == "conflict"

    def test_confirmed_cancel_is_a_success(self) -> None:
        """A confirmed cancellation is classified as a success."""
        outcome = _classify_outcome(
            op_type="CANCEL",
            assistant_text="I could not cancel it.",
            err_text=None,
            tool_summary=[
                _propose_entry("propose_cancellation"),
                _confirm_entry(),
            ],
        )

        assert outcome == "success"

    def test_python_level_error_short_circuits_to_error(self) -> None:
        """Invocation errors always classify as ``error``."""
        outcome = _classify_outcome(
            op_type="BOOK",
            assistant_text="Booked successfully.",
            err_text="RuntimeError: boom",
            tool_summary=[],
        )

        assert outcome == "error"

    def test_text_fallback_is_used_when_no_proposal_was_made(self) -> None:
        """Text heuristics remain the fallback when no propose_* call happened."""
        outcome = _classify_outcome(
            op_type="BOOK",
            assistant_text="That room isn't available for those nights.",
            err_text=None,
            tool_summary=[],
        )

        assert outcome == "conflict"
