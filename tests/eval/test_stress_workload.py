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


def _run_sql_entry(*, error_kind: str | None = None) -> dict[str, object]:
    """Build a minimal `run_sql` tool-summary entry.

    Args:
        error_kind: `SqlErrorKind` string to attach, or `None` for a
            successful call (`status="ok"`, no `error_kind`).

    Returns:
        `run_sql` summary dictionary.

    """
    if error_kind is None:
        return {"tool": "run_sql", "status": "ok"}
    return {"tool": "run_sql", "status": "error", "error_kind": error_kind}


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

    def test_run_sql_unavailable_error_kind_beats_conflict_sounding_text(self) -> None:
        """A DB-unavailable run_sql call classifies as error, not text-matched.

        The assistant text below contains "unavailable", which
        `_classify_text_outcome` alone would read as a room conflict. The
        structured `error_kind` must win so an outage is never counted as
        contention.
        """
        outcome = _classify_outcome(
            op_type="BOOK",
            assistant_text="The booking system is temporarily unavailable right now.",
            err_text=None,
            tool_summary=[_run_sql_entry(error_kind="unavailable")],
        )

        assert outcome == "error"

    def test_run_sql_sql_error_kind_falls_back_to_text(self) -> None:
        """A non-unavailable error_kind (an ordinary query error) is not decisive."""
        outcome = _classify_outcome(
            op_type="BOOK",
            assistant_text="That room isn't available for those nights.",
            err_text=None,
            tool_summary=[_run_sql_entry(error_kind="sql")],
        )

        assert outcome == "conflict"

    def test_successful_run_sql_call_does_not_short_circuit(self) -> None:
        """A run_sql call with no error still falls through to other sources."""
        outcome = _classify_outcome(
            op_type="BOOK",
            assistant_text="Booked successfully.",
            err_text=None,
            tool_summary=[
                _run_sql_entry(),
                _propose_entry("propose_booking"),
                _confirm_entry(),
            ],
        )

        assert outcome == "success"
