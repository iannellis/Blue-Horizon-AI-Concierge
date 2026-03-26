"""Tests for structured outcome classification in eval/stress/workload.py."""

# ruff: noqa: S101

from eval.stress.workload import _classify_outcome


def _sql_call(
    row: dict[str, object],
    *,
    status: str = "ok",
    error: str | None = None,
) -> dict[str, object]:
    """Build a minimal ``run_sql`` tool-summary entry.

    Args:
        row: First result row to attach to the entry.
        status: Tool status string.
        error: Optional error payload.

    Returns:
        ``run_sql`` summary dictionary.

    """
    entry: dict[str, object] = {
        "tool": "run_sql",
        "status": status,
        "rows": [row],
    }
    if error is not None:
        entry["error"] = error
    return entry


class TestClassifyOutcome:
    """_classify_outcome() prefers SQL outcomes over assistant phrasing."""

    def test_sql_book_success_beats_conflict_sounding_text(self) -> None:
        """Structured SQL success wins over misleading assistant wording."""
        outcome = _classify_outcome(
            op_type="BOOK",
            assistant_text="That room is unavailable.",
            err_text=None,
            sql_calls=[
                _sql_call({"nights_requested": 2, "nights_booked": 2}),
            ],
        )

        assert outcome == "success"

    def test_sql_book_conflict_is_detected_from_row_counts(self) -> None:
        """A partial or zero-night booking is classified as a conflict."""
        outcome = _classify_outcome(
            op_type="BOOK",
            assistant_text="Booked successfully.",
            err_text=None,
            sql_calls=[
                _sql_call({"nights_requested": 2, "nights_booked": 0}),
            ],
        )

        assert outcome == "conflict"

    def test_sql_modify_success_uses_release_and_acquire_counts(self) -> None:
        """Modify success is derived from matching release and acquire counts."""
        outcome = _classify_outcome(
            op_type="MODIFY",
            assistant_text="I could not modify it.",
            err_text=None,
            sql_calls=[
                _sql_call(
                    {
                        "release_needed": 2,
                        "acquire_needed": 2,
                        "nights_released": 2,
                        "nights_acquired": 2,
                    },
                ),
            ],
        )

        assert outcome == "success"

    def test_sql_cancel_success_uses_canceled_night_count(self) -> None:
        """Cancel success is derived from matching requested and canceled nights."""
        outcome = _classify_outcome(
            op_type="CANCEL",
            assistant_text="I could not cancel it.",
            err_text=None,
            sql_calls=[
                _sql_call({"nights_requested": 2, "nights_canceled": 2}),
            ],
        )

        assert outcome == "success"

    def test_python_level_error_short_circuits_to_error(self) -> None:
        """Invocation errors always classify as ``error``."""
        outcome = _classify_outcome(
            op_type="BOOK",
            assistant_text="Booked successfully.",
            err_text="RuntimeError: boom",
            sql_calls=[],
        )

        assert outcome == "error"

    def test_text_fallback_is_used_when_sql_data_is_unavailable(self) -> None:
        """Text heuristics remain the fallback when no SQL result can be parsed."""
        outcome = _classify_outcome(
            op_type="BOOK",
            assistant_text="That room isn't available for those nights.",
            err_text=None,
            sql_calls=[],
        )

        assert outcome == "conflict"
