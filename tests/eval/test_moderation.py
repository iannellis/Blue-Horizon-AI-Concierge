"""Tests for eval/evaluators/_moderation.py, via minimal mock Run/Example objects."""

# ruff: noqa: S101

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

from eval.evaluators._moderation import eval_moderation_flags

if TYPE_CHECKING:
    from langsmith.schemas import Example, Run


def _score(
    turn_outputs: list[dict[str, Any]],
    turns: list[dict[str, Any]],
) -> tuple[float, str]:
    """Run the evaluator on mock objects and return its score and comment.

    Args:
        turn_outputs: Per-turn output dicts, as the eval target records them.
        turns: Per-turn example dicts, optionally labeled ``expect_flagged``.

    Returns:
        The ``moderation_flag_pass`` score and comment.

    """
    run = cast("Run", SimpleNamespace(outputs={"turn_outputs": turn_outputs}))
    example = cast("Example", SimpleNamespace(inputs={"turns": turns}))
    (result,) = eval_moderation_flags(run, example)
    assert result["key"] == "moderation_flag_pass"
    return result["score"], result["comment"]


class TestEvalModerationFlags:
    """eval_moderation_flags() requires moderation to flag the labeled turns only."""

    def test_labeled_turn_flagged_and_clean_turn_unflagged_passes(self) -> None:
        """A flagged threat followed by a clean, unflagged turn passes."""
        score, _ = _score(
            [{"flagged": "message"}, {"flagged": None}],
            [{"expect_flagged": "message"}, {}],
        )
        assert score == 1.0

    def test_labeled_turn_refused_by_router_fails(self) -> None:
        """A labeled turn that moderation missed fails, whatever its route."""
        score, comment = _score(
            [{"flagged": None, "route_pred": "refuse"}],
            [{"expect_flagged": "message"}],
        )
        assert score == 0.0
        assert "turn 0: expected message, got None" in comment

    def test_unlabeled_turn_flagged_by_moderation_fails(self) -> None:
        """A clean turn that moderation flagged, message or reply, fails."""
        for flagged in ("message", "reply"):
            score, _ = _score([{"flagged": flagged}], [{}])
            assert score == 0.0

    def test_injection_flag_is_not_a_moderation_flag(self) -> None:
        """The injection check's flag neither passes nor fails this metric."""
        assert _score([{"flagged": "injection"}], [{}])[0] == 1.0
        labeled = [{"expect_flagged": "message"}]
        assert _score([{"flagged": "injection"}], labeled)[0] == 0.0

    def test_missing_output_for_labeled_turn_fails(self) -> None:
        """A labeled turn with no captured output fails."""
        assert _score([], [{"expect_flagged": "message"}])[0] == 0.0
