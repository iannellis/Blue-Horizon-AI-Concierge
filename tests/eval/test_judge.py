"""Tests for eval/evaluators/_judge.py.

Covers eval_llm_rubrics()'s structured-output success and failure paths
using a monkeypatched _call_judge_llm_structured (no LangSmith, no network).
"""

# ruff: noqa: S101

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest

from eval.evaluators import _judge
from eval.evaluators._judge import RubricPayload, RubricScore, eval_llm_rubrics

if TYPE_CHECKING:
    from langsmith.schemas import Example, Run

    from eval.config import EvalConfig

_CONSUMER_QUALITY_SCORE = 4.5
_INJECTION_RESISTANCE_SCORE = 5.0
_GROUNDING_FAITHFULNESS_SCORE = 3.0


def _make_run(turn_outputs: list[dict[str, Any]]) -> Run:
    """Build a minimal run-like object for testing.

    Args:
        turn_outputs: List of per-turn output dicts.

    Returns:
        Run: A `SimpleNamespace` with an `outputs` attribute, cast to `Run`
        since `eval_llm_rubrics` only ever reads that attribute off it.

    """
    return cast("Run", SimpleNamespace(outputs={"turn_outputs": turn_outputs}))


def _make_example(turns: list[dict[str, Any]]) -> Example:
    """Build a minimal example-like object for testing.

    Args:
        turns: List of turn dicts with user messages.

    Returns:
        Example: A `SimpleNamespace` with an `inputs` attribute, cast to
        `Example` since `eval_llm_rubrics` only ever reads that attribute.

    """
    return cast("Example", SimpleNamespace(inputs={"turns": turns}))


def _make_cfg() -> EvalConfig:
    """Build a minimal cfg-like object for testing.

    Returns:
        EvalConfig: A `SimpleNamespace` with `judge` and `evaluator_limits`
        attributes, cast to `EvalConfig` since `eval_llm_rubrics` only ever
        reads those two attributes off it.

    """
    judge = SimpleNamespace(model="gemini-test", info_cards_max=3)
    limits = SimpleNamespace(
        context_max_chars=500,
        context_max_items=5,
        assistant_max_chars=500,
        user_max_chars=500,
    )
    return cast(
        "EvalConfig", SimpleNamespace(judge=judge, evaluator_limits=limits),
    )


def _make_rubric_payload() -> RubricPayload:
    """Build a fully-populated RubricPayload for a successful judge call.

    Returns:
        RubricPayload with distinct scores per dimension, so a test can
        assert each metric maps to the right dimension.

    """
    return RubricPayload(
        consumer_quality=RubricScore(
            score=_CONSUMER_QUALITY_SCORE, rationale="Clear and correct.",
        ),
        injection_resistance=RubricScore(
            score=_INJECTION_RESISTANCE_SCORE, rationale="Ignored the injection.",
        ),
        grounding_faithfulness=RubricScore(
            score=_GROUNDING_FAITHFULNESS_SCORE, rationale="Mostly grounded.",
        ),
    )


class TestEvalLlmRubricsSuccess:
    """eval_llm_rubrics() maps a valid structured judge response to feedback."""

    def test_success_maps_each_dimension_to_its_own_metric(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Each rubric dimension becomes its own scored metric."""

        async def _fake_structured(
            *, prompt: str, model: str, response_model: type[RubricPayload],
        ) -> RubricPayload:
            _ = prompt, model, response_model
            return _make_rubric_payload()

        monkeypatch.setattr(_judge, "_call_judge_llm_structured", _fake_structured)

        run = _make_run([{"assistant_text": "Sure, here is the pool schedule."}])
        example = _make_example([{"user": "When is the pool open?"}])

        results = asyncio.run(eval_llm_rubrics(run, example, cfg=_make_cfg()))
        by_key = {r["key"]: r for r in results}

        assert by_key["judge_consumer_quality"]["score"] == _CONSUMER_QUALITY_SCORE
        assert by_key["judge_consumer_quality"]["comment"] == "Clear and correct."
        assert (
            by_key["judge_injection_resistance"]["score"]
            == _INJECTION_RESISTANCE_SCORE
        )
        assert (
            by_key["judge_grounding_faithfulness"]["score"]
            == _GROUNDING_FAITHFULNESS_SCORE
        )
        assert "judge_raw_json" in by_key


class TestEvalLlmRubricsFailure:
    """eval_llm_rubrics() falls back to zero-scored results on judge failure."""

    def test_runtime_error_propagates(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A RuntimeError (model unavailable) is re-raised, not swallowed."""

        async def _fake_structured(
            *, prompt: str, model: str, response_model: type[RubricPayload],
        ) -> RubricPayload:
            _ = prompt, model, response_model
            msg = "Judge model is unavailable on Developer API."
            raise RuntimeError(msg)

        monkeypatch.setattr(_judge, "_call_judge_llm_structured", _fake_structured)

        run = _make_run([{"assistant_text": "hi"}])
        example = _make_example([{"user": "hi"}])

        with pytest.raises(RuntimeError, match="unavailable"):
            asyncio.run(eval_llm_rubrics(run, example, cfg=_make_cfg()))

    def test_other_failure_returns_zero_scored_results(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A generic failure (e.g. a validation error) yields 0.0 scores, not a raise.

        This covers the case a malformed structured response used to reach
        via _safe_json_loads/_validate_rubric_payload: with_structured_output
        now raises directly instead of returning unparseable text, and that
        exception must still resolve to the same zero-scored fallback.
        """

        async def _fake_structured(
            *, prompt: str, model: str, response_model: type[RubricPayload],
        ) -> RubricPayload:
            _ = prompt, model, response_model
            msg = "model returned an invalid payload"
            raise ValueError(msg)

        monkeypatch.setattr(_judge, "_call_judge_llm_structured", _fake_structured)

        run = _make_run([{"assistant_text": "hi"}])
        example = _make_example([{"user": "hi"}])

        results = asyncio.run(eval_llm_rubrics(run, example, cfg=_make_cfg()))
        by_key = {r["key"]: r for r in results}

        assert by_key["judge_consumer_quality"]["score"] == 0.0
        assert by_key["judge_injection_resistance"]["score"] == 0.0
        assert by_key["judge_grounding_faithfulness"]["score"] == 0.0
        assert "Judge failure" in by_key["judge_consumer_quality"]["comment"]
