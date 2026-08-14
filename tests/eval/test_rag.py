"""Tests for eval/evaluators/_rag.py and eval/evaluators/_rag_prompts.py.

Covers the no-match reference sentinel check that gates context-precision
scoring, the custom context-precision prompt's structural compatibility with
Ragas, and _rag_score_turn's skip behaviour via stub metric objects.
"""

# ruff: noqa: S101

import asyncio

import pytest
from ragas.metrics.collections.context_precision.util import (
    ContextPrecisionInput,
    ContextPrecisionOutput,
)

from eval.evaluators._rag import _rag_reference_is_no_match, _rag_score_turn
from eval.evaluators._rag_prompts import HotelContextPrecisionPrompt

SENTINEL = "I could not find exactly what you requested"
EXPECTED_EXAMPLE_COUNT = 3


# ---------------------------------------------------------------------------
# _rag_reference_is_no_match
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reference",
    [
        SENTINEL,
        f"  {SENTINEL}  ",
        SENTINEL.upper(),
        f"{SENTINEL}.",
        f"{SENTINEL} | {SENTINEL}",
    ],
)
def test_pure_no_match_references_are_detected(reference: str) -> None:
    """References consisting only of the sentinel are flagged as no-match."""
    assert _rag_reference_is_no_match(reference, SENTINEL) is True


@pytest.mark.parametrize(
    "reference",
    [
        f"{SENTINEL} | Yes, breakfast is included with most room rates.",
        f"We offer both self-parking and valet parking. | {SENTINEL}",
        "Check-in time is 3:00 PM and check-out time is 11:00 AM local time.",
        "",
    ],
)
def test_mixed_and_ordinary_references_are_not_no_match(reference: str) -> None:
    """A sentinel paired with real content stays eligible for scoring."""
    assert _rag_reference_is_no_match(reference, SENTINEL) is False


def test_empty_sentinel_disables_the_check() -> None:
    """An empty sentinel never classifies a reference as no-match."""
    assert _rag_reference_is_no_match(SENTINEL, "") is False


# ---------------------------------------------------------------------------
# HotelContextPrecisionPrompt
# ---------------------------------------------------------------------------


def test_custom_prompt_keeps_stock_input_and_output_models() -> None:
    """The override inherits Ragas' models so the judge schema is unchanged."""
    prompt = HotelContextPrecisionPrompt()
    assert prompt.input_model is ContextPrecisionInput
    assert prompt.output_model is ContextPrecisionOutput


def test_custom_prompt_examples_use_ragas_model_instances() -> None:
    """Examples must be real Ragas models for to_string() to serialize them."""
    prompt = HotelContextPrecisionPrompt()
    assert len(prompt.examples) == EXPECTED_EXAMPLE_COUNT
    for example_input, example_output in prompt.examples:
        assert isinstance(example_input, ContextPrecisionInput)
        assert isinstance(example_output, ContextPrecisionOutput)
        assert example_output.verdict in {0, 1}


def test_custom_prompt_renders_with_instruction_and_input() -> None:
    """to_string() embeds the overridden instruction and the supplied input."""
    prompt = HotelContextPrecisionPrompt()
    rendered = prompt.to_string(
        ContextPrecisionInput(
            question="Do you have a pool, and what time is checkout?",
            context="The rooftop pool is open daily from 7:00 AM to 10:00 PM.",
            answer="Yes, there is a rooftop pool. Checkout is at 11:00 AM.",
        ),
    )
    assert "useful if it supports at least one part of the question" in rendered
    assert "rooftop pool is open daily" in rendered
    assert "verdict" in rendered


# ---------------------------------------------------------------------------
# _rag_score_turn context-precision gating
# ---------------------------------------------------------------------------


class _StubMetric:
    """Metric stub recording whether it was scored."""

    def __init__(self, value: float) -> None:
        """Initialize the stub with the score it should return.

        Args:
            value: Score returned from ascore().

        """
        self.value = value
        self.calls = 0

    async def ascore(self, **_kwargs: object) -> float:
        """Return the configured score and count the invocation.

        Args:
            **_kwargs: Ignored scoring arguments.

        Returns:
            The configured score.

        """
        self.calls += 1
        return self.value


def _make_metrics() -> tuple[_StubMetric, _StubMetric, _StubMetric, _StubMetric]:
    """Build a metrics tuple of stubs in Ragas' expected order.

    Returns:
        Tuple of (faithfulness, answer_relevancy, context_precision,
        context_recall) stubs.

    """
    return (_StubMetric(1.0), _StubMetric(0.9), _StubMetric(0.8), _StubMetric(0.7))


def test_no_match_reference_skips_context_precision_only() -> None:
    """Precision is skipped for a pure no-match turn; recall still runs."""
    metrics = _make_metrics()
    scores = asyncio.run(
        _rag_score_turn(
            question="I need a 15-minute dining option under $10.",
            answer="I could not find exactly what you requested.",
            contexts=["Room service is available 24/7."],
            reference=SENTINEL,
            metrics=metrics,  # type: ignore[arg-type]
            no_match_reference=SENTINEL,
        ),
    )
    assert scores["context_precision"] is None
    assert metrics[2].calls == 0
    assert scores["context_recall"] == pytest.approx(0.7)
    assert metrics[3].calls == 1


def test_mixed_reference_still_scores_context_precision() -> None:
    """A sentinel alongside real content keeps precision scoring enabled."""
    metrics = _make_metrics()
    scores = asyncio.run(
        _rag_score_turn(
            question="Any 15-minute dining under $10, and is breakfast included?",
            answer="Nothing matched, but breakfast is included with most rates.",
            contexts=["Breakfast is included with most room rates."],
            reference=f"{SENTINEL} | Yes, breakfast is included with most room rates.",
            metrics=metrics,  # type: ignore[arg-type]
            no_match_reference=SENTINEL,
        ),
    )
    assert scores["context_precision"] == pytest.approx(0.8)
    assert metrics[2].calls == 1


def test_ordinary_reference_scores_all_metrics() -> None:
    """A normal turn scores every metric, including context precision."""
    metrics = _make_metrics()
    scores = asyncio.run(
        _rag_score_turn(
            question="What time is check-in?",
            answer="Check-in is at 3:00 PM.",
            contexts=["Check-in time is 3:00 PM."],
            reference="Check-in time is 3:00 PM and check-out is 11:00 AM.",
            metrics=metrics,  # type: ignore[arg-type]
            no_match_reference=SENTINEL,
        ),
    )
    assert scores["faithfulness"] == pytest.approx(1.0)
    assert scores["answer_relevancy"] == pytest.approx(0.9)
    assert scores["context_precision"] == pytest.approx(0.8)
    assert scores["context_recall"] == pytest.approx(0.7)
