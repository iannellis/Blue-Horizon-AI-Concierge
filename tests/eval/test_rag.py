"""Tests for eval/evaluators/_rag.py and eval/evaluators/_rag_prompts.py.

Covers the no-match reference sentinel check that gates context-precision
scoring, the custom context-precision prompt's structural compatibility with
Ragas, _rag_score_turn's skip behaviour via stub metric objects, and the retry
policy applied to transient Gemini failures.
"""

# ruff: noqa: S101

import asyncio

import pytest
from google.genai import errors as genai_errors
from ragas.metrics.collections.context_precision.util import (
    ContextPrecisionInput,
    ContextPrecisionOutput,
)
from tenacity import AsyncRetrying

from eval.config import RagasConfig
from eval.evaluators._rag import (
    _build_retrying,
    _is_transient_gemini_error,
    _rag_reference_is_no_match,
    _rag_score_turn,
)
from eval.evaluators._rag_prompts import PartialCoverageContextPrecisionPrompt

SENTINEL = "I could not find exactly what you requested"
EXPECTED_EXAMPLE_COUNT = 4
RETRY_ATTEMPTS = 3


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
# PartialCoverageContextPrecisionPrompt
# ---------------------------------------------------------------------------


def test_custom_prompt_keeps_stock_input_and_output_models() -> None:
    """The override inherits Ragas' models so the judge schema is unchanged."""
    prompt = PartialCoverageContextPrecisionPrompt()
    assert prompt.input_model is ContextPrecisionInput
    assert prompt.output_model is ContextPrecisionOutput


def test_custom_prompt_examples_use_ragas_model_instances() -> None:
    """Examples must be real Ragas models for to_string() to serialize them."""
    prompt = PartialCoverageContextPrecisionPrompt()
    assert len(prompt.examples) == EXPECTED_EXAMPLE_COUNT
    for example_input, example_output in prompt.examples:
        assert isinstance(example_input, ContextPrecisionInput)
        assert isinstance(example_output, ContextPrecisionOutput)
        assert example_output.verdict in {0, 1}


def test_custom_prompt_renders_with_instruction_and_input() -> None:
    """to_string() embeds the overridden instruction and the supplied input."""
    prompt = PartialCoverageContextPrecisionPrompt()
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


# ---------------------------------------------------------------------------
# Transient-error retry
# ---------------------------------------------------------------------------


def _gemini_error(code: int) -> genai_errors.APIError:
    """Build the SDK error Gemini raises for an HTTP status code.

    Args:
        code: HTTP status code.

    Returns:
        A ``ServerError`` for 5xx codes, otherwise a ``ClientError``.

    """
    body = {"error": {"code": code, "message": "x", "status": "UNAVAILABLE"}}
    if code >= 500:  # noqa: PLR2004
        return genai_errors.ServerError(code, body)
    return genai_errors.ClientError(code, body)


def _wrapped(inner: BaseException) -> RuntimeError:
    """Wrap an error the way Instructor does, with ``raise ... from``.

    Args:
        inner: The error to chain as ``__cause__``.

    Returns:
        The outer error, carrying ``inner`` as its cause.

    """
    outer = RuntimeError(str(inner))
    outer.__cause__ = inner
    return outer


def _no_wait_retrying() -> AsyncRetrying:
    """Build the production retry policy with the waits set to zero.

    Returns:
        A policy allowing ``RETRY_ATTEMPTS`` attempts and no sleeping.

    """
    return _build_retrying(
        RagasConfig(
            turns_max=10,
            contexts_max=20,
            context_chars=1200,
            query_chars=1200,
            response_chars=2000,
            reference_chars=2000,
            llm_model="gemini-test",
            llm_max_tokens=2048,
            embedding_model="gemini-embedding-test",
            retry_attempts=RETRY_ATTEMPTS,
            retry_backoff_s=0.0,
            retry_backoff_max_s=0.0,
        ),
    )


class _FailingMetric(_StubMetric):
    """Metric stub that raises a set number of times before succeeding."""

    def __init__(self, value: float, error: BaseException, failures: int) -> None:
        """Initialize the stub.

        Args:
            value: Score returned once the failures are used up.
            error: Exception raised on each failing call.
            failures: Number of calls that raise before one succeeds.

        """
        super().__init__(value)
        self.error = error
        self.failures = failures

    async def ascore(self, **_kwargs: object) -> float:
        """Raise the configured error until the failures are used up.

        Args:
            **_kwargs: Ignored scoring arguments.

        Returns:
            The configured score.

        Raises:
            BaseException: The configured error, on each failing call.

        """
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error
        return self.value


def _score_with(
    answer_relevancy: _StubMetric,
    retrying: AsyncRetrying | None,
) -> dict[str, float | None]:
    """Score an ordinary turn with a chosen answer-relevancy stub.

    Args:
        answer_relevancy: Stub standing in for the answer-relevancy metric.
        retrying: Retry policy passed to ``_rag_score_turn``.

    Returns:
        The turn's scores.

    """
    faithfulness, _, precision, recall = _make_metrics()
    return asyncio.run(
        _rag_score_turn(
            question="What time is check-in?",
            answer="Check-in is at 3:00 PM.",
            contexts=["Check-in time is 3:00 PM."],
            reference="Check-in time is 3:00 PM.",
            metrics=(faithfulness, answer_relevancy, precision, recall),  # type: ignore[arg-type]
            no_match_reference=SENTINEL,
            retrying=retrying,
        ),
    )


@pytest.mark.parametrize("code", [429, 500, 502, 503, 504])
def test_transient_status_codes_are_retryable(code: int) -> None:
    """Rate limiting and server errors are worth another attempt."""
    assert _is_transient_gemini_error(_gemini_error(code)) is True


@pytest.mark.parametrize("code", [400, 401, 403, 404])
def test_client_errors_are_not_retryable(code: int) -> None:
    """A bad request or a denied key would fail the same way again."""
    assert _is_transient_gemini_error(_gemini_error(code)) is False


def test_transient_error_is_found_through_a_wrapping_exception() -> None:
    """Instructor's wrapper exception still counts when its cause is a 503."""
    assert _is_transient_gemini_error(_wrapped(_gemini_error(503))) is True


def test_unrelated_exception_is_not_retryable() -> None:
    """An error with no Gemini error in its chain is not retried."""
    assert _is_transient_gemini_error(ValueError("bad value")) is False


def test_transient_failure_is_retried_until_it_succeeds() -> None:
    """A metric that 503s once is scored on the second attempt."""
    metric = _FailingMetric(0.9, _wrapped(_gemini_error(503)), failures=1)
    scores = _score_with(metric, _no_wait_retrying())
    assert scores["answer_relevancy"] == pytest.approx(0.9)
    assert metric.calls == 2  # noqa: PLR2004


def test_exhausted_retries_raise_the_last_error() -> None:
    """A metric that keeps failing raises after the configured attempts."""
    error = _gemini_error(503)
    metric = _FailingMetric(0.9, error, failures=RETRY_ATTEMPTS)
    with pytest.raises(genai_errors.ServerError):
        _score_with(metric, _no_wait_retrying())
    assert metric.calls == RETRY_ATTEMPTS


def test_non_transient_failure_is_not_retried() -> None:
    """A client error is raised on the first attempt."""
    metric = _FailingMetric(0.9, _gemini_error(400), failures=1)
    with pytest.raises(genai_errors.ClientError):
        _score_with(metric, _no_wait_retrying())
    assert metric.calls == 1


def test_answer_relevancy_failure_is_not_scored_as_zero() -> None:
    """A failed answer-relevancy call raises instead of recording 0.0."""
    metric = _FailingMetric(0.9, _gemini_error(503), failures=1)
    with pytest.raises(genai_errors.ServerError):
        _score_with(metric, None)
