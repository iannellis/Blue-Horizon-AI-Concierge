"""RAG metrics evaluators for Blue Horizon hotel agent.

This module provides Ragas-based evaluation metrics (faithfulness, answer
relevancy, context precision, and context recall) for information retrieval
turns.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel
from ragas.embeddings.base import (
    BaseRagasEmbedding,
    BaseRagasEmbeddings,
    embedding_factory,
)
from ragas.llms.base import InstructorBaseRagasLLM, llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from eval._utils import json_value
from eval.config import load_eval_config

if TYPE_CHECKING:
    from langsmith.schemas import Example, Run

try:  # Optional dependency for Gemini clients used by Ragas.
    from google import genai as _genai

    _GENAI_IMPORT_ERROR: Exception | None = None
except Exception as _exc:  # noqa: BLE001
    _genai = None
    _GENAI_IMPORT_ERROR = _exc

_RAGAS_LOCK = asyncio.Lock()
_RAGAS_METRICS: (
    tuple[
        Faithfulness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
    ]
    | None
) = None

InstructorTypeVar = TypeVar("InstructorTypeVar", bound=BaseModel)

logger = logging.getLogger(__name__)


@runtime_checkable
class _MetricResultLike(Protocol):
    """Protocol for Ragas MetricResult-like objects."""

    @property
    def value(self) -> object:
        """Return the underlying metric value."""

    def __float__(self) -> float:
        """Return a float representation of the metric."""
        raise NotImplementedError


class AsyncFromSyncInstructorLLM(InstructorBaseRagasLLM):
    """Wrap a sync Instructor LLM to provide async calls via threads.

    This keeps the google-genai client flow while satisfying Ragas collection
    metrics that require an async-capable InstructorBaseRagasLLM.

    """

    _sync_llm: InstructorBaseRagasLLM

    def __init__(self, *, sync_llm: InstructorBaseRagasLLM) -> None:
        """Initialize the wrapper with a sync Instructor LLM.

        Args:
            sync_llm: Synchronous Instructor LLM instance.

        """
        self._sync_llm = sync_llm

    def generate(
        self,
        prompt: str,
        response_model: type[InstructorTypeVar],
    ) -> InstructorTypeVar:
        """Generate a response using the wrapped sync LLM.

        Args:
            prompt: Prompt to send to the LLM.
            response_model: Pydantic model used for structured output.

        Returns:
            Parsed structured response from the LLM.

        """
        return self._sync_llm.generate(prompt, response_model)

    async def agenerate(
        self,
        prompt: str,
        response_model: type[InstructorTypeVar],
    ) -> InstructorTypeVar:
        """Generate a response asynchronously using a worker thread.

        Args:
            prompt: Prompt to send to the LLM.
            response_model: Pydantic model used for structured output.

        Returns:
            Parsed structured response from the LLM.

        """
        return await asyncio.to_thread(
            self._sync_llm.generate,
            prompt,
            response_model,
        )


async def eval_rag_metrics_info_turns(
    run: Run,
    example: Example,
) -> list[dict[str, Any]]:
    """Compute Ragas metrics for information turns in a run.

    Args:
        run: LangSmith run object containing turn outputs.
        example: LangSmith example object containing dataset turns.

    Returns:
        List of LangSmith feedback dicts with per-turn Ragas scores and means.

    """
    cfg = load_eval_config()
    ragas_cfg = cfg.ragas
    limits = cfg.evaluator_limits
    max_turns = ragas_cfg.turns_max
    max_contexts = ragas_cfg.contexts_max
    max_context_chars = ragas_cfg.context_chars
    max_query_chars = ragas_cfg.query_chars
    max_response_chars = ragas_cfg.response_chars
    max_reference_chars = ragas_cfg.reference_chars

    turn_outputs, example_turns, reference_answers = _rag_extract_turn_inputs(
        run,
        example,
    )
    total_turns = min(len(turn_outputs), len(example_turns), max_turns)
    if total_turns == 0:
        return [
            {
                "key": "rag_metrics_skipped",
                "score": 1.0,
                "comment": "No info/RAG turns in run",
            },
        ]

    metrics = await _get_ragas_metrics()

    per_turn: list[dict[str, Any]] = []
    faithfulness_scores: list[float] = []
    answer_relevancy_scores: list[float] = []
    context_precision_scores: list[float] = []
    context_recall_scores: list[float] = []

    for idx in range(total_turns):
        turn_output = turn_outputs[idx]
        if not _rag_is_eligible_turn(turn_output):
            continue
        question = _rag_truncate_text(
            example_turns[idx].get("user"),
            max_query_chars,
        )
        answer = _rag_truncate_text(
            turn_output.get("assistant_text"),
            max_response_chars,
        )
        contexts = _rag_prepare_contexts(
            turn_output.get("contexts_used"),
            max_contexts,
            max_context_chars,
        )
        reference = _rag_extract_reference(
            example_turns[idx],
            reference_answers,
            idx,
            max_reference_chars,
        )

        turn_scores = await _rag_score_turn(
            question=question,
            answer=answer,
            contexts=contexts,
            reference=reference,
            metrics=metrics,
        )
        per_turn.append(
            {
                "turn_index": idx,
                "faithfulness": turn_scores["faithfulness"],
                "answer_relevancy": turn_scores["answer_relevancy"],
                "context_precision": turn_scores["context_precision"],
                "context_recall": turn_scores["context_recall"],
                "has_reference": reference is not None,
            },
        )
        faithfulness = turn_scores["faithfulness"]
        answer_relevancy = turn_scores["answer_relevancy"]
        if faithfulness is None or answer_relevancy is None:
            continue
        faithfulness_scores.append(float(faithfulness))
        answer_relevancy_scores.append(float(answer_relevancy))
        context_precision = turn_scores["context_precision"]
        if context_precision is not None:
            context_precision_scores.append(float(context_precision))
        context_recall = turn_scores["context_recall"]
        if context_recall is not None:
            context_recall_scores.append(float(context_recall))

    if not per_turn:
        return [
            {
                "key": "rag_metrics_skipped",
                "score": 1.0,
                "comment": "No info/RAG turns in run",
            },
        ]

    precision_mean, precision_comment = _rag_mean_or_comment(
        context_precision_scores,
        "No references",
    )
    recall_mean, recall_comment = _rag_mean_or_comment(
        context_recall_scores,
        "No references",
    )

    raw_per_turn = json_value(per_turn)
    if len(raw_per_turn) > limits.rag_per_turn_json_max:
        per_turn_entry = {
            "key": "rag_per_turn",
            "value": json_value(per_turn, max_len=limits.rag_per_turn_json_max),
            "comment": "JSON truncated",
        }
    else:
        per_turn_entry = {"key": "rag_per_turn", "value": raw_per_turn}

    return [
        {
            "key": "rag_faithfulness_mean",
            "score": _rag_mean(faithfulness_scores),
        },
        {
            "key": "rag_answer_relevancy_mean",
            "score": _rag_mean(answer_relevancy_scores),
        },
        {
            "key": "rag_context_precision_mean",
            "score": precision_mean,
            "comment": precision_comment,
        },
        {
            "key": "rag_context_recall_mean",
            "score": recall_mean,
            "comment": recall_comment,
        },
        per_turn_entry,
    ]


def _rag_extract_turn_inputs(
    run: Run,
    example: Example,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[object]]:
    """Extract turn data from run outputs and example inputs.

    Args:
        run: LangSmith run object containing turn outputs.
        example: LangSmith example object containing dataset turns.

    Returns:
        Tuple of (turn_outputs, example_turns, reference_answers).

    """
    outputs = run.outputs or {}
    turn_outputs_raw = outputs.get("turn_outputs") or []
    turn_outputs = [t for t in turn_outputs_raw if isinstance(t, dict)]

    inputs = example.inputs or {}
    example_turns_raw = inputs.get("turns") or []
    example_turns = [t for t in example_turns_raw if isinstance(t, dict)]

    reference_answers = inputs.get("reference_answers") or []
    if not isinstance(reference_answers, list):
        reference_answers = []

    return turn_outputs, example_turns, reference_answers


async def _get_ragas_metrics() -> tuple[
    Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall,
]:
    """Lazily initialize and return Ragas metrics with configured models.

    Returns:
        Tuple of metric instances (faithfulness, answer_relevancy,
        context_precision, context_recall).

    """
    global _RAGAS_METRICS  # noqa: PLW0603
    if _RAGAS_METRICS is not None:
        return _RAGAS_METRICS

    async with _RAGAS_LOCK:
        if _RAGAS_METRICS is not None:
            return _RAGAS_METRICS
        _ensure_google_api_key_for_ragas()
        ragas_llm = _build_ragas_llm()
        ragas_embeddings = _ensure_base_embedding(_build_ragas_embeddings())
        _RAGAS_METRICS = (
            Faithfulness(llm=ragas_llm),
            AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
            ContextPrecision(llm=ragas_llm),
            ContextRecall(llm=ragas_llm),
        )
        return _RAGAS_METRICS


def _ensure_google_api_key_for_ragas() -> None:
    """Ensure GOOGLE_API_KEY is set for Ragas Gemini providers.

    This mirrors GEMINI_API_KEY into GOOGLE_API_KEY when needed, without
    changing environment variables if GOOGLE_API_KEY is already present.

    """
    if os.getenv("GOOGLE_API_KEY"):
        return
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        os.environ["GOOGLE_API_KEY"] = gemini_key


def _build_ragas_llm() -> InstructorBaseRagasLLM:
    """Build the Ragas LLM configured for Gemini.

    Returns:
        Ragas LLM instance configured for the Gemini provider.

    """
    client = _get_google_genai_client()
    ragas_cfg = load_eval_config().ragas
    sync_llm = llm_factory(
        model=ragas_cfg.llm_model,
        provider="google",
        client=client,
        max_tokens=ragas_cfg.llm_max_tokens,
    )
    return AsyncFromSyncInstructorLLM(sync_llm=sync_llm)


def _build_ragas_embeddings() -> BaseRagasEmbeddings | BaseRagasEmbedding:
    """Build the Ragas embeddings configured for Gemini.

    Returns:
        Ragas embeddings instance configured for the Gemini provider.

    """
    client = _get_google_genai_client()
    ragas_cfg = load_eval_config().ragas
    return embedding_factory(
        provider="google",
        model=ragas_cfg.embedding_model,
        client=client,
    )


def _get_google_genai_client() -> object:
    """Create a Google GenAI client using environment-based auth.

    Returns:
        Initialized Google GenAI client.

    """
    if _genai is None:
        msg = "google-genai SDK is required for Ragas Gemini models."
        raise RuntimeError(msg) from _GENAI_IMPORT_ERROR
    return _genai.Client()


def _ensure_base_embedding(
    embeddings: BaseRagasEmbeddings | BaseRagasEmbedding,
) -> BaseRagasEmbedding:
    """Ensure the Ragas embeddings object matches BaseRagasEmbedding.

    Args:
        embeddings: Embeddings instance returned by the factory.

    Returns:
        Embeddings instance narrowed to BaseRagasEmbedding.

    Raises:
        RuntimeError: If the embeddings type is incompatible with metrics.

    """
    if isinstance(embeddings, BaseRagasEmbedding):
        return embeddings
    msg = "Ragas embeddings must be BaseRagasEmbedding for AnswerRelevancy."
    raise RuntimeError(msg)


def _rag_is_eligible_turn(turn_output: dict[str, object]) -> bool:
    """Determine whether a turn should be scored by Ragas.

    Args:
        turn_output: Turn output dict from the run outputs.

    Returns:
        True only when the agent routed the turn to the info path.

    """
    route_pred = turn_output.get("route_pred")
    return isinstance(route_pred, str) and route_pred == "info"


def _rag_prepare_contexts(
    contexts: object,
    max_contexts: int,
    max_chars: int,
) -> list[str]:
    """Prepare retrieved contexts for Ragas scoring.

    Args:
        contexts: Contexts list from the run outputs.
        max_contexts: Maximum number of contexts to include.
        max_chars: Maximum length per context string.

    Returns:
        List of cleaned and truncated context strings.

    """
    if not isinstance(contexts, list) or max_contexts <= 0:
        return []
    trimmed: list[str] = []
    for item in contexts[:max_contexts]:
        if item is None:
            continue
        text = _rag_truncate_text(item, max_chars)
        if text:
            trimmed.append(text)
    return trimmed


def _rag_extract_reference(
    example_turn: dict[str, object],
    reference_answers: list[object],
    index: int,
    max_chars: int,
) -> str | None:
    """Extract a reference answer for a turn if available.

    Args:
        example_turn: Example turn dict containing potential reference fields.
        reference_answers: Optional list of reference answers from example inputs.
        index: Turn index used to lookup reference list entries.
        max_chars: Maximum length for the reference string.

    Returns:
        Truncated reference string if available, otherwise None.

    """
    reference = example_turn.get("reference")
    if reference is None:
        reference = example_turn.get("expected_answer")
    if reference is None:
        reference = example_turn.get("ground_truth")
    if reference is None and index < len(reference_answers):
        reference = reference_answers[index]
    if reference is None:
        return None
    text = _rag_truncate_text(reference, max_chars)
    return text if text else None


def _rag_truncate_text(value: object, limit: int) -> str:
    """Convert a value to a truncated string.

    Args:
        value: Value to coerce into a string.
        limit: Maximum character length.

    Returns:
        Truncated string value.

    """
    if limit <= 0:
        return ""
    if value is None:
        return ""
    text = str(value)
    if len(text) <= limit:
        return text
    if limit <= 3:  # noqa: PLR2004
        return text[:limit]
    return f"{text[: limit - 3]}..."


async def _rag_score_turn(
    *,
    question: str,
    answer: str,
    contexts: list[str],
    reference: str | None,
    metrics: tuple[Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall],
) -> dict[str, float | None]:
    """Score a single turn using Ragas metrics.

    Args:
        question: User query string.
        answer: Assistant response string.
        contexts: Retrieved context strings.
        reference: Reference answer string when available.
        metrics: Tuple of Ragas metrics in the order
            (faithfulness, answer_relevancy, context_precision, context_recall).

    Returns:
        Dict with metric scores for the turn.

    """
    faithfulness, answer_relevancy, context_precision, context_recall = metrics

    faithfulness_score = await faithfulness.ascore(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
    )

    try:
        answer_relevancy_score = await answer_relevancy.ascore(
            user_input=question,
            response=answer,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "AnswerRelevancy metric failed: %s. Defaulting to 0.0",
            exc,
        )
        answer_relevancy_score = 0.0

    context_precision_score: float | None = None
    context_recall_score: float | None = None
    if reference:
        context_precision_score = _metric_result_to_float(
            await context_precision.ascore(
                user_input=question,
                reference=reference,
                retrieved_contexts=contexts,
            ),
        )
        context_recall_score = _metric_result_to_float(
            await context_recall.ascore(
                user_input=question,
                reference=reference,
                retrieved_contexts=contexts,
            ),
        )
    return {
        "faithfulness": _metric_result_to_float(faithfulness_score),
        "answer_relevancy": _metric_result_to_float(answer_relevancy_score),
        "context_precision": context_precision_score,
        "context_recall": context_recall_score,
    }


def _metric_result_to_float(value: object) -> float:
    """Coerce a metric result to float.

    Args:
        value: Metric result or numeric value returned by Ragas.

    Returns:
        Float representation of the metric result.

    """
    result: float
    if isinstance(value, bool):
        result = 1.0 if value else 0.0
    elif isinstance(value, (int, float)):
        result = float(value)
    elif isinstance(value, _MetricResultLike):
        try:
            raw_value = value.value
            if isinstance(raw_value, (int, float, str)):
                result = float(raw_value)
            else:
                result = _fallback_float(value)
        except (TypeError, ValueError):
            result = _fallback_float(value)
    else:
        result = _fallback_float(value)
    return result


def _fallback_float(value: object) -> float:
    """Coerce an arbitrary value into a float with string fallback.

    Args:
        value: Value to coerce.

    Returns:
        Float value, defaulting to 0.0 when coercion fails.

    """
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        try:
            return float(str(value))
        except (TypeError, ValueError):
            return 0.0


def _rag_mean_or_comment(
    values: list[float],
    empty_comment: str,
) -> tuple[float, str | None]:
    """Compute a mean score with an optional comment for empty lists.

    Args:
        values: List of float values to average.
        empty_comment: Comment to return when the list is empty.

    Returns:
        Tuple of (mean, comment) where comment is None when values exist.

    """
    if not values:
        return 0.0, empty_comment
    return _rag_mean(values), None


def _rag_mean(values: list[float]) -> float:
    """Compute the mean of a list of floats.

    Args:
        values: List of float values.

    Returns:
        Mean of the values, or 0.0 when empty.

    """
    if not values:
        return 0.0
    return sum(values) / len(values)
