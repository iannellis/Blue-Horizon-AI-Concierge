"""Info reference subset evaluators for Blue Horizon hotel agent.

This module provides evaluators to verify that expected reference snippets
appear in retrieval contexts, ensuring proper information grounding.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from eval._utils import json_value, truncate
from eval.config import load_eval_config

if TYPE_CHECKING:
    from langsmith.schemas import Example, Run


def eval_info_reference_subset(run: Run, example: Example) -> list[dict[str, Any]]:
    """Evaluate whether expected reference snippets appear in retrieval contexts.

    Args:
        run: LangSmith run object containing turn outputs.
        example: LangSmith example object containing dataset turns.

    Returns:
        List of LangSmith feedback dicts for subset reference checks.

    """
    cfg = load_eval_config()
    limits = cfg.evaluator_limits
    max_reference_chars = cfg.ragas.reference_chars
    turn_outputs, example_turns, reference_answers = _rag_extract_turn_inputs(
        run,
        example,
    )
    total_turns = min(len(turn_outputs), len(example_turns))
    scored_turns = 0
    passed_turns = 0
    failures: list[dict[str, Any]] = []

    for idx in range(total_turns):
        example_turn = example_turns[idx]
        reference = _rag_extract_reference(
            example_turn,
            reference_answers,
            idx,
            max_reference_chars,
        )
        if reference is None:
            continue
        expected_snippets = _split_expected_reference(reference)
        if not expected_snippets:
            continue
        turn_output = turn_outputs[idx] if isinstance(turn_outputs[idx], dict) else {}
        candidate_text = _build_reference_candidate_text(turn_output)
        matched, missing_snippets = _reference_subset_match(
            expected_snippets,
            candidate_text,
        )
        scored_turns += 1
        if matched:
            passed_turns += 1
            continue
        if len(failures) < limits.info_filter_failures_max:
            failures.append(
                {
                    "turn_index": idx,
                    "user_snippet": truncate(
                        str(example_turn.get("user", "")),
                        160,
                    ),
                    "expected_snippets": expected_snippets,
                    "missing_snippets": missing_snippets,
                },
            )

    if scored_turns == 0:
        return [
            {
                "key": "info_reference_subset_skipped",
                "score": 1.0,
                "comment": "No turns with reference snippets to score.",
            },
        ]

    pass_rate = passed_turns / scored_turns if scored_turns else 0.0
    raw_failures = json_value(failures)
    if len(raw_failures) > limits.json_value_max:
        failures_entry = {
            "key": "info_reference_subset_failures",
            "value": json_value(failures, max_len=limits.json_value_max),
            "comment": "JSON truncated",
        }
    else:
        failures_entry = {
            "key": "info_reference_subset_failures",
            "value": raw_failures,
        }

    return [
        {
            "key": "info_reference_subset_turns",
            "score": float(scored_turns),
        },
        {
            "key": "info_reference_subset_pass_rate",
            "score": pass_rate,
        },
        failures_entry,
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


def _split_expected_reference(reference: str) -> list[str]:
    """Split a reference string into expected snippet parts.

    Args:
        reference: Reference string containing one or more snippets.

    Returns:
        List of non-empty snippet strings.

    """
    if not reference:
        return []
    return [snippet.strip() for snippet in reference.split(" | ") if snippet.strip()]


def _build_reference_candidate_text(turn_output: dict[str, Any]) -> str:
    """Build a candidate text blob from retrieval contexts and the response.

    Args:
        turn_output: Turn output dict containing contexts and assistant text.

    Returns:
        Combined candidate text used for subset matching.

    """
    contexts = turn_output.get("contexts_used")
    context_list = contexts if isinstance(contexts, list) else []
    context_text = "\n".join([str(item) for item in context_list if item is not None])
    assistant_text = turn_output.get("assistant_text")
    if assistant_text:
        if context_text:
            return f"{context_text}\n{assistant_text}"
        return str(assistant_text)
    return context_text


def _reference_subset_match(
    expected_snippets: list[str],
    candidate_text: str,
) -> tuple[bool, list[str]]:
    """Determine whether all expected snippets appear in candidate text.

    Args:
        expected_snippets: Snippets expected to appear in the candidate text.
        candidate_text: Combined retrieval contexts and assistant response.

    Returns:
        Tuple of (matched, missing_snippets).

    """
    normalized_candidate = _normalize_text(candidate_text)
    missing = []
    for snippet in expected_snippets:
        normalized_snippet = _normalize_text(snippet)
        if normalized_snippet and normalized_snippet not in normalized_candidate:
            missing.append(snippet)
    return len(missing) == 0, missing


def _normalize_text(text: str) -> str:
    """Normalize text for substring matching.

    Args:
        text: Raw input text.

    Returns:
        Lowercased text with collapsed whitespace.

    """
    if not text:
        return ""
    return " ".join(str(text).lower().split())
