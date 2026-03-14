"""Info reference subset evaluators for Blue Horizon hotel agent.

This module provides evaluators to verify that expected reference snippets
appear in retrieval contexts, ensuring proper information grounding.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from eval._utils import json_value, truncate
from eval.evaluators._common import _rag_extract_reference, _rag_extract_turn_inputs

if TYPE_CHECKING:
    from langsmith.schemas import Example, Run

    from eval.config import EvalConfig


def eval_info_reference_subset(
    run: Run,
    example: Example,
    *,
    cfg: EvalConfig,
) -> list[dict[str, Any]]:
    """Evaluate whether expected reference snippets appear in retrieval contexts.

    Args:
        run: LangSmith run object containing turn outputs.
        example: LangSmith example object containing dataset turns.
        cfg: Evaluation configuration for limits and Ragas parameters.

    Returns:
        List of LangSmith feedback dicts for subset reference checks.

    """
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
