"""LangSmith evaluators for Blue Horizon LangGraph hotel agent runs.

This package provides deterministic evaluators for routing accuracy, injection
tripwire detection, expected filters validation, reference subset matching,
required tool calls presence, rooms tool outcomes with database invariant checks,
LLM-as-judge rubric evaluation, and RAG metrics scoring.

The evaluators are designed to emit LangSmith-compatible dicts and work with
the compact run outputs produced by the evaluation target in eval/langsmith_target.
"""

from __future__ import annotations

from eval.evaluators._info_filters import eval_info_expected_filters
from eval.evaluators._info_references import eval_info_reference_subset
from eval.evaluators._injection import eval_injection_tripwires
from eval.evaluators._judge import eval_llm_rubrics
from eval.evaluators._rag import eval_rag_metrics_info_turns
from eval.evaluators._required_tools import eval_required_tool_calls_present
from eval.evaluators._rooms import eval_rooms_outcome_and_invariants
from eval.evaluators._routing import eval_routing_accuracy

__all__ = [
    "eval_info_expected_filters",
    "eval_info_reference_subset",
    "eval_injection_tripwires",
    "eval_llm_rubrics",
    "eval_rag_metrics_info_turns",
    "eval_required_tool_calls_present",
    "eval_rooms_outcome_and_invariants",
    "eval_routing_accuracy",
]
