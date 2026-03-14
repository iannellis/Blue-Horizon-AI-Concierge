"""Required tool call evaluators for Blue Horizon hotel agent.

This module provides evaluators to verify that required tool calls are present
for rooms turns, ensuring that SQL tools are invoked appropriately.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from eval._utils import json_value, truncate
from eval.evaluators._common import (
    _INFO_REQUIRED_TOOLS,
    _SQL_TOOL_NAMES,
    _get_example_turns,
    _iter_turn_outputs,
)

if TYPE_CHECKING:
    from langsmith.schemas import Example, Run

    from eval.config import EvalConfig, EvaluatorLimitsConfig


def eval_required_tool_calls_present(
    run: Run,
    example: Example,
    *,
    cfg: EvalConfig,
) -> list[dict[str, Any]]:
    """Verify required tool calls were present for rooms turns.

    Args:
        run: LangSmith run object containing turn outputs and tool summaries.
        example: LangSmith example object containing dataset turns.
        cfg: Evaluation configuration for evaluator limits.

    Returns:
        List of LangSmith feedback dicts reporting required tool usage.

    """
    limits = cfg.evaluator_limits
    turn_outputs = _iter_turn_outputs(run)
    example_turns = _get_example_turns(example)
    results = _collect_required_tool_results(
        turn_outputs=turn_outputs,
        example_turns=example_turns,
        limits=limits,
    )
    total_scored = results["total_scored"]
    passing_turns = results["passing_turns"]
    info_missing_counts = results["info_missing_counts"]
    rooms_missing_sql = results["rooms_missing_sql"]
    failures = results["failures"]

    if total_scored == 0:
        return [
            {
                "key": "required_tool_calls_skipped",
                "score": 1.0,
                "comment": "No rooms turns to score.",
            },
        ]

    pass_rate = passing_turns / total_scored if total_scored else 0.0
    missing_counts = {
        "info_missing": info_missing_counts,
        "rooms_missing_sql": rooms_missing_sql,
    }
    raw_missing_counts = json_value(missing_counts)
    if len(raw_missing_counts) > limits.json_value_max:
        missing_entry = {
            "key": "required_tool_calls_missing_counts",
            "value": json_value(missing_counts, max_len=limits.json_value_max),
            "comment": "JSON truncated",
        }
    else:
        missing_entry = {
            "key": "required_tool_calls_missing_counts",
            "value": raw_missing_counts,
        }

    raw_failures = json_value(failures)
    if len(raw_failures) > limits.json_value_max:
        failures_entry = {
            "key": "required_tool_calls_failures",
            "value": json_value(failures, max_len=limits.json_value_max),
            "comment": "JSON truncated",
        }
    else:
        failures_entry = {
            "key": "required_tool_calls_failures",
            "value": raw_failures,
        }
    return [
        {
            "key": "required_tool_calls_pass_rate",
            "score": pass_rate,
        },
        {
            "key": "required_tool_calls_scored_turns",
            "score": float(total_scored),
        },
        missing_entry,
        failures_entry,
    ]


def _collect_required_tool_results(
    *,
    turn_outputs: list[dict[str, Any]],
    example_turns: list[dict[str, Any]],
    limits: EvaluatorLimitsConfig,
) -> dict[str, Any]:
    """Collect aggregated required-tool results over all turns.

    Args:
        turn_outputs: Run turn outputs.
        example_turns: Example turn inputs.
        limits: Evaluator limits for failure capture.

    Returns:
        Dict containing aggregate counts and failure details.

    """
    total_turns = min(len(turn_outputs), len(example_turns))
    total_scored = 0
    passing_turns = 0
    failures: list[dict[str, Any]] = []
    info_missing_counts = dict.fromkeys(_INFO_REQUIRED_TOOLS, 0)
    rooms_missing_sql = 0

    for idx in range(total_turns):
        (
            scored,
            passed,
            missing_info,
            missing_rooms_sql,
            failure_context,
        ) = _score_required_tool_turn(
            {
                "idx": idx,
                "turn_output": turn_outputs[idx],
                "example_turn": example_turns[idx],
            },
        )
        if not scored:
            continue
        total_scored += 1
        if passed:
            passing_turns += 1
        if missing_info:
            for tool_name in missing_info:
                if tool_name in info_missing_counts:
                    info_missing_counts[tool_name] += 1
        if missing_rooms_sql:
            rooms_missing_sql += 1
        if failure_context:
            failure_context["failures"] = failures
            _append_required_tool_failure(failure_context, limits=limits)

    return {
        "total_scored": total_scored,
        "passing_turns": passing_turns,
        "info_missing_counts": info_missing_counts,
        "rooms_missing_sql": rooms_missing_sql,
        "failures": failures,
    }


def _score_required_tool_turn(
    turn_context: dict[str, Any],
) -> tuple[bool, bool, list[str], bool, dict[str, Any] | None]:
    """Score a single rooms turn for required tool usage.

    Args:
        turn_context: Dict containing example and run turn data.

    Returns:
        Tuple of (scored, passed, missing_info_tools, missing_rooms_sql,
        failure_context or None).

    """
    idx = turn_context.get("idx")
    turn_index = idx if isinstance(idx, int) else -1
    turn_output = turn_context.get("turn_output")
    output = turn_output if isinstance(turn_output, dict) else {}
    example_turn = turn_context.get("example_turn")
    example = example_turn if isinstance(example_turn, dict) else {}
    route_pred = output.get("route_pred")
    if route_pred not in {"info", "rooms"}:
        return False, False, [], False, None

    tool_summary = output.get("tool_summary")
    called_tools = _tools_called(tool_summary)
    if route_pred == "info":
        return False, False, [], False, None

    if called_tools.intersection(_SQL_TOOL_NAMES):
        return True, True, [], False, None
    return (
        True,
        False,
        [],
        True,
        _required_tool_failure_context(
            idx=turn_index,
            route_pred="rooms",
            user_text=example.get("user"),
            missing_tools=["run_sql"],
            called_tools=called_tools,
        ),
    )


def _required_tool_failure_context(
    *,
    idx: int,
    route_pred: str,
    user_text: object,
    missing_tools: list[str],
    called_tools: set[str],
) -> dict[str, Any]:
    """Build failure context for required tool checks.

    Args:
        idx: Turn index within the run/example.
        route_pred: Route prediction string.
        user_text: Raw user text for snippet extraction.
        missing_tools: List of missing tool names.
        called_tools: Set of called tool names.

    Returns:
        Dict payload for failure reporting.

    """
    return {
        "idx": idx,
        "route_pred": route_pred,
        "user_text": user_text,
        "missing_tools": missing_tools,
        "called_tools": called_tools,
    }


def _tools_called(tool_summary: object) -> set[str]:
    """Extract tool names from a tool summary list.

    Args:
        tool_summary: Tool summary list from run outputs.

    Returns:
        Set of tool name strings found in the summary.

    """
    if not isinstance(tool_summary, list):
        return set()
    tools: set[str] = set()
    for entry in tool_summary:
        if not isinstance(entry, dict):
            continue
        tool_name = entry.get("tool")
        if isinstance(tool_name, str) and tool_name:
            tools.add(tool_name)
    return tools


def _append_required_tool_failure(
    failure_context: dict[str, Any],
    *,
    limits: EvaluatorLimitsConfig,
) -> None:
    """Append a required-tool failure record, respecting the cap.

    Args:
        failure_context: Dict containing failure details for reporting.
        limits: Evaluator limits for failure capture cap.

    """
    failures = failure_context.get("failures")
    if not isinstance(failures, list):
        return
    if len(failures) >= limits.required_tool_failures_max:
        return

    idx = failure_context.get("idx")
    route_pred = failure_context.get("route_pred")
    user_text = failure_context.get("user_text")
    missing_tools = failure_context.get("missing_tools")
    called_tools = failure_context.get("called_tools")

    turn_index = idx if isinstance(idx, int) else -1
    route_value = route_pred if isinstance(route_pred, str) else ""
    missing_list = missing_tools if isinstance(missing_tools, list) else []
    called_set = called_tools if isinstance(called_tools, set) else set()

    failures.append(
        {
            "turn_index": turn_index,
            "route_pred": route_value,
            "user_snippet": truncate(str(user_text or ""), 140),
            "missing_tools": missing_list,
            "called_tools": sorted(called_set)[:12],
        },
    )
