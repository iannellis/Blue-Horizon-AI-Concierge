"""Info expected filters evaluators for Blue Horizon hotel agent.

This module provides evaluators to check that info tool calls use expected
filters correctly, ensuring that amenity and service queries include proper
filtering parameters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from eval._utils import json_value, truncate
from eval.config import load_eval_config
from eval.evaluators._common import _get_example_turns, _iter_turn_outputs

if TYPE_CHECKING:
    from langsmith.schemas import Example, Run


def eval_info_expected_filters(run: Run, example: Example) -> list[dict[str, Any]]:
    """Evaluate expected info-tool filters against logged tool usage.

    Args:
        run: LangSmith run object containing turn outputs.
        example: LangSmith example object containing dataset turns.

    Returns:
        List of LangSmith feedback dicts for expected filter checks.

    """
    turn_outputs = _iter_turn_outputs(run)
    example_turns = _get_example_turns(example)
    total_turns = min(len(turn_outputs), len(example_turns))
    labeled_turns = 0
    total_checks = 0
    passed_checks = 0
    failures: list[dict[str, Any]] = []
    divergence_checks = 0
    divergence_failures = 0
    limits = load_eval_config().evaluator_limits

    for idx in range(total_turns):
        example_turn = example_turns[idx]
        if "expected_filters" not in example_turn:
            continue
        labeled_turns += 1
        turn_output = turn_outputs[idx] if isinstance(turn_outputs[idx], dict) else {}
        (
            check_count,
            pass_count,
            divergence_count,
            divergence_failed,
            failure,
        ) = _evaluate_expected_filters_turn(
            idx=idx,
            example_turn=example_turn,
            turn_output=turn_output,
        )
        total_checks += check_count
        passed_checks += pass_count
        divergence_checks += divergence_count
        divergence_failures += divergence_failed
        if failure and len(failures) < limits.info_filter_failures_max:
            failures.append(failure)

    if labeled_turns == 0:
        return [
            {
                "key": "info_expected_filters_skipped",
                "score": 1.0,
                "comment": "No turns with expected_filters in dataset.",
            },
        ]

    pass_rate = passed_checks / total_checks if total_checks else 0.0
    if divergence_checks:
        # Convergence rate: 1.0 = all filters converge, 0.0 = all diverge
        convergence_rate = 1.0 - (divergence_failures / divergence_checks)
        convergence_comment = None
    else:
        convergence_rate = 1.0
        convergence_comment = "No convergence checks performed."

    raw_failures = json_value(failures)
    if len(raw_failures) > limits.json_value_max:
        failures_value = json_value(failures, max_len=limits.json_value_max)
        failures_entry = {
            "key": "info_expected_filters_failures",
            "value": failures_value,
            "comment": "JSON truncated",
        }
    else:
        failures_entry = {
            "key": "info_expected_filters_failures",
            "value": raw_failures,
        }

    return [
        {
            "key": "info_expected_filters_turns",
            "score": float(labeled_turns),
        },
        {
            "key": "info_expected_filters_pass_rate",
            "score": pass_rate,
        },
        {
            "key": "info_expected_filters_convergence_rate",
            "score": convergence_rate,
            "comment": convergence_comment,
        },
        failures_entry,
    ]


def _evaluate_expected_filters_turn(
    *,
    idx: int,
    example_turn: dict[str, Any],
    turn_output: dict[str, Any],
) -> tuple[int, int, int, int, dict[str, Any] | None]:
    """Evaluate expected filters for a single labeled turn.

    Args:
        idx: Turn index within the run/example.
        example_turn: Dataset turn containing expected filters.
        turn_output: Run output for the aligned turn.

    Returns:
        Tuple of (total_checks, passed_checks, divergence_checks,
        divergence_failures, failure_entry or None).

    """
    expected_full = example_turn.get("expected_filters")
    expected = (
        _canon_expected_filters(expected_full)
        if isinstance(expected_full, dict)
        else {}
    )
    tool_summary = turn_output.get("tool_summary")
    amenity_filters, amenity_unknown = _extract_filters_for_tool(
        tool_summary,
        "query_amenities",
    )
    service_filters, service_unknown = _extract_filters_for_tool(
        tool_summary,
        "query_services",
    )

    (
        check_count,
        pass_count,
        divergence_count,
        divergence_failed,
        failed_checks,
    ) = _score_expected_filters(
        {
            "expected": expected,
            "amenity_filters": amenity_filters,
            "service_filters": service_filters,
            "amenity_unknown": amenity_unknown,
            "service_unknown": service_unknown,
        },
    )

    failure = None
    if failed_checks:
        failure = _build_info_expected_filters_failure(
            {
                "idx": idx,
                "example_turn": example_turn,
                "turn_output": turn_output,
                "expected": expected,
                "amenity_filters": amenity_filters,
                "service_filters": service_filters,
                "amenity_unknown": amenity_unknown,
                "service_unknown": service_unknown,
                "failed_checks": failed_checks,
            },
        )

    return check_count, pass_count, divergence_count, divergence_failed, failure


def _canon_expected_filters(raw_filters: dict[str, Any]) -> dict[str, Any]:
    """Canonicalize expected filters by removing None and coercing types.

    Args:
        raw_filters: Expected filters dict with canonical keys and optional values.

    Returns:
        Dict containing only keys with non-null values and coerced types.

    """
    canonical_keys = (
        "booking_required",
        "min_price",
        "max_price",
        "min_notice_hours",
        "max_notice_hours",
        "min_duration_minutes",
        "max_duration_minutes",
    )
    output: dict[str, Any] = {}
    for key in canonical_keys:
        if key not in raw_filters:
            continue
        value = raw_filters.get(key)
        if value is None:
            continue
        if key == "booking_required":
            output[key] = bool(value)
            continue
        if key in ("min_price", "max_price"):
            try:
                output[key] = float(value)
            except (TypeError, ValueError):
                output[key] = value
            continue
        try:
            output[key] = int(value)
        except (TypeError, ValueError):
            output[key] = value
    return output


def _extract_filters_for_tool(
    tool_summary: list[dict[str, Any]] | None,
    tool_name: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Extract normalized filters and unknown keys for a tool from summaries.

    Args:
        tool_summary: Tool summary list from run outputs.
        tool_name: Tool name to search for.

    Returns:
        Tuple of (filters_norm, unknown_keys) for the tool.

    """
    if not isinstance(tool_summary, list):
        return None, []
    for entry in reversed(tool_summary):
        if not isinstance(entry, dict):
            continue
        if entry.get("tool") != tool_name:
            continue
        filters_norm = entry.get("filters_norm")
        filters_value = filters_norm if isinstance(filters_norm, dict) else None
        unknown_keys = entry.get("filters_unknown_keys", [])
        unknown_list = unknown_keys if isinstance(unknown_keys, list) else []
        return filters_value, unknown_list
    return None, []


def _score_expected_filters(
    filter_context: dict[str, Any],
) -> tuple[int, int, int, int, list[str]]:
    """Score expected filters against amenities/services tool outputs.

    Args:
        filter_context: Dict containing expected filters, actual filters, and
            unknown keys.

    Returns:
        Tuple of (total_checks, passed_checks, divergence_checks,
        divergence_failures, failed_checks).

    """
    expected = filter_context.get("expected")
    if not isinstance(expected, dict):
        expected = {}
    if expected:
        return _score_expected_filters_present(filter_context)
    return _score_expected_filters_empty(filter_context)


def _score_expected_filters_present(
    filter_context: dict[str, Any],
) -> tuple[int, int, int, int, list[str]]:
    """Score a turn where filters are expected to be present.

    Args:
        filter_context: Dict containing expected filters, actual filters, and
            unknown keys.

    Returns:
        Tuple of (total_checks, passed_checks, divergence_checks,
        divergence_failures, failed_checks).

    """
    expected = filter_context.get("expected")
    expected_filters = expected if isinstance(expected, dict) else {}
    amenity_filters = filter_context.get("amenity_filters")
    service_filters = filter_context.get("service_filters")
    amenity_unknown = filter_context.get("amenity_unknown")
    service_unknown = filter_context.get("service_unknown")
    amenity_unknown_list = amenity_unknown if isinstance(amenity_unknown, list) else []
    service_unknown_list = service_unknown if isinstance(service_unknown, list) else []

    total_checks = 0
    passed_checks = 0
    divergence_checks = 0
    divergence_failures = 0
    failed_checks: list[str] = []

    total_checks, passed_checks = _apply_filter_check(
        condition=_has_filters(amenity_filters),
        label="amenities_filters_present",
        total_checks=total_checks,
        passed_checks=passed_checks,
        failed_checks=failed_checks,
    )
    total_checks, passed_checks = _apply_filter_check(
        condition=_has_filters(service_filters),
        label="services_filters_present",
        total_checks=total_checks,
        passed_checks=passed_checks,
        failed_checks=failed_checks,
    )
    total_checks, passed_checks = _apply_filter_check(
        condition=amenity_filters == expected_filters,
        label="amenities_filters_match_expected",
        total_checks=total_checks,
        passed_checks=passed_checks,
        failed_checks=failed_checks,
    )
    total_checks, passed_checks = _apply_filter_check(
        condition=service_filters == expected_filters,
        label="services_filters_match_expected",
        total_checks=total_checks,
        passed_checks=passed_checks,
        failed_checks=failed_checks,
    )

    if _has_filters(amenity_filters) and _has_filters(service_filters):
        divergence_checks += 1
        total_checks += 1
        if amenity_filters == service_filters:
            passed_checks += 1
        else:
            divergence_failures += 1
            failed_checks.append("filters_diverge_between_tools")

    total_checks, passed_checks = _apply_filter_check(
        condition=not (amenity_unknown_list or service_unknown_list),
        label="unknown_filter_keys_used",
        total_checks=total_checks,
        passed_checks=passed_checks,
        failed_checks=failed_checks,
    )

    return (
        total_checks,
        passed_checks,
        divergence_checks,
        divergence_failures,
        failed_checks,
    )


def _score_expected_filters_empty(
    filter_context: dict[str, Any],
) -> tuple[int, int, int, int, list[str]]:
    """Score a turn where no filters should be passed.

    Args:
        filter_context: Dict containing expected filters, actual filters, and
            unknown keys.

    Returns:
        Tuple of (total_checks, passed_checks, divergence_checks,
        divergence_failures, failed_checks).

    """
    amenity_filters = filter_context.get("amenity_filters")
    service_filters = filter_context.get("service_filters")
    amenity_unknown = filter_context.get("amenity_unknown")
    service_unknown = filter_context.get("service_unknown")
    amenity_unknown_list = amenity_unknown if isinstance(amenity_unknown, list) else []
    service_unknown_list = service_unknown if isinstance(service_unknown, list) else []

    total_checks = 0
    passed_checks = 0
    failed_checks: list[str] = []

    total_checks, passed_checks = _apply_filter_check(
        condition=not _has_filters(amenity_filters),
        label="amenities_no_filters",
        total_checks=total_checks,
        passed_checks=passed_checks,
        failed_checks=failed_checks,
    )
    total_checks, passed_checks = _apply_filter_check(
        condition=not _has_filters(service_filters),
        label="services_no_filters",
        total_checks=total_checks,
        passed_checks=passed_checks,
        failed_checks=failed_checks,
    )
    total_checks, passed_checks = _apply_filter_check(
        condition=not (amenity_unknown_list or service_unknown_list),
        label="unknown_filter_keys_used",
        total_checks=total_checks,
        passed_checks=passed_checks,
        failed_checks=failed_checks,
    )

    return total_checks, passed_checks, 0, 0, failed_checks


def _apply_filter_check(
    *,
    condition: bool,
    label: str,
    total_checks: int,
    passed_checks: int,
    failed_checks: list[str],
) -> tuple[int, int]:
    """Apply a filter check and update counters.

    Args:
        condition: Boolean indicating whether the check passes.
        label: Failure label to record when the check fails.
        total_checks: Current total checks count.
        passed_checks: Current passed checks count.
        failed_checks: List of failure labels to append to.

    Returns:
        Updated (total_checks, passed_checks).

    """
    total_checks += 1
    if condition:
        passed_checks += 1
    else:
        failed_checks.append(label)
    return total_checks, passed_checks


def _has_filters(filters: dict[str, Any] | None) -> bool:
    """Check whether a filters dict is non-empty.

    Args:
        filters: Filters dict or None.

    Returns:
        True if the dict is present and non-empty, otherwise False.

    """
    return isinstance(filters, dict) and bool(filters)


def _build_info_expected_filters_failure(
    failure_context: dict[str, Any],
) -> dict[str, Any]:
    """Build a failure entry for expected filter evaluation.

    Args:
        failure_context: Dict containing turn data for failure reporting.

    Returns:
        Failure dict for LangSmith feedback reporting.

    """
    idx = failure_context.get("idx")
    example_turn = failure_context.get("example_turn")
    turn_output = failure_context.get("turn_output")
    expected = failure_context.get("expected")
    amenity_filters = failure_context.get("amenity_filters")
    service_filters = failure_context.get("service_filters")
    amenity_unknown = failure_context.get("amenity_unknown")
    service_unknown = failure_context.get("service_unknown")
    failed_checks = failure_context.get("failed_checks")

    turn_index = idx if isinstance(idx, int) else -1
    example_turn_dict = example_turn if isinstance(example_turn, dict) else {}
    turn_output_dict = turn_output if isinstance(turn_output, dict) else {}
    expected_dict = expected if isinstance(expected, dict) else {}
    amenity_unknown_list = amenity_unknown if isinstance(amenity_unknown, list) else []
    service_unknown_list = service_unknown if isinstance(service_unknown, list) else []
    failed_checks_list = failed_checks if isinstance(failed_checks, list) else []

    return {
        "turn_index": turn_index,
        "user_snippet": truncate(str(example_turn_dict.get("user", "")), 160),
        "route_pred": turn_output_dict.get("route_pred"),
        "expected": expected_dict,
        "actual_amenities": amenity_filters,
        "actual_services": service_filters,
        "unknown_amenities": amenity_unknown_list,
        "unknown_services": service_unknown_list,
        "failed_checks": failed_checks_list,
    }
