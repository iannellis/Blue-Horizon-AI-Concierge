"""Info expected filters evaluators for Blue Horizon hotel agent.

This module provides evaluators to check that info tool calls use expected
filters correctly, ensuring that amenity and service queries include proper
filtering parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from eval._utils import json_detail_metric, truncate
from eval.evaluators._common import _get_example_turns, _iter_turn_outputs

if TYPE_CHECKING:
    from langsmith.schemas import Example, Run

    from eval.config import EvalConfig
    from eval.models import ExampleTurn, ToolSummaryEntry, TurnOutput


@dataclass(frozen=True)
class _FilterObservation:
    """Amenities/services filters observed for one turn, against expectation.

    Attributes:
        expected: Canonicalized expected filters for the turn (empty when
            the case expects no filters at all).
        amenity_filters: Normalized filters query_amenities was called with,
            or None if the tool was not called this turn.
        service_filters: Normalized filters query_services was called with,
            or None if the tool was not called this turn.
        amenity_unknown: Filter keys query_amenities could not canonicalize.
        service_unknown: Filter keys query_services could not canonicalize.

    """

    expected: dict[str, Any]
    amenity_filters: dict[str, Any] | None
    service_filters: dict[str, Any] | None
    amenity_unknown: list[str]
    service_unknown: list[str]


@dataclass(frozen=True)
class _FilterScore:
    """Tally of named pass/fail checks for one turn's filter evaluation.

    Attributes:
        total_checks: Number of checks evaluated for the turn.
        passed_checks: Number of those checks that passed.
        divergence_checks: 1 if both tools were called this turn (a
            divergence check applies), else 0.
        divergence_failures: 1 if both tools were called with different
            filters, else 0.
        failed_checks: Labels of the checks that failed, for the failure
            detail record.

    """

    total_checks: int
    passed_checks: int
    divergence_checks: int
    divergence_failures: int
    failed_checks: list[str]


def eval_info_expected_filters(
    run: Run,
    example: Example,
    *,
    cfg: EvalConfig,
) -> list[dict[str, Any]]:
    """Evaluate expected info-tool filters against logged tool usage.

    Args:
        run: LangSmith run object containing turn outputs.
        example: LangSmith example object containing dataset turns.
        cfg: Evaluation configuration for evaluator limits.

    Returns:
        List of LangSmith feedback dicts for expected filter checks.

    """
    turn_outputs = _iter_turn_outputs(run)
    example_turns = _get_example_turns(example)
    total_turns = min(len(turn_outputs), len(example_turns))
    labeled_turns = 0
    total_checks = 0
    passed_checks = 0
    per_turn_pass_rates: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    divergence_checks = 0
    divergence_failures = 0
    limits = cfg.evaluator_limits

    for idx in range(total_turns):
        example_turn = example_turns[idx]
        if example_turn.expected_filters is None:
            continue
        labeled_turns += 1
        score, failure = _evaluate_expected_filters_turn(
            idx=idx,
            example_turn=example_turn,
            turn_output=turn_outputs[idx],
        )
        total_checks += score.total_checks
        passed_checks += score.passed_checks
        divergence_checks += score.divergence_checks
        divergence_failures += score.divergence_failures
        if score.total_checks:
            per_turn_pass_rates.append(
                {
                    "turn_index": idx,
                    "pass_rate": score.passed_checks / score.total_checks,
                },
            )
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

    failures_entry = json_detail_metric(
        key="info_expected_filters_failures",
        data=failures,
        max_len=limits.json_value_max,
    )
    per_turn_entry = json_detail_metric(
        key="info_expected_filters_per_turn",
        data=per_turn_pass_rates,
        max_len=limits.json_value_max,
    )

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
        per_turn_entry,
        failures_entry,
    ]


def _evaluate_expected_filters_turn(
    *,
    idx: int,
    example_turn: ExampleTurn,
    turn_output: TurnOutput,
) -> tuple[_FilterScore, dict[str, Any] | None]:
    """Evaluate expected filters for a single labeled turn.

    Args:
        idx: Turn index within the run/example.
        example_turn: Dataset turn containing expected filters.
        turn_output: Run output for the aligned turn.

    Returns:
        Tuple of (score, failure_entry or None).

    """
    expected_full = example_turn.expected_filters
    expected = _canon_expected_filters(expected_full) if expected_full else {}
    amenity_filters, amenity_unknown = _extract_filters_for_tool(
        turn_output.tool_summary,
        "query_amenities",
    )
    service_filters, service_unknown = _extract_filters_for_tool(
        turn_output.tool_summary,
        "query_services",
    )
    observation = _FilterObservation(
        expected=expected,
        amenity_filters=amenity_filters,
        service_filters=service_filters,
        amenity_unknown=amenity_unknown,
        service_unknown=service_unknown,
    )
    score = _score_expected_filters(observation)

    failure = None
    if score.failed_checks:
        failure = _build_info_expected_filters_failure(
            idx=idx,
            example_turn=example_turn,
            turn_output=turn_output,
            observation=observation,
            failed_checks=score.failed_checks,
        )

    return score, failure


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
    tool_summary: list[ToolSummaryEntry],
    tool_name: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Extract normalized filters and unknown keys for a tool from summaries.

    Args:
        tool_summary: Tool summary entries from run outputs.
        tool_name: Tool name to search for.

    Returns:
        Tuple of (filters_norm, unknown_keys) for the tool.

    """
    for entry in reversed(tool_summary):
        if entry.tool != tool_name:
            continue
        return entry.filters_norm, entry.filters_unknown_keys
    return None, []


def _score_expected_filters(observation: _FilterObservation) -> _FilterScore:
    """Score expected filters against amenities/services tool outputs.

    Args:
        observation: Filters observed for the turn, against expectation.

    Returns:
        The tallied score for the turn.

    """
    if observation.expected:
        return _score_expected_filters_present(observation)
    return _score_expected_filters_empty(observation)


def _score_expected_filters_present(observation: _FilterObservation) -> _FilterScore:
    """Score a turn where filters are expected to be present.

    Args:
        observation: Filters observed for the turn, against expectation.

    Returns:
        The tallied score for the turn.

    """
    checks: list[tuple[str, bool]] = [
        ("amenities_filters_present", _has_filters(observation.amenity_filters)),
        ("services_filters_present", _has_filters(observation.service_filters)),
        (
            "amenities_filters_match_expected",
            observation.amenity_filters == observation.expected,
        ),
        (
            "services_filters_match_expected",
            observation.service_filters == observation.expected,
        ),
    ]

    divergence_checks = 0
    divergence_failures = 0
    if _has_filters(observation.amenity_filters) and _has_filters(
        observation.service_filters,
    ):
        divergence_checks = 1
        matched = observation.amenity_filters == observation.service_filters
        checks.append(("filters_diverge_between_tools", matched))
        divergence_failures = 0 if matched else 1

    checks.append(
        (
            "unknown_filter_keys_used",
            not (observation.amenity_unknown or observation.service_unknown),
        ),
    )

    return _fold_checks(
        checks,
        divergence_checks=divergence_checks,
        divergence_failures=divergence_failures,
    )


def _score_expected_filters_empty(observation: _FilterObservation) -> _FilterScore:
    """Score a turn where no filters should be passed.

    Args:
        observation: Filters observed for the turn, against expectation.

    Returns:
        The tallied score for the turn.

    """
    checks: list[tuple[str, bool]] = [
        ("amenities_no_filters", not _has_filters(observation.amenity_filters)),
        ("services_no_filters", not _has_filters(observation.service_filters)),
        (
            "unknown_filter_keys_used",
            not (observation.amenity_unknown or observation.service_unknown),
        ),
    ]
    return _fold_checks(checks)


def _fold_checks(
    checks: list[tuple[str, bool]],
    *,
    divergence_checks: int = 0,
    divergence_failures: int = 0,
) -> _FilterScore:
    """Tally a list of named pass/fail checks into a _FilterScore.

    Args:
        checks: (label, passed) pairs; a failing check's label is recorded.
        divergence_checks: Divergence checks to fold in, if any (0 or 1).
        divergence_failures: Divergence failures to fold in, if any (0 or 1).

    Returns:
        The tallied _FilterScore.

    """
    failed = [label for label, passed in checks if not passed]
    return _FilterScore(
        total_checks=len(checks),
        passed_checks=len(checks) - len(failed),
        divergence_checks=divergence_checks,
        divergence_failures=divergence_failures,
        failed_checks=failed,
    )


def _has_filters(filters: dict[str, Any] | None) -> bool:
    """Check whether a filters dict is non-empty.

    Args:
        filters: Filters dict or None.

    Returns:
        True if the dict is present and non-empty, otherwise False.

    """
    return isinstance(filters, dict) and bool(filters)


def _build_info_expected_filters_failure(
    *,
    idx: int,
    example_turn: ExampleTurn,
    turn_output: TurnOutput,
    observation: _FilterObservation,
    failed_checks: list[str],
) -> dict[str, Any]:
    """Build a failure entry for expected filter evaluation.

    Args:
        idx: Turn index within the run/example.
        example_turn: Dataset turn the filters were checked against.
        turn_output: Run output for the aligned turn.
        observation: Filters observed for the turn.
        failed_checks: Labels of the checks that failed.

    Returns:
        Failure dict for LangSmith feedback reporting.

    """
    return {
        "turn_index": idx,
        "user_snippet": truncate(example_turn.user or "", 160),
        "route_pred": turn_output.route_pred,
        "expected": observation.expected,
        "actual_amenities": observation.amenity_filters,
        "actual_services": observation.service_filters,
        "unknown_amenities": observation.amenity_unknown,
        "unknown_services": observation.service_unknown,
        "failed_checks": failed_checks,
    }
