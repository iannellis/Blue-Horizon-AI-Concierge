"""Expected search_rooms arguments evaluator for the Blue Horizon booking agent.

This module checks that the booking agent's ``search_rooms`` calls used the
filters a dataset turn expects. A turn labels its expectation in
``expected_search``: a partial spec of arguments, or a list of alternative
specs of which any one may match. The check is deterministic: it compares the
arguments the model actually sent, as captured by ``EvalCaptureCallback``,
without looking at what the search returned.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from eval._utils import json_detail_metric, truncate
from eval.evaluators._common import _get_example_turns, _iter_turn_outputs

if TYPE_CHECKING:
    from langsmith.schemas import Example, Run

    from eval.config import EvalConfig
    from eval.models import ExampleTurn, TurnOutput

_SEARCH_TOOL_NAME = "search_rooms"

# A larger requested limit is clamped to the same maximum, so an expected
# limit is a floor rather than an exact value.
_MINIMUM_KEYS = frozenset({"limit"})


def eval_booking_expected_search(
    run: Run,
    example: Example,
    *,
    cfg: EvalConfig,
) -> list[dict[str, Any]]:
    """Evaluate labeled turns' search_rooms arguments against expectations.

    A labeled turn passes when at least one of its search_rooms calls matches
    at least one of its expected specs. A labeled turn with no search_rooms
    call fails.

    Args:
        run: LangSmith run object containing turn outputs.
        example: LangSmith example object containing dataset turns.
        cfg: Evaluation configuration for evaluator limits.

    Returns:
        List of LangSmith feedback dicts. When no turn is labeled, a single
        ``booking_expected_search_skipped`` entry.

    """
    turn_outputs = _iter_turn_outputs(run)
    example_turns = _get_example_turns(example)
    limits = cfg.evaluator_limits
    labeled_turns = 0
    passed_turns = 0
    per_turn: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for idx in range(min(len(turn_outputs), len(example_turns))):
        example_turn = example_turns[idx]
        if example_turn.expected_search is None:
            continue
        labeled_turns += 1
        failure = _evaluate_turn(
            idx=idx,
            example_turn=example_turn,
            turn_output=turn_outputs[idx],
        )
        passed = failure is None
        passed_turns += int(passed)
        per_turn.append({"turn_index": idx, "pass_rate": float(passed)})
        if failure and len(failures) < limits.search_filter_failures_max:
            failures.append(failure)

    if labeled_turns == 0:
        return [
            {
                "key": "booking_expected_search_skipped",
                "score": 1.0,
                "comment": "No turns with expected_search in dataset.",
            },
        ]

    return [
        {"key": "booking_expected_search_turns", "score": float(labeled_turns)},
        {
            "key": "booking_expected_search_pass_rate",
            "score": passed_turns / labeled_turns,
        },
        json_detail_metric(
            key="booking_expected_search_per_turn",
            data=per_turn,
            max_len=limits.json_value_max,
        ),
        json_detail_metric(
            key="booking_expected_search_failures",
            data=failures,
            max_len=limits.json_value_max,
        ),
    ]


def _evaluate_turn(
    *,
    idx: int,
    example_turn: ExampleTurn,
    turn_output: TurnOutput,
) -> dict[str, Any] | None:
    """Check one labeled turn's search_rooms calls against its specs.

    Args:
        idx: Turn index within the run/example.
        example_turn: Dataset turn carrying ``expected_search``.
        turn_output: Run output for the aligned turn.

    Returns:
        None when some call matches some spec, else a failure record naming
        the mismatched keys of the closest call.

    """
    specs = _expected_specs(example_turn)
    calls = _search_calls(turn_output)
    closest: list[str] | None = None
    for spec in specs:
        for args in calls:
            mismatched = _mismatches(spec, args)
            if not mismatched:
                return None
            if closest is None or len(mismatched) < len(closest):
                closest = mismatched
    return {
        "turn_index": idx,
        "user_snippet": truncate(example_turn.user or "", 160),
        "expected": example_turn.expected_search,
        "calls": calls,
        "closest_mismatches": closest,
        "failed_check": "search_args_match" if calls else "search_called",
    }


def _expected_specs(example_turn: ExampleTurn) -> list[dict[str, Any]]:
    """Normalize a turn's ``expected_search`` into a list of alternative specs.

    Args:
        example_turn: Dataset turn carrying ``expected_search``.

    Returns:
        The spec alone in a list, or the list of alternatives as given.

    """
    expected = example_turn.expected_search
    if expected is None:
        return []
    if isinstance(expected, dict):
        return [expected]
    return expected


def _search_calls(turn_output: TurnOutput) -> list[dict[str, Any]]:
    """Collect the arguments of every search_rooms call in a turn.

    Args:
        turn_output: Run output for one turn.

    Returns:
        The model-supplied arguments of each search_rooms call, in call
        order. A call whose arguments were not captured counts as an empty
        dict.

    """
    return [
        dict(entry.search_args or {})
        for entry in turn_output.tool_summary
        if entry.tool == _SEARCH_TOOL_NAME
    ]


def _mismatches(spec: dict[str, Any], args: dict[str, Any]) -> list[str]:
    """List the spec keys a single search_rooms call fails to satisfy.

    Keys absent from the spec are unconstrained. A ``None`` spec value
    requires the argument to be unset. Lists compare as sets, numbers
    numerically, and keys in `_MINIMUM_KEYS` pass at or above the expected
    value. Anything else, including ISO date strings, compares by equality.

    Args:
        spec: One expected partial spec of search_rooms arguments.
        args: The arguments one search_rooms call was made with.

    Returns:
        Sorted names of the keys that did not match; empty when the call
        satisfies the spec.

    """
    return sorted(
        key
        for key, expected in spec.items()
        if not _value_matches(key, expected, args.get(key))
    )


def _value_matches(key: str, expected: object, actual: object) -> bool:
    """Check one spec entry against the value the model sent.

    Args:
        key: search_rooms argument name.
        expected: Value from the spec.
        actual: Value the model sent, or None when it omitted the argument.

    Returns:
        True when the model's value satisfies the spec entry.

    """
    if expected is None:
        return actual is None
    if isinstance(expected, list):
        return isinstance(actual, list) and _as_set(actual) == _as_set(expected)
    if _is_number(expected):
        if not _is_number(actual):
            return False
        if key in _MINIMUM_KEYS:
            return float(actual) >= float(expected)  # type: ignore[arg-type]
        return float(actual) == float(expected)  # type: ignore[arg-type]
    return actual == expected


def _as_set(values: list[Any]) -> set[str]:
    """Convert a list of filter values into a set for order-free comparison.

    Args:
        values: Filter values, normally strings.

    Returns:
        The values as a set of strings.

    """
    return {str(value) for value in values}


def _is_number(value: object) -> bool:
    """Check whether a value is an int or float, excluding bools.

    Args:
        value: Value to test.

    Returns:
        True for ints and floats that are not bools.

    """
    return isinstance(value, int | float) and not isinstance(value, bool)
