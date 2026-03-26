"""Routing accuracy evaluator for Blue Horizon orchestration.

Evaluates per-turn routing accuracy by comparing predicted routes against expected
routes, computing overall accuracy scores and confusion matrices.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from eval._utils import json_value
from eval.evaluators._common import _get_example_turns, _iter_turn_outputs

if TYPE_CHECKING:
    from langsmith.schemas import Example, Run


def eval_routing_accuracy(run: Run, example: Example) -> list[dict[str, Any]]:
    """Evaluate per-turn routing accuracy.

    Args:
        run: LangSmith run object containing turn outputs.
        example: LangSmith example object containing expected routes.

    Returns:
        List of LangSmith metric dicts including accuracy and confusion counts.

    """
    turn_outputs = _iter_turn_outputs(run)
    turns = _get_example_turns(example)

    total_turns = max(len(turn_outputs), len(turns))
    if total_turns == 0:
        return [
            {
                "key": "route_turns",
                "score": 0.0,
            },
            {
                "key": "route_accuracy",
                "score": 0.0,
                "comment": "No turns available to score.",
            },
        ]

    correct = 0
    confusions: dict[str, int] = {}

    for idx in range(total_turns):
        expected = None
        if idx < len(turns):
            expected = turns[idx].get("expected_route")
        expected_str = str(expected) if expected is not None else "<missing>"

        pred = None
        if idx < len(turn_outputs):
            pred = turn_outputs[idx].get("route_pred")
        pred_str = str(pred) if pred is not None else "<missing>"

        if expected_str == pred_str and expected is not None:
            correct += 1
        key = f"{expected_str}->{pred_str}"
        confusions[key] = confusions.get(key, 0) + 1

    missing_outputs = max(0, len(turns) - len(turn_outputs))
    extra_outputs = max(0, len(turn_outputs) - len(turns))
    score = correct / total_turns if total_turns else 0.0

    comment = (
        f"Correct {correct}/{total_turns}. "
        f"Missing outputs: {missing_outputs}. Extra outputs: {extra_outputs}."
    )

    return [
        {
            "key": "route_turns",
            "score": float(total_turns),
        },
        {
            "key": "route_accuracy",
            "score": score,
            "comment": comment,
        },
        {
            "key": "route_confusions",
            "value": json_value(confusions),
        },
    ]
