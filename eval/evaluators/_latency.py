"""Per-turn latency evaluator for Blue Horizon orchestration runs.

Captures per-turn wall-clock latency recorded by the eval target and stores
it as a LangSmith feedback value for downstream aggregation by ci_check.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from eval._utils import json_value
from eval.evaluators._common import _iter_turn_outputs

if TYPE_CHECKING:
    from langsmith.schemas import Example, Run


def eval_turn_latency(run: Run, example: Example) -> list[dict[str, Any]]:  # noqa: ARG001
    """Collect per-turn latency measurements and store them as a feedback value.

    Reads ``latency_ms`` from each turn output dict recorded by the eval
    target and emits a single ``latency_per_turn`` feedback item containing
    the list of ``{route, latency_ms}`` records.  The value is intended for
    aggregation across all cases by ``ci_check.py``; no per-case score is
    emitted.

    Args:
        run: LangSmith run object containing turn outputs with latency data.
        example: LangSmith example object (unused; present for evaluator signature).

    Returns:
        List containing a single LangSmith feedback dict with the raw latency
        records as a JSON-encoded value.

    """
    records = [
        {"route": t.get("route_pred"), "latency_ms": t["latency_ms"]}
        for t in _iter_turn_outputs(run)
        if "latency_ms" in t
    ]
    return [
        {
            "key": "latency_per_turn",
            "value": json_value(records),
        },
    ]
