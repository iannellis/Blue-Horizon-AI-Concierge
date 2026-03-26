"""CI baseline checker for Blue Horizon evaluation results.

Reads a results.jsonl produced by run_experiment.py, aggregates per-metric
scores across all cases, and compares each metric against a pinned baseline
JSON file.  Exits with code 0 when all checks pass, 1 when any fail.

Usage::

    python eval/ci_check.py <results.jsonl> [--baseline PATH]

"""

# ruff: noqa: T201

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from contextlib import suppress
from pathlib import Path
from typing import Any

from eval._result_utils import (
    _normalize_feedback,
    compute_latency_summary,
    format_latency_table,
)

_DEFAULT_BASELINE = (
    Path(__file__).parent / "baselines" / "hotel_agent_eval_20_baseline.json"
)

# Keys whose scores are metadata / skip-markers rather than real metric values.
_SKIP_KEYS: frozenset[str] = frozenset(
    {
        "route_confusions",
        "route_turns",
        "judge_raw_json",
        "info_reference_subset_failures",
        "info_reference_subset_turns",
        "info_expected_filters_failures",
        "info_expected_filters_turns",
        "info_expected_filters_per_turn",
        "injection_turns_labeled",
        "injection_turns_scanned",
        "rooms_outcome_per_turn",
        "rag_per_turn",
        "latency_per_turn",
    },
)


def main() -> None:
    """Parse arguments, run the baseline check, and exit with the result code.

    Raises:
        SystemExit: Always raised — exit code 0 on success, 1 on failure.

    """
    args = _parse_args()
    results_path = Path(args.results)
    baseline_path = Path(args.baseline)

    if not results_path.exists():
        print(f"ERROR: results file not found: {results_path}", file=sys.stderr)
        sys.exit(1)
    if not baseline_path.exists():
        print(f"ERROR: baseline file not found: {baseline_path}", file=sys.stderr)
        sys.exit(1)

    scores = _collect_scores(results_path)
    baseline = _load_baseline(baseline_path)
    passed = _run_checks(scores, baseline)
    _print_latency_stats(results_path)
    sys.exit(0 if passed else 1)


def _parse_args() -> argparse.Namespace:
    """Build and parse the CLI argument parser.

    Returns:
        Parsed namespace with ``results`` and ``baseline`` attributes.

    """
    parser = argparse.ArgumentParser(
        description="Compare eval results.jsonl against a pinned baseline.",
    )
    parser.add_argument("results", help="Path to results.jsonl from run_experiment.py")
    parser.add_argument(
        "--baseline",
        default=str(_DEFAULT_BASELINE),
        help=(
            "Path to baseline JSON file "
            "(default: eval/baselines/hotel_agent_eval_20_baseline.json)"
        ),
    )
    return parser.parse_args()


def _collect_scores(results_path: Path) -> dict[str, list[float]]:
    """Parse results.jsonl and collect numeric scores per metric.

    Each line in the file is a JSON object representing one evaluated case.
    Feedback items are stored as lists of ``[key, value]`` pairs.  Only
    numeric ``score`` values are collected; ``None`` scores and metadata-only
    keys listed in ``_SKIP_KEYS`` are ignored.

    Args:
        results_path: Path to the results JSONL file.

    Returns:
        Mapping from metric name to list of per-case float scores.

    """
    scores: dict[str, list[float]] = defaultdict(list)
    with results_path.open(encoding="utf-8") as fh:
        for raw_line in fh:
            stripped = raw_line.strip()
            if not stripped:
                continue
            case = json.loads(stripped)
            feedback = _normalize_feedback(case.get("evaluators", {}))
            for key, payload in feedback.items():
                score = payload.get("score")
                if key is None or key in _SKIP_KEYS:
                    continue
                if score is None:
                    continue
                with suppress(TypeError, ValueError):
                    scores[key].append(float(score))
    return dict(scores)


def _load_baseline(baseline_path: Path) -> dict[str, Any]:
    """Load and return the baseline JSON file.

    Args:
        baseline_path: Path to the baseline JSON file.

    Returns:
        Parsed baseline dict containing a ``metrics`` key.

    """
    with baseline_path.open(encoding="utf-8") as fh:
        return json.load(fh)  # type: ignore[no-any-return]


def _run_checks(
    scores: dict[str, list[float]],
    baseline: dict[str, Any],
) -> bool:
    """Compare collected scores against the baseline and print a report.

    Strict metrics require every individual case score to equal 1.0.  Soft
    metrics require the mean across all cases to be >= ``min_allowed``.

    Args:
        scores: Per-metric lists of float scores collected from results.jsonl.
        baseline: Parsed baseline dict with a ``metrics`` sub-dict.

    Returns:
        True if all checks pass, False if any fail.

    """
    metrics: dict[str, Any] = baseline.get("metrics", {})
    all_passed = True

    header = f"{'Metric':<40} {'Baseline':>9} {'Min':>9} {'Actual':>9}  Result"
    print(header)
    print("-" * len(header))

    for metric_name, spec in metrics.items():
        baseline_val: float = spec["baseline"]
        min_allowed: float = spec["min_allowed"]
        strict: bool = spec["strict"]

        case_scores = scores.get(metric_name, [])
        passed, actual_str = _evaluate_metric(
            case_scores=case_scores,
            min_allowed=min_allowed,
            strict=strict,
        )
        if not passed:
            all_passed = False

        result_label = "PASS" if passed else "FAIL"
        print(
            f"{metric_name:<40} {baseline_val:>9.4f} {min_allowed:>9.4f}"
            f" {actual_str:>9}  {result_label}",
        )

    print()
    if all_passed:
        print("All checks passed.")
    else:
        print("One or more checks FAILED.", file=sys.stderr)

    return all_passed


def _evaluate_metric(
    *,
    case_scores: list[float],
    min_allowed: float,
    strict: bool,
) -> tuple[bool, str]:
    """Evaluate a single metric against its threshold.

    Args:
        case_scores: Per-case float scores from results.jsonl.
        min_allowed: Minimum acceptable value (mean for soft, 1.0 for strict).
        strict: When True every case score must equal 1.0; when False the mean
            must be >= ``min_allowed``.

    Returns:
        Tuple of (passed, display_string) where display_string is the actual
        value formatted for the report table.

    """
    if not case_scores:
        return False, "N/A"

    if strict:
        failing = [s for s in case_scores if s < 1.0]
        passed = len(failing) == 0
        actual_str = f"{len(failing)}fail" if not passed else "1.0000"
    else:
        mean = sum(case_scores) / len(case_scores)
        passed = mean >= min_allowed
        actual_str = f"{mean:.4f}"

    return passed, actual_str


def _print_latency_stats(results_path: Path) -> None:
    """Print p50/p95/p99 per-route latency quantiles from results.jsonl.

    Args:
        results_path: Path to the results JSONL file produced by
            ``run_experiment.py``.

    """
    latency = compute_latency_summary(results_path)
    table = format_latency_table(latency)
    print(table if table else "\nNo latency data found.")


if __name__ == "__main__":
    main()
