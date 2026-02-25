"""Artifact writing and summary generation for the stress test harness.

Provides helpers for computing run statistics, writing JSONL/JSON artifacts,
and configuring logging for a stress run.
"""

from __future__ import annotations

import json
import logging
import statistics
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from pathlib import Path

    from eval.stress.models import StressRunConfig

logger = logging.getLogger(__name__)


def _build_summary(  # noqa: PLR0913
    cfg: StressRunConfig,
    *,
    op_logs: list[dict[str, object]],
    invariants: dict[str, object],
    reconciliation: dict[str, object],
    targets: list[dict[str, object]],
    hot_targets: list[dict[str, object]],
    elapsed_s: float,
) -> dict[str, object]:
    """Compute summary statistics and status for the stress run.

    Args:
        cfg: The stress run configuration.
        op_logs: The per-operation logs.
        invariants: The invariant check results.
        reconciliation: The post-hoc log/DB reconciliation results.
        targets: The full target pool.
        hot_targets: The hot contention subset.
        elapsed_s: The total elapsed time in seconds.

    Returns:
        The stress summary dictionary.

    """
    total_ops = len(op_logs)
    ops_per_s = (total_ops / elapsed_s) if elapsed_s > 0 else 0.0

    latencies = [
        float(cast("float", op["latency_ms"])) for op in op_logs if "latency_ms" in op
    ]
    lat_sorted = sorted(latencies)
    latency_mean = statistics.fmean(latencies) if latencies else 0.0
    latency_median = statistics.median(latencies) if latencies else 0.0
    latency_p95 = _percentile(lat_sorted, 0.95) if lat_sorted else 0.0

    success_count = sum(1 for op in op_logs if op.get("outcome") == "success")
    conflict_count = sum(1 for op in op_logs if op.get("outcome") == "conflict")
    error_count = sum(1 for op in op_logs if op.get("outcome") == "error")

    def _rate(count: int) -> float:
        """Compute a rate safely when the denominator may be zero.

        Args:
            count: The numerator count.

        Returns:
            The rate ``count / total_ops`` or ``0.0`` when ``total_ops`` is 0.

        """
        return (count / total_ops) if total_ops else 0.0

    summary: dict[str, object] = {
        "users": cfg.users,
        "ops_per_user": cfg.ops_per_user,
        "max_concurrency": cfg.max_concurrency,
        "total_ops": total_ops,
        "elapsed_s": round(elapsed_s, 3),
        "ops_per_s": round(ops_per_s, 3),
        "latency_ms": {
            "mean": round(latency_mean, 2),
            "median": round(latency_median, 2),
            "p95": round(latency_p95, 2),
        },
        "outcomes": {
            "success": {"count": success_count, "rate": round(_rate(success_count), 4)},
            "conflict": {
                "count": conflict_count,
                "rate": round(_rate(conflict_count), 4),
            },
            "error": {"count": error_count, "rate": round(_rate(error_count), 4)},
        },
        "invariants": invariants,
        "reconciliation": reconciliation,
        "targets": {
            "total": len(targets),
            "hot_count": len(hot_targets),
            "hot_targets": hot_targets,
        },
    }

    summary["status"] = (
        "PASS"
        if bool(invariants.get("passed")) and bool(reconciliation.get("passed"))
        else "FAIL"
    )
    return summary


def _percentile(sorted_vals: list[float], p: float) -> float:
    """Compute a percentile from a pre-sorted list without numpy.

    Args:
        sorted_vals: A list of numeric values sorted in ascending order.
        p: The percentile as a fraction in the inclusive range ``[0, 1]``.

    Returns:
        The value at index ``int(p * (n - 1))``, or ``0.0`` if empty.

    """
    if not sorted_vals:
        return 0.0
    idx = int(p * (len(sorted_vals) - 1))
    return float(sorted_vals[idx])


def _write_artifacts(
    cfg: StressRunConfig,
    *,
    run_id: str,
    op_logs: list[dict[str, object]],
    user_logs: list[dict[str, object]],
    summary: dict[str, object],
) -> None:
    """Write JSON artifacts for operations, users, summary, and failures.

    Args:
        cfg: The stress run configuration.
        run_id: The unique run identifier, used as the output directory stem so
            artifact paths match the ``run:{run_id}`` LangSmith tag.
        op_logs: The per-operation logs.
        user_logs: The per-user logs.
        summary: The computed summary dictionary.

    """
    run_dir = cfg.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    ops_path = run_dir / "stress_ops.jsonl"
    with ops_path.open("w", encoding="utf-8") as f:
        for op in op_logs:
            f.write(json.dumps(op, ensure_ascii=True) + "\n")

    users_path = run_dir / "stress_users.jsonl"
    with users_path.open("w", encoding="utf-8") as f:
        for user in user_logs:
            f.write(json.dumps(user, ensure_ascii=True) + "\n")

    summary_path = run_dir / "stress_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)

    failures = [op for op in op_logs if _is_failure_entry(op)]
    if failures:
        failures_path = run_dir / "stress_failures.jsonl"
        with failures_path.open("w", encoding="utf-8") as f:
            for op in failures:
                failure_record: dict[str, object] = {
                    "ts": op.get("ts"),
                    "user_idx": op.get("user_idx"),
                    "op_idx": op.get("op_idx"),
                    "op_type": op.get("op_type"),
                    "outcome": op.get("outcome"),
                    "user_query": op.get("prompt"),
                    "agent_response": op.get("agent_response"),
                    "error": op.get("error"),
                    "sql_calls": op.get("sql_calls"),
                }
                f.write(json.dumps(failure_record, ensure_ascii=True) + "\n")


def _is_failure_entry(op: dict[str, object]) -> bool:
    """Return True when an operation log entry represents a failure worth diagnosing.

    An entry is a failure if it has a Python-level error, an LLM-classified
    "error" outcome, or any ``run_sql`` call that returned an error from the
    database layer.

    Args:
        op: A per-operation log dictionary.

    Returns:
        ``True`` when the entry should be included in ``stress_failures.jsonl``.

    """
    if op.get("outcome") == "error" or op.get("error"):
        return True
    sql_calls = op.get("sql_calls")
    if isinstance(sql_calls, list):
        return any(
            isinstance(c, dict) and c.get("error") for c in sql_calls
        )
    return False


def _configure_logging(run_name: str, log_dir: Path) -> None:
    """Configure logging for the stress run.

    Attaches both a file handler (``<log_dir>/<run_name>.log``) and a stderr
    stream handler so that progress is visible in the terminal as well as
    persisted to disk.

    Args:
        run_name: Name of the current stress run, used as the log filename stem.
        log_dir: Directory where log files should be written.

    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_name}.log"

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(file_handler)
    root.addHandler(stream_handler)
