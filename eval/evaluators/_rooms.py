"""Rooms evaluators for Blue Horizon hotel agent.

This module provides evaluators for rooms tool outcomes and database invariant
checks, ensuring booking operations behave correctly.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, TypeGuard

from psycopg_pool import AsyncConnectionPool

from eval._utils import coerce_int, json_value
from eval.evaluators._common import (
    _get_example_turns,
    _iter_turn_outputs,
)

if TYPE_CHECKING:
    from langsmith.schemas import Example, Run

    from eval.config import EvalConfig

try:  # Optional app import for DB URL lookup.
    from blue_horizon.config import load_app_config as _load_app_config

    _PGSQL_IMPORT_ERROR: Exception | None = None
except Exception as _exc:  # noqa: BLE001
    _load_app_config = None
    _PGSQL_IMPORT_ERROR = _exc

_EVAL_POOL_LOCK = asyncio.Lock()
_EVAL_POOL: AsyncConnectionPool[Any] | None = None


async def eval_rooms_outcome_and_invariants(
    run: Run,
    example: Example,
    *,
    cfg: EvalConfig,
) -> list[dict[str, Any]]:
    """Evaluate rooms tool outcomes and DB invariants for rooms turns.

    Args:
        run: LangSmith run object containing tool summaries.
        example: LangSmith example with per-turn ``expected_success`` flags.
        cfg: Evaluation configuration with DB URL and evaluator limits.

    Returns:
        List of LangSmith metric dicts for tool outcomes and DB invariants.

    """
    outputs = run.outputs or {}
    turn_outputs = outputs.get("turn_outputs") or []
    has_rooms = any(
        t.get("route_pred") == "rooms" for t in turn_outputs if isinstance(t, dict)
    )
    if not has_rooms:
        return [
            {
                "key": "rooms_invariants_skipped",
                "score": 1.0,
                "comment": "No rooms turns detected.",
            },
        ]

    outcome_metrics = _score_rooms_tool_outcomes(run, example, cfg)
    invariants_results = await _check_rooms_db_invariants(cfg)
    return [
        *outcome_metrics,
        *invariants_results,
    ]


def _score_rooms_tool_outcomes(
    run: Run,
    example: Example,
    cfg: EvalConfig,
) -> list[dict[str, Any]]:
    """Score rooms tool outcomes against expected results.

    Args:
        run: LangSmith run object containing tool summaries.
        example: LangSmith example with per-turn ``expected_success`` flags.
        cfg: Evaluation configuration for evaluator limits.

    Returns:
        List of LangSmith metrics for tool errors, outcome match rate, and a
        soft rowcount presence check.

    """
    limits = cfg.evaluator_limits
    turn_outputs = _iter_turn_outputs(run)
    example_turns = _get_example_turns(example)

    counts = _init_rooms_outcome_counts()
    unexpected_fail_details: list[dict[str, Any]] = []

    for global_idx, turn in enumerate(turn_outputs):
        if turn.get("route_pred") != "rooms":
            continue
        tool_summary = turn.get("tool_summary") or []
        if not isinstance(tool_summary, list):
            continue

        expected_success: bool | None = None
        if global_idx < len(example_turns):
            es = example_turns[global_idx].get("expected_success")
            if isinstance(es, bool):
                expected_success = es

        _accumulate_rooms_turn(
            turn_idx=global_idx,
            tool_summary=tool_summary,
            counts=counts,
            unexpected_fail_details=unexpected_fail_details,
            expected_success=expected_success,
        )

    tool_error_score, tool_error_comment = _format_tool_error_metric(counts)
    match_rate, match_comment = _format_outcome_match_metric(counts)
    rowcount_score, rowcount_comment = _format_rowcount_metric(counts)

    metrics: list[dict[str, Any]] = [
        {
            "key": "rooms_tool_errors",
            "score": tool_error_score,
            "comment": tool_error_comment,
        },
        {
            "key": "rooms_no_unexpected_failure_rate",
            "score": match_rate,
            "comment": match_comment,
        },
        {
            "key": "rooms_rowcount_sanity",
            "score": rowcount_score,
            "comment": rowcount_comment,
        },
    ]
    if unexpected_fail_details:
        raw_failures = json_value(unexpected_fail_details)
        if len(raw_failures) > limits.json_value_max:
            metrics.append(
                {
                    "key": "rooms_unexpected_failures",
                    "value": json_value(
                        unexpected_fail_details,
                        max_len=limits.json_value_max,
                    ),
                    "comment": "JSON truncated",
                },
            )
        else:
            metrics.append(
                {"key": "rooms_unexpected_failures", "value": raw_failures},
            )
    return metrics


def _init_rooms_outcome_counts() -> dict[str, int]:
    """Initialize counters used for rooms tool outcome scoring.

    Returns:
        Dict of integer counters keyed by metric name.

    """
    return {
        "tool_errors": 0,
        "rowcount_present": 0,
        "run_sql_calls": 0,
        "outcome_matches": 0,
        "unexpected_failures": 0,
    }


def _accumulate_rooms_turn(
    *,
    turn_idx: int,
    tool_summary: list[object],
    counts: dict[str, int],
    unexpected_fail_details: list[dict[str, Any]],
    expected_success: bool | None,
) -> None:
    """Accumulate outcome counts for a single rooms turn.

    Args:
        turn_idx: Global turn index within the run.
        tool_summary: Tool summary entries for the turn.
        counts: Mutable counter dict updated in place.
        unexpected_fail_details: Unexpected failure details list updated in
            place.
        expected_success: Whether the example expects this turn to succeed.
            ``False`` means a failure is expected and should not be penalised.

    """
    for entry in tool_summary:
        if not _is_run_sql_entry(entry):
            continue
        counts["run_sql_calls"] += 1
        if _has_tool_error(entry):
            counts["tool_errors"] += 1
        if entry.get("rowcount") is not None:
            counts["rowcount_present"] += 1
        _accumulate_outcome(
            turn_idx=turn_idx,
            entry=entry,
            counts=counts,
            unexpected_fail_details=unexpected_fail_details,
            expected_success=expected_success,
        )


def _is_run_sql_entry(entry: object) -> TypeGuard[dict[str, Any]]:
    """Check whether a tool summary entry represents a run_sql call.

    Args:
        entry: Tool summary entry.

    Returns:
        True when the entry is a dict for the run_sql tool.

    """
    return isinstance(entry, dict) and entry.get("tool") == "run_sql"


def _has_tool_error(summary: dict[str, Any]) -> bool:
    """Detect whether a tool summary indicates an error.

    Args:
        summary: Tool summary dict for a run_sql call.

    Returns:
        True if the summary indicates error status or includes an error message.

    """
    status = summary.get("status")
    if isinstance(status, str) and status.lower() == "error":
        return True
    return bool(summary.get("error"))


def _accumulate_outcome(
    *,
    turn_idx: int,
    entry: dict[str, Any],
    counts: dict[str, int],
    unexpected_fail_details: list[dict[str, Any]],
    expected_success: bool | None,
) -> None:
    """Accumulate outcome counts for a single run_sql entry.

    Args:
        turn_idx: Global turn index within the run.
        entry: Tool summary dict for a run_sql call.
        counts: Mutable counter dict updated in place.
        unexpected_fail_details: Unexpected failure details list updated in
            place.
        expected_success: Whether the example expects this turn to succeed.
            When ``False``, a booking/modify failure is not penalised.

    """
    rows = _extract_rows(entry)
    first_row = _extract_first_row(rows)
    op = _detect_atomic_op(first_row)
    if op is None or first_row is None:
        return
    success, detail = _score_atomic_outcome(op=op, row=first_row)
    if success:
        counts["outcome_matches"] += 1
        return
    if expected_success is False:
        counts["outcome_matches"] += 1
        return
    counts["unexpected_failures"] += 1
    unexpected_fail_details.append({"turn": turn_idx, "op": op, **detail})


def _extract_rows(entry: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract rows from a run_sql tool summary entry.

    Args:
        entry: Tool summary dict for a run_sql call.

    Returns:
        List of row dicts, or an empty list when unavailable.

    """
    rows = entry.get("rows")
    if isinstance(rows, list):
        return [r for r in rows if isinstance(r, dict)]
    result = entry.get("result")
    if isinstance(result, dict):
        nested_rows = result.get("rows")
        if isinstance(nested_rows, list):
            return [r for r in nested_rows if isinstance(r, dict)]
    return []


def _extract_first_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Extract the first row from a rows list.

    Args:
        rows: List of row dicts.

    Returns:
        First row dict, or None when empty.

    """
    if not rows:
        return None
    return rows[0]


def _detect_atomic_op(row: dict[str, Any] | None) -> str | None:
    """Detect BOOK/MODIFY atomic outcomes based on row fields.

    Args:
        row: First row dict returned by run_sql.

    Returns:
        Operation string ("book" or "modify") when detected, else None.

    """
    if not row:
        return None
    keys = set(row.keys())
    if {"nights_requested", "nights_booked"}.issubset(keys):
        return "book"
    if {
        "nights_to_release",
        "nights_released",
        "nights_to_acquire",
        "nights_acquired",
    }.issubset(keys):
        return "modify"
    return None


def _score_atomic_outcome(op: str, row: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Score an atomic BOOK or MODIFY outcome row.

    Args:
        op: Operation name detected from the row.
        row: Row dict containing outcome counts.

    Returns:
        Tuple of (success, details) where details are safe for logging.

    """
    if op == "book":
        nights_requested = coerce_int(row.get("nights_requested"))
        nights_booked = coerce_int(row.get("nights_booked"))
        success = (
            nights_requested is not None
            and nights_booked is not None
            and nights_requested > 0
            and nights_booked == nights_requested
        )
        return success, {
            "nights_requested": nights_requested,
            "nights_booked": nights_booked,
        }

    nights_to_release = coerce_int(row.get("nights_to_release"))
    nights_released = coerce_int(row.get("nights_released"))
    nights_to_acquire = coerce_int(row.get("nights_to_acquire"))
    nights_acquired = coerce_int(row.get("nights_acquired"))
    success = (
        nights_to_release is not None
        and nights_released is not None
        and nights_to_acquire is not None
        and nights_acquired is not None
        and nights_released == nights_to_release
        and nights_acquired == nights_to_acquire
    )
    return success, {
        "nights_to_release": nights_to_release,
        "nights_released": nights_released,
        "nights_to_acquire": nights_to_acquire,
        "nights_acquired": nights_acquired,
    }


def _format_tool_error_metric(counts: dict[str, int]) -> tuple[float, str]:
    """Format the tool error metric from accumulated counts.

    Args:
        counts: Counter dict accumulated across rooms turns.

    Returns:
        Tuple of (score, comment) for rooms_tool_errors.

    """
    tool_errors = counts["tool_errors"]
    if tool_errors == 0:
        return 1.0, "No tool errors detected."
    return 0.0, f"{tool_errors} run_sql errors detected."


def _format_outcome_match_metric(counts: dict[str, int]) -> tuple[float, str]:
    """Format the outcome match metric from accumulated counts.

    Args:
        counts: Counter dict accumulated across rooms turns.

    Returns:
        Tuple of (score, comment) for rooms_outcome_match_rate.

    """
    matches = counts["outcome_matches"]
    unexpected = counts["unexpected_failures"]
    total = matches + unexpected
    if total == 0:
        return 1.0, "No BOOK/MODIFY outcomes to evaluate."
    return (
        matches / total,
        f"Outcomes matched {matches}/{total}.",
    )


def _format_rowcount_metric(counts: dict[str, int]) -> tuple[float, str]:
    """Format the rowcount presence metric from accumulated counts.

    Args:
        counts: Counter dict accumulated across rooms turns.

    Returns:
        Tuple of (score, comment) for rooms_rowcount_sanity.

    """
    run_sql_calls = counts["run_sql_calls"]
    rowcount_present = counts["rowcount_present"]
    if run_sql_calls == 0:
        return 1.0, "No run_sql calls observed."
    score = rowcount_present / run_sql_calls
    comment = f"Rowcount present on {rowcount_present}/{run_sql_calls} run_sql calls."
    return score, comment


async def _check_rooms_db_invariants(cfg: EvalConfig) -> list[dict[str, Any]]:
    """Check rooms DB invariants in the public schema.

    Args:
        cfg: Evaluation configuration with DB URL.

    Returns:
        List of LangSmith metric dicts for DB invariants.

    """
    pool = await _ensure_eval_pool(cfg)
    double_booking_rows = 0
    null_status_count = 0

    async with pool.connection() as conn, conn.cursor() as cur:
        await cur.execute("SET search_path TO public;")

        await cur.execute(
            """
            SELECT room_number, date, COUNT(*) c
            FROM room_availability
            WHERE status = 'Booked'
            GROUP BY room_number, date
            HAVING COUNT(*) > 1;
            """,
        )
        rows = await cur.fetchall()
        double_booking_rows = len(rows)

        await cur.execute(
            "SELECT COUNT(*) FROM room_availability WHERE status IS NULL;",
        )
        null_row = await cur.fetchone()
        if null_row:
            null_status_count = int(null_row[0])

    no_double_booking = double_booking_rows == 0
    no_null_status = null_status_count == 0
    invariants_pass = no_double_booking and no_null_status

    return [
        {
            "key": "db_no_double_booking",
            "score": 1.0 if no_double_booking else 0.0,
            "comment": (
                "No double bookings detected."
                if no_double_booking
                else f"{double_booking_rows} double-booking rows detected."
            ),
        },
        {
            "key": "db_no_null_status",
            "score": 1.0 if no_null_status else 0.0,
            "comment": (
                "No null statuses detected."
                if no_null_status
                else f"{null_status_count} rows have NULL status."
            ),
        },
        {
            "key": "db_invariants_pass",
            "score": 1.0 if invariants_pass else 0.0,
            "comment": (
                "All DB invariants passed."
                if invariants_pass
                else "One or more DB invariants failed."
            ),
        },
    ]


async def _ensure_eval_pool(cfg: EvalConfig) -> AsyncConnectionPool[Any]:
    """Create or return the shared async connection pool for evaluator queries.

    Args:
        cfg: Evaluation configuration with DB URL.

    Returns:
        Open AsyncConnectionPool instance.

    """
    global _EVAL_POOL  # noqa: PLW0603
    if _EVAL_POOL is not None:
        return _EVAL_POOL

    async with _EVAL_POOL_LOCK:
        if _EVAL_POOL is not None:
            return _EVAL_POOL
        conninfo = await _get_eval_db_url(cfg)
        pool = AsyncConnectionPool(
            conninfo=conninfo,
            min_size=1,
            max_size=2,
            timeout=30,
            open=False,
        )
        await pool.open()
        _EVAL_POOL = pool
        return pool


async def _get_eval_db_url(cfg: EvalConfig) -> str:
    """Resolve the evaluation database URL.

    Args:
        cfg: Evaluation configuration with DB URL.

    Returns:
        Database URL string.

    Raises:
        RuntimeError: If no database URL is available.

    """
    env_url = cfg.pgsql_eval_db_url
    if env_url:
        return env_url

    if _load_app_config is None:
        msg = "Unable to import load_app_config for evaluator DB access."
        raise RuntimeError(msg) from _PGSQL_IMPORT_ERROR

    return _load_app_config().pgsql_db_url
