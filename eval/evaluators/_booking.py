"""Booking evaluators for Blue Horizon hotel agent.

This module provides evaluators for booking tool outcomes and database
invariant checks, ensuring booking operations behave correctly.

Booking writes are never performed by the model: it calls a ``propose_*``
tool, and only the eval/stress harness's auto-confirm helper (mirroring a
human clicking Confirm) commits the write via ``ProposalStore.confirm()``.
``EvalCaptureCallback`` captures both halves -- the ``propose_*`` tool call
and the injected ``confirm_booking`` outcome -- as tool-summary entries on
the same turn, which is what the functions below score against.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from psycopg_pool import AsyncConnectionPool

from eval._utils import json_value, truncate
from eval.db_invariants import find_overlapping_booking_rooms
from eval.evaluators._common import (
    _call_judge_llm,
    _get_example_turns,
    _iter_turn_outputs,
    _safe_json_loads,
)

if TYPE_CHECKING:
    from langsmith.schemas import Example, Run

    from eval.config import EvalConfig, EvaluatorLimitsConfig

try:  # Optional app import for DB URL lookup.
    from blue_horizon.config import load_app_config as _load_app_config

    _PGSQL_IMPORT_ERROR: Exception | None = None
except Exception as _exc:  # noqa: BLE001
    _load_app_config = None
    _PGSQL_IMPORT_ERROR = _exc

_EVAL_POOL_LOCK = asyncio.Lock()
_EVAL_POOL: AsyncConnectionPool[Any] | None = None

# Tool names whose calls count toward booking-turn tool-error/call counters.
_TRACKED_TOOLS: frozenset[str] = frozenset(
    {
        "run_sql",
        "propose_booking",
        "propose_cancellation",
        "propose_modification",
        "confirm_booking",
    },
)

# Maps a propose_* tool name to the proposal action it creates.
_PROPOSE_ACTIONS: dict[str, str] = {
    "propose_booking": "book",
    "propose_cancellation": "cancel",
    "propose_modification": "modify",
}

# Fragments of guest-facing language that claim a write already happened.
# This is a cheap pre-filter, not a verdict: it decides which turns are worth
# a judge call, since most turns match none of these and cost nothing. A
# match alone never flags a turn -- see `_score_unbacked_success_claims`,
# which sends every match to `_judge_claims_success` for adjudication, since
# a plain substring check cannot tell "your room is booked" from "I can't say
# the room is booked without a completed confirmation."
_UNBACKED_SUCCESS_PHRASES: tuple[str, ...] = (
    "is booked",
    "has been booked",
    "successfully booked",
    "booking is confirmed",
    "booking confirmed",
    "you're all set",
    "you are all set",
    "has been cancelled",
    "has been canceled",
    "successfully cancelled",
    "successfully canceled",
    "cancellation is confirmed",
    "cancellation confirmed",
    "has been modified",
    "has been updated",
    "successfully modified",
    "modification confirmed",
    "confirmation number",
)


@dataclass
class BookingOutcomeState:
    """Mutable aggregation state for booking outcome scoring.

    Attributes:
        counts: Aggregate counters across all evaluated booking turns.
        per_turn_outcomes: Per-turn pass-rate records used for turn-based
            summary reporting.
        unexpected_fail_details: Failure detail records captured for logging.

    """

    counts: dict[str, int]
    per_turn_outcomes: list[dict[str, Any]]
    unexpected_fail_details: list[dict[str, Any]]


async def eval_booking_outcome_and_invariants(
    run: Run,
    example: Example,
    *,
    cfg: EvalConfig,
) -> list[dict[str, Any]]:
    """Evaluate booking tool outcomes and DB invariants for booking turns.

    Args:
        run: LangSmith run object containing tool summaries.
        example: LangSmith example with per-turn ``expected_success`` flags.
        cfg: Evaluation configuration with DB URL and evaluator limits.

    Returns:
        List of LangSmith metric dicts for tool outcomes, unbacked success
        claims, and DB invariants.

    """
    outputs = run.outputs or {}
    turn_outputs = outputs.get("turn_outputs") or []
    has_rooms = any(
        t.get("route_pred") == "booking" for t in turn_outputs if isinstance(t, dict)
    )
    if not has_rooms:
        return [
            {
                "key": "booking_invariants_skipped",
                "score": 1.0,
                "comment": "No booking turns detected.",
            },
        ]

    outcome_metrics = _score_booking_tool_outcomes(run, example, cfg)
    unbacked_claim_metrics = await _score_unbacked_success_claims(run, cfg)
    invariants_results = await _check_booking_db_invariants(cfg)
    return [
        *outcome_metrics,
        *unbacked_claim_metrics,
        *invariants_results,
    ]


def _score_booking_tool_outcomes(
    run: Run,
    example: Example,
    cfg: EvalConfig,
) -> list[dict[str, Any]]:
    """Score booking tool outcomes against expected results.

    Args:
        run: LangSmith run object containing tool summaries.
        example: LangSmith example with per-turn ``expected_success`` flags.
        cfg: Evaluation configuration for evaluator limits.

    Returns:
        List of LangSmith metrics for tool errors, outcome match rate, and a
        soft rowcount presence check. Metrics with nothing to measure are
        omitted rather than defaulting to a passing score -- see
        ``_format_tool_error_metric`` / ``_format_outcome_match_metric``.

    """
    limits = cfg.evaluator_limits
    turn_outputs = _iter_turn_outputs(run)
    example_turns = _get_example_turns(example)

    state = BookingOutcomeState(
        counts=_init_booking_outcome_counts(),
        per_turn_outcomes=[],
        unexpected_fail_details=[],
    )

    for global_idx, turn in enumerate(turn_outputs):
        if turn.get("route_pred") != "booking":
            continue
        tool_summary = turn.get("tool_summary") or []
        if not isinstance(tool_summary, list):
            continue

        expected_success: bool | None = None
        if global_idx < len(example_turns):
            es = example_turns[global_idx].get("expected_success")
            if isinstance(es, bool):
                expected_success = es

        _accumulate_booking_turn(
            turn_idx=global_idx,
            tool_summary=tool_summary,
            state=state,
            expected_success=expected_success,
        )

    metrics = _format_booking_outcome_metrics(state)
    metrics.extend(_format_booking_outcome_detail_metrics(state, limits))
    return metrics


def _format_booking_outcome_metrics(state: BookingOutcomeState) -> list[dict[str, Any]]:
    """Build the score metrics (tool errors, outcome match, rowcount sanity).

    Args:
        state: Aggregated booking outcome state for one run.

    Returns:
        List of LangSmith score metric dicts.

    """
    metrics: list[dict[str, Any]] = []

    tool_error_result = _format_tool_error_metric(state.counts)
    if tool_error_result is not None:
        score, comment = tool_error_result
        metrics.append(
            {"key": "booking_tool_errors", "score": score, "comment": comment},
        )

    match_result = _format_outcome_match_metric(state.counts)
    if match_result is not None:
        score, comment = match_result
        metrics.append(
            {
                "key": "booking_no_unexpected_failure_rate",
                "score": score,
                "comment": comment,
            },
        )

    rowcount_score, rowcount_comment = _format_rowcount_metric(state.counts)
    metrics.append(
        {
            "key": "booking_rowcount_sanity",
            "score": rowcount_score,
            "comment": rowcount_comment,
        },
    )
    return metrics


def _format_booking_outcome_detail_metrics(
    state: BookingOutcomeState,
    limits: EvaluatorLimitsConfig,
) -> list[dict[str, Any]]:
    """Build the per-turn and unexpected-failure detail (non-score) metrics.

    Args:
        state: Aggregated booking outcome state for one run.
        limits: Evaluator limits, for the JSON truncation threshold.

    Returns:
        List of LangSmith value metric dicts; empty when there is nothing
        to report.

    """
    metrics: list[dict[str, Any]] = []
    if state.per_turn_outcomes:
        metrics.append(
            _json_detail_metric(
                key="booking_outcome_per_turn",
                data=state.per_turn_outcomes,
                max_len=limits.json_value_max,
            ),
        )
    if state.unexpected_fail_details:
        metrics.append(
            _json_detail_metric(
                key="booking_unexpected_failures",
                data=state.unexpected_fail_details,
                max_len=limits.json_value_max,
            ),
        )
    return metrics


def _json_detail_metric(
    *,
    key: str,
    data: list[dict[str, Any]],
    max_len: int,
) -> dict[str, Any]:
    """Build one JSON-value metric, truncating and flagging it if oversized.

    Args:
        key: LangSmith metric key.
        data: JSON-serializable detail records.
        max_len: Maximum serialized length before truncation.

    Returns:
        A LangSmith value metric dict.

    """
    raw = json_value(data)
    if len(raw) <= max_len:
        return {"key": key, "value": raw}
    return {
        "key": key,
        "value": json_value(data, max_len=max_len),
        "comment": "JSON truncated",
    }


def _init_booking_outcome_counts() -> dict[str, int]:
    """Initialize counters used for booking tool outcome scoring.

    Returns:
        Dict of integer counters keyed by metric name.

    """
    return {
        "tool_calls": 0,
        "tool_errors": 0,
        "rowcount_present": 0,
        "run_sql_calls": 0,
        "outcome_matches": 0,
        "unexpected_failures": 0,
    }


def _accumulate_booking_turn(
    *,
    turn_idx: int,
    tool_summary: list[object],
    state: BookingOutcomeState,
    expected_success: bool | None,
) -> None:
    """Accumulate outcome counts for a single booking turn.

    Args:
        turn_idx: Global turn index within the run.
        tool_summary: Tool summary entries for the turn.
        state: Mutable booking scoring state updated in place.
        expected_success: Whether the example expects this turn to complete
            a write. ``False`` means no write is expected and a failed
            propose/confirm should not be penalised.

    """
    for entry in tool_summary:
        if not isinstance(entry, dict) or entry.get("tool") not in _TRACKED_TOOLS:
            continue
        state.counts["tool_calls"] += 1
        # A propose_*/confirm_booking call correctly rejecting an invalid
        # request (unavailable dates, an already-committed booking, ...) sets
        # the same error status a genuine bug would. On a turn the dataset
        # expects to fail, that rejection is the tool doing its job, not a
        # defect -- only count it toward `booking_tool_errors` when the turn
        # was not expected to fail, mirroring `_score_outcome_match` below.
        if _has_tool_error(entry) and expected_success is not False:
            state.counts["tool_errors"] += 1
        if entry.get("tool") == "run_sql":
            state.counts["run_sql_calls"] += 1
            if entry.get("rowcount") is not None:
                state.counts["rowcount_present"] += 1

    outcome_matched, failure_detail = _score_outcome_match(
        turn_idx=turn_idx,
        tool_summary=tool_summary,
        expected_success=expected_success,
    )
    if outcome_matched is None:
        return

    state.per_turn_outcomes.append(
        {"turn_index": turn_idx, "pass_rate": 1.0 if outcome_matched else 0.0},
    )
    if outcome_matched:
        state.counts["outcome_matches"] += 1
        return
    state.counts["unexpected_failures"] += 1
    if failure_detail is not None:
        state.unexpected_fail_details.append(failure_detail)


def _has_tool_error(summary: dict[str, Any]) -> bool:
    """Detect whether a tool summary indicates an error.

    Args:
        summary: Tool summary dict for a run_sql, propose_*, or
            confirm_booking call.

    Returns:
        True if the summary indicates error status or includes an error message.

    """
    status = summary.get("status")
    if isinstance(status, str) and status.lower() == "error":
        return True
    return bool(summary.get("error"))


def _score_outcome_match(
    *,
    turn_idx: int,
    tool_summary: list[object],
    expected_success: bool | None,
) -> tuple[bool | None, dict[str, Any] | None]:
    """Score the propose+confirm outcome for a single booking turn.

    Args:
        turn_idx: Global turn index within the run.
        tool_summary: Tool summary entries for the turn.
        expected_success: Whether the example expects this turn to complete
            a write. When ``False``, a failed propose/confirm is not penalised.

    Returns:
        Tuple of ``(outcome_matched, failure_detail)``. Returns ``(None,
        None)`` when the turn made no ``propose_*`` call -- a pure search
        turn has nothing here to score.

    """
    action, propose_entry, confirm_entry = _detect_propose_confirm(tool_summary)
    if action is None or propose_entry is None:
        return None, None
    success, detail = _score_atomic_outcome(
        action=action, propose_entry=propose_entry, confirm_entry=confirm_entry,
    )
    if success:
        return True, None
    if expected_success is False:
        return True, None
    return False, {"turn": turn_idx, "action": action, **detail}


def _detect_propose_confirm(
    tool_summary: list[object],
) -> tuple[str | None, dict[str, Any] | None, dict[str, Any] | None]:
    """Find the propose_* and confirm_booking entries for a turn, if any.

    A turn has at most one pending proposal (the store supersedes any prior
    one), so the *last* ``propose_*`` entry and the *last* ``confirm_booking``
    entry are the ones that pair together.

    Args:
        tool_summary: Tool summary entries for the turn.

    Returns:
        Tuple of ``(action, propose_entry, confirm_entry)``. ``action`` and
        ``propose_entry`` are ``None`` when no ``propose_*`` call was made
        this turn; ``confirm_entry`` is ``None`` when the proposal was never
        confirmed.

    """
    action: str | None = None
    propose_entry: dict[str, Any] | None = None
    confirm_entry: dict[str, Any] | None = None
    for entry in tool_summary:
        if not isinstance(entry, dict):
            continue
        tool = entry.get("tool")
        if tool in _PROPOSE_ACTIONS:
            action = _PROPOSE_ACTIONS[tool]
            propose_entry = entry
        elif tool == "confirm_booking":
            confirm_entry = entry
    return action, propose_entry, confirm_entry


def _score_atomic_outcome(
    *,
    action: str,
    propose_entry: dict[str, Any],
    confirm_entry: dict[str, Any] | None,
) -> tuple[bool, dict[str, Any]]:
    """Score a single propose+confirm pair.

    `commit_booking`/`cancel_booking`/`modify_booking` are each atomic --
    they either fully succeed or raise before writing anything -- so there is
    no partial-success case left to model here, unlike the old SQL-CTE
    era's nights-requested/nights-booked counts.

    Args:
        action: Proposal action (`book`, `cancel`, or `modify`).
        propose_entry: The turn's `propose_*` tool summary entry.
        confirm_entry: The turn's `confirm_booking` entry, or ``None`` if the
            proposal was never confirmed (for example, a harness-level
            abandonment case).

    Returns:
        Tuple of (success, details) where details are safe for logging.

    """
    _ = action
    if propose_entry.get("status") == "error":
        return False, {"stage": "propose", "error": propose_entry.get("error")}
    if confirm_entry is None:
        return False, {"stage": "confirm", "note": "proposal was never confirmed"}
    if confirm_entry.get("status") == "error":
        return False, {"stage": "confirm", "error": confirm_entry.get("error")}
    return True, {"already_confirmed": bool(confirm_entry.get("already_confirmed"))}


def _format_tool_error_metric(counts: dict[str, int]) -> tuple[float, str] | None:
    """Format the tool error metric from accumulated counts.

    Args:
        counts: Counter dict accumulated across booking turns.

    Returns:
        Tuple of (score, comment) for booking_tool_errors, or ``None`` when
        no run_sql/propose/confirm calls were observed -- omitted rather
        than defaulting to a passing score, since a broken capture path and
        a genuinely error-free run would otherwise be indistinguishable.

    """
    tool_calls = counts["tool_calls"]
    if tool_calls == 0:
        return None
    tool_errors = counts["tool_errors"]
    if tool_errors == 0:
        return 1.0, "No tool errors detected."
    return 0.0, f"{tool_errors} tool error(s) detected across {tool_calls} call(s)."


def _format_outcome_match_metric(counts: dict[str, int]) -> tuple[float, str] | None:
    """Format the outcome match metric from accumulated counts.

    Args:
        counts: Counter dict accumulated across booking turns.

    Returns:
        Tuple of (score, comment) for booking_no_unexpected_failure_rate, or
        ``None`` when no propose/confirm outcomes were observed -- omitted
        rather than defaulting to a passing score. See
        ``_format_tool_error_metric`` for why a silent 1.0 here is exactly
        the failure mode this scoring was rewritten to avoid.

    """
    matches = counts["outcome_matches"]
    unexpected = counts["unexpected_failures"]
    total = matches + unexpected
    if total == 0:
        return None
    return matches / total, f"Outcomes matched {matches}/{total}."


def _format_rowcount_metric(counts: dict[str, int]) -> tuple[float, str]:
    """Format the rowcount presence metric from accumulated counts.

    Args:
        counts: Counter dict accumulated across booking turns.

    Returns:
        Tuple of (score, comment) for booking_rowcount_sanity.

    """
    run_sql_calls = counts["run_sql_calls"]
    rowcount_present = counts["rowcount_present"]
    if run_sql_calls == 0:
        return 1.0, "No run_sql calls observed."
    score = rowcount_present / run_sql_calls
    comment = f"Rowcount present on {rowcount_present}/{run_sql_calls} run_sql calls."
    return score, comment


async def _score_unbacked_success_claims(
    run: Run, cfg: EvalConfig,
) -> list[dict[str, Any]]:
    """Flag assistant turns that claim a write succeeded with none behind it.

    This is the failure Solution 3 exists to prevent, so unlike the outcome
    metrics above it deliberately does **not** consult ``expected_success``:
    an unbacked claim on a turn where failure was expected is exactly the
    case most likely to hide a regression, not a case to wave through.

    A plain substring match cannot separate "your room is booked" from "I
    can't say the room is booked without a completed confirmation" -- both
    contain the phrase, only one is a claim. So phrase matching here is only
    a cheap pre-filter (most turns match nothing, so most turns cost no LLM
    call); every match is adjudicated by ``_judge_claims_success`` before it
    counts against the score.

    Args:
        run: LangSmith run object containing tool summaries and per-turn
            assistant text.
        cfg: Evaluation configuration with the judge model to adjudicate
            phrase matches.

    Returns:
        List with one score metric, plus a details metric when any turn is
        flagged. Empty when no booking turn had text to scan.

    """
    turn_outputs = _iter_turn_outputs(run)
    candidates: list[tuple[int, str]] = []
    scanned = 0

    for idx, turn in enumerate(turn_outputs):
        if turn.get("route_pred") != "booking":
            continue
        text = turn.get("assistant_text")
        if not isinstance(text, str) or not text:
            continue
        scanned += 1
        if not _mentions_success_phrase(text.lower()):
            continue
        tool_summary = turn.get("tool_summary")
        if _has_backing_confirm(tool_summary if isinstance(tool_summary, list) else []):
            continue
        candidates.append((idx, text))

    if scanned == 0:
        return []

    flagged: list[dict[str, Any]] = []
    for idx, text in candidates:
        claims_success, rationale = await _judge_claims_success(text, cfg.judge.model)
        if claims_success:
            flagged.append(
                {
                    "turn": idx,
                    "assistant_text": text[:280],
                    "judge_rationale": rationale,
                },
            )

    score = 0.0 if flagged else 1.0
    comment = (
        "No unbacked success claims detected."
        if not flagged
        else f"{len(flagged)} turn(s) claimed success with no write behind it."
    )
    metrics: list[dict[str, Any]] = [
        {
            "key": "booking_no_unbacked_success_claims",
            "score": score,
            "comment": comment,
        },
    ]
    if flagged:
        metrics.append(
            {"key": "booking_unbacked_success_details", "value": json_value(flagged)},
        )
    return metrics


def _mentions_success_phrase(lowered_text: str) -> bool:
    """Cheaply check whether text contains any success-claim phrase.

    A pre-filter only, not a verdict -- see `_score_unbacked_success_claims`.

    Args:
        lowered_text: Lowercased assistant turn text.

    Returns:
        True if any phrase in `_UNBACKED_SUCCESS_PHRASES` appears in the text.

    """
    return any(phrase in lowered_text for phrase in _UNBACKED_SUCCESS_PHRASES)


def _has_backing_confirm(tool_summary: list[object]) -> bool:
    """Check whether a turn's tool summary contains a successful confirm.

    Args:
        tool_summary: Tool summary entries for one turn.

    Returns:
        True if a `confirm_booking` entry with `status == "ok"` is present.

    """
    return any(
        isinstance(entry, dict)
        and entry.get("tool") == "confirm_booking"
        and entry.get("status") == "ok"
        for entry in tool_summary
    )


async def _judge_claims_success(text: str, model: str) -> tuple[bool, str]:
    """Ask the judge whether one assistant turn claims a write already succeeded.

    Args:
        text: Assistant turn text to adjudicate (already matched a
            `_UNBACKED_SUCCESS_PHRASES` pre-filter).
        model: Judge model name.

    Returns:
        tuple[bool, str]: Whether the turn claims success, and a short
        rationale. Any judge failure -- an unavailable model, an unparseable
        response, an invalid payload -- returns `(True, <reason>)`: an
        unbacked claim is exactly the failure this metric exists to catch,
        so an inconclusive adjudication is flagged rather than waved through.

    """
    prompt = (
        "You are checking one message from a hotel booking assistant for a "
        "specific failure: claiming that a booking, cancellation, or "
        "modification has already succeeded (e.g. stating a room is booked, "
        "giving a confirmation number, saying a cancellation went through) "
        "when nothing has actually been confirmed by the guest yet. The "
        "assistant may only *propose* a booking for the guest to review -- "
        "it never commits one itself. A refusal to confirm, an explanation "
        "that the guest must review or confirm first, or a description of "
        "what *would* happen, is NOT a success claim, even if it repeats "
        "phrases like 'confirmation number' while declining to give one.\n\n"
        "Output STRICT JSON only, no markdown fences, no prose:\n"
        '{"claims_success": true|false, "rationale": "<=30 words"}\n\n'
        f"Message:\n{truncate(text, 2000)}\n"
    )
    try:
        raw_text = await _call_judge_llm(prompt=prompt, model=model)
    except Exception as exc:  # noqa: BLE001
        reason = (
            f"Judge call failed, flagging conservatively: {truncate(str(exc), 150)}"
        )
        return True, reason

    parsed = _safe_json_loads(raw_text)
    unparseable_reason = (
        "Judge returned unparseable output, flagging conservatively. "
        f"Raw: {truncate(raw_text, 150)}"
    )
    if not isinstance(parsed, dict):
        return True, unparseable_reason

    claims_success = parsed.get("claims_success")
    if not isinstance(claims_success, bool):
        return True, unparseable_reason

    rationale = parsed.get("rationale")
    return claims_success, rationale if isinstance(rationale, str) else ""


async def _check_booking_db_invariants(cfg: EvalConfig) -> list[dict[str, Any]]:
    """Check booking DB invariants against bookings/booking_rooms/room_availability.

    ``db_no_double_booking`` keeps its historical key name for baseline
    continuity, but its query changed completely: the old
    ``room_availability`` grouped-by-``(room_number, date)`` check was a
    tautology (that table carries a ``UNIQUE (room_id, date)`` constraint, so
    the query could never return a row) and always scored 1.0 regardless of
    what the agent did. It now checks for real overlapping `booking_rooms`
    ranges on the same room via
    ``eval.db_invariants.find_overlapping_booking_rooms`` -- the same helper
    ``eval.stress.db._check_invariants`` calls, so both checks can never
    drift out of sync with each other.

    The overlap check is safe to run globally (unscoped to one example or
    customer) even though eval examples run concurrently: `commit_booking`
    and `modify_booking` both take `FOR UPDATE` locks before booking a night,
    so two concurrently running examples can never legitimately produce an
    overlap. If this check ever fires, it is a genuine `write_ops` bug
    regardless of which example's evaluator run happened to observe it, not
    a false positive from concurrency.

    Args:
        cfg: Evaluation configuration with DB URL.

    Returns:
        List of LangSmith metric dicts for DB invariants.

    """
    pool = await _ensure_eval_pool(cfg)

    async with pool.connection() as conn, conn.cursor() as cur:
        await cur.execute("SET search_path TO public;")

        overlap_rows = await find_overlapping_booking_rooms(cur)

        await cur.execute(
            "SELECT COUNT(*) FROM room_availability WHERE status IS NULL;",
        )
        null_status_row = await cur.fetchone()
        null_status_count = int(null_status_row[0]) if null_status_row else 0

        await cur.execute(
            """
            SELECT confirmation_number, COUNT(*) AS c
            FROM bookings
            WHERE confirmation_number IS NOT NULL
            GROUP BY confirmation_number
            HAVING COUNT(*) > 1;
            """,
        )
        duplicate_confirmation_rows = await cur.fetchall()

        await cur.execute(
            """
            SELECT booking_id FROM bookings
            WHERE status != 'cancelled' AND confirmation_number IS NULL;
            """,
        )
        missing_confirmation_rows = await cur.fetchall()

        await cur.execute(
            "SELECT room_id, date FROM room_availability "
            "WHERE status = 'Available' AND price IS NULL;",
        )
        null_price_rows = await cur.fetchall()

    checks: list[tuple[str, int, str]] = [
        ("db_no_double_booking", len(overlap_rows), "overlapping room-stay pair(s)"),
        ("db_no_null_status", null_status_count, "row(s) have NULL status"),
        (
            "db_confirmation_numbers_unique",
            len(duplicate_confirmation_rows),
            "duplicated confirmation number(s)",
        ),
        (
            "db_confirmed_bookings_have_receipt",
            len(missing_confirmation_rows),
            "non-cancelled booking(s) missing a confirmation number",
        ),
        (
            "db_no_available_with_null_price",
            len(null_price_rows),
            "'Available' room-night(s) with no price",
        ),
    ]

    results: list[dict[str, Any]] = []
    all_pass = True
    for key, count, noun in checks:
        passed = count == 0
        all_pass = all_pass and passed
        results.append(
            {
                "key": key,
                "score": 1.0 if passed else 0.0,
                "comment": f"No {noun} detected." if passed else f"{count} {noun}.",
            },
        )

    results.append(
        {
            "key": "db_invariants_pass",
            "score": 1.0 if all_pass else 0.0,
            "comment": (
                "All DB invariants passed."
                if all_pass
                else "One or more DB invariants failed."
            ),
        },
    )
    return results


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

    Invariant checks read `bookings`/`booking_rooms`, which are only
    reachable through the read-write role (`bh_agent_rw`) -- unlike
    `run_sql`, they are deliberately off the read-only role's grants. So
    this always resolves the read-write URL, never `pgsql_ro_eval_db_url`.

    Args:
        cfg: Evaluation configuration with DB URL.

    Returns:
        Database URL string.

    Raises:
        RuntimeError: If no database URL is available.

    """
    env_url = cfg.pgsql_rw_eval_db_url
    if env_url:
        return env_url

    if _load_app_config is None:
        msg = "Unable to import load_app_config for evaluator DB access."
        raise RuntimeError(msg) from _PGSQL_IMPORT_ERROR

    return _load_app_config().pgsql_rw_db_url
