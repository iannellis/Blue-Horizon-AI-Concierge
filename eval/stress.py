"""Stress and race testing suite for the LangGraph hotel agent.

This module resets the Neon development branch to a clean baseline, discovers
bookable availability targets, and then drives concurrent simulated users
through the orchestrator.  Each user executes a fixed number of operations
whose types are sampled from configurable weights (BOOK / MODIFY / CANCEL).
State constraints enforce a single-booking-per-user invariant: a user with no
active booking always BOOKs; a user who already holds a booking and draws BOOK
is promoted to MODIFY instead.  Per-operation and per-user JSONL artifacts are
written, summary statistics and database invariants are computed, and LangSmith
traces are emitted via tags and metadata.

"""

from __future__ import annotations

import asyncio
import json
import logging
import platform
import random
import statistics
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

import psycopg
from dotenv import load_dotenv
from psycopg_pool import PoolTimeout

from blue_horizon.config import load_app_config
from eval._utils import truncate as _truncate
from eval.config import NeonConfig, load_eval_config
from eval.langsmith_target import EvalCaptureCallback, OrchestrationManager
from eval.rooms_db_manager import open_schema_pool, reset_neon_branch

if TYPE_CHECKING:
    from pathlib import Path

    from psycopg_pool import AsyncConnectionPool

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StressRunConfig:
    """Configuration for a single stress run.

    Attributes:
        users: The number of concurrent simulated users.
        ops_per_user: The number of operations per user.
        max_concurrency: The maximum concurrent users allowed at once.
        output_dir: The base output directory for artifacts.
        log_dir: The directory for stress run log files.
        stay_nights: The length of stay in nights for generated targets.
        num_targets: The number of available targets to precompute.
        hot_target_count: The size of the hot contention target subset.
        hot_target_probability: Probability of selecting a hot target vs any target.
        start_date: The search start date for available targets.
        horizon_days: The search horizon, in days, for available targets.
        pool_max: The maximum connection pool size.
        db_retry_attempts: Retry attempts on transient DB connection errors.
        db_retry_delay_s: Base delay in seconds between DB retry attempts.
        reconcile_max_detail: Maximum entries in reconciliation failure detail lists.
        book_weight: Relative weight for BOOK operations in the op-type mix.
        modify_weight: Relative weight for MODIFY operations in the op-type mix.
        cancel_weight: Relative weight for CANCEL operations in the op-type mix.
        db_url: The Postgres connection URL.

    """

    users: int
    ops_per_user: int
    max_concurrency: int
    output_dir: Path
    log_dir: Path
    stay_nights: int
    num_targets: int
    hot_target_count: int
    hot_target_probability: float
    start_date: date
    horizon_days: int
    pool_max: int
    db_retry_attempts: int
    db_retry_delay_s: float
    reconcile_max_detail: int
    book_weight: float
    modify_weight: float
    cancel_weight: float
    db_url: str


@dataclass
class UserState:
    """Track best-effort booking state for a simulated user thread.

    Attributes:
        last_room_number: The last room number used by the user, if any.
        last_check_in: The last check-in date as an ISO string, if any.
        last_check_out: The last check-out date as an ISO string, if any.
        has_booking: Whether the user likely has an active booking.

    """

    last_room_number: int | None = None
    last_check_in: str | None = None
    last_check_out: str | None = None
    has_booking: bool = False


@dataclass(frozen=True)
class OperationBuildResult:
    """Container for a prepared operation prompt and context.

    Attributes:
        op_type: The requested operation type label.
        prompt: The user prompt to send to the agent.
        target_used: The target dict used for the operation, if any.
        old_room: The prior room number before the operation.
        old_check_in: The prior check-in date before the operation.
        old_check_out: The prior check-out date before the operation.

    """

    op_type: str
    prompt: str
    target_used: dict[str, object] | None
    old_room: int | None
    old_check_in: str | None
    old_check_out: str | None


def _load_config() -> StressRunConfig:
    """Load stress configuration from eval_config.toml.

    Returns:
        A fully populated ``StressRunConfig`` instance.

    Raises:
        RuntimeError: If a database URL cannot be determined.

    """
    cfg = load_eval_config().stress

    db_url = load_eval_config().pgsql_eval_db_url or load_app_config().pgsql_db_url
    if not db_url:
        msg = "Database URL is required but was not found"
        raise RuntimeError(msg)

    return StressRunConfig(
        users=cfg.workload.users,
        ops_per_user=cfg.workload.ops_per_user,
        max_concurrency=cfg.workload.max_concurrency,
        output_dir=cfg.output.output_dir,
        log_dir=cfg.output.log_dir,
        stay_nights=cfg.targets.stay_nights,
        num_targets=cfg.targets.num_targets,
        hot_target_count=cfg.targets.hot_target_count,
        hot_target_probability=cfg.targets.hot_target_probability,
        start_date=cfg.targets.start_date,
        horizon_days=cfg.targets.horizon_days,
        pool_max=cfg.db.pool_max,
        db_retry_attempts=cfg.db.db_retry_attempts,
        db_retry_delay_s=cfg.db.db_retry_delay_s,
        reconcile_max_detail=cfg.db.reconcile_max_detail,
        book_weight=cfg.workload.book_weight,
        modify_weight=cfg.workload.modify_weight,
        cancel_weight=cfg.workload.cancel_weight,
        db_url=db_url,
    )


async def _start_orchestration() -> OrchestrationManager:
    """Start the orchestrator and wait for readiness with a configured timeout.

    The timeout is read from ``eval_config.toml`` via
    ``orchestration.ready_timeout_s``.

    Returns:
        A ready ``OrchestrationManager`` instance.

    Raises:
        TimeoutError: If readiness is not reached within the configured timeout.

    """
    ready_timeout_s = load_eval_config().orchestration.ready_timeout_s
    orchestration = OrchestrationManager(prune_tool_messages=False)
    await orchestration.start()

    ready_deadline = time.perf_counter() + ready_timeout_s
    while not orchestration.is_ready:
        if time.perf_counter() > ready_deadline:
            msg = (
                f"OrchestrationManager did not become ready"
                f" within {ready_timeout_s:.0f}s"
            )
            raise TimeoutError(msg)
        await asyncio.sleep(0.1)

    return orchestration


async def _set_search_path(conn: object, schema: str) -> None:
    """Set the connection's search path to the target schema safely.

    Args:
        conn: An async psycopg connection-like object.
        schema: The schema name to place at the front of ``search_path``.

    """
    # Use set_config to avoid interpolating identifiers in SQL text.
    conn_obj = cast("object", conn)
    await cast("object", conn_obj).execute(  # type: ignore[attr-defined]
        "SELECT set_config('search_path', %s, false)",
        (schema,),
    )


async def find_available_targets(
    conn: object,
    *,
    num_targets: int,
    stay_nights: int,
    start_date: date,
    horizon_days: int,
) -> list[dict[str, object]]:
    """Find random bookable targets with contiguous availability and pricing.

    A target is considered bookable only if all nights in the stay are marked
    ``status='Available'`` and ``price IS NOT NULL``. The SQL uses an islands
    and gaps pattern to discover contiguous spans, and Python then selects a
    random check-in that fits within each span.

    Args:
        conn: An async psycopg connection-like object.
        num_targets: The maximum number of candidate spans to sample.
        stay_nights: The fixed stay length in nights.
        start_date: The inclusive lower bound for candidate dates.
        horizon_days: The size of the search window in days from ``start_date``.

    Returns:
        A list of dictionaries with keys ``room_number``, ``check_in``, and
        ``check_out`` (ISO-8601 date strings).

    """
    sql = """
    WITH avail AS (
      SELECT room_number, date
      FROM room_availability
      WHERE status = 'Available'
        AND price IS NOT NULL
        AND date >= %s::date
        AND date < (%s::date + (%s::int || ' days')::interval)
    ),
    seq AS (
      SELECT
        room_number,
        date,
        (
          date
          - (ROW_NUMBER() OVER (PARTITION BY room_number ORDER BY date))
            * INTERVAL '1 day'
        ) AS grp
      FROM avail
    ),
    spans AS (
      SELECT
        room_number,
        MIN(date) AS span_start,
        MAX(date) AS span_end,
        COUNT(*) AS span_len
      FROM seq
      GROUP BY room_number, grp
      HAVING COUNT(*) >= %s
    )
    SELECT room_number, span_start, span_end
    FROM spans
    ORDER BY RANDOM()
    LIMIT %s;
    """

    rows: list[tuple[int, date, date]] = []
    conn_obj = cast("object", conn)
    async with cast("object", conn_obj).cursor() as cur:  # type: ignore[attr-defined]
        await cur.execute(
            sql,
            (start_date, start_date, horizon_days, stay_nights, num_targets),
        )
        rows = await cur.fetchall()

    targets: list[dict[str, object]] = []
    for room_number, span_start, span_end in rows:
        latest_start = span_end - timedelta(days=stay_nights - 1)
        if latest_start < span_start:
            continue
        day_span = (latest_start - span_start).days
        offset = random.randint(0, day_span) if day_span > 0 else 0  # noqa: S311
        check_in = span_start + timedelta(days=offset)
        check_out = check_in + timedelta(days=stay_nights)
        targets.append(
            {
                "room_number": int(room_number),
                "check_in": check_in.isoformat(),
                "check_out": check_out.isoformat(),
            },
        )

    return targets


def _is_transient_db_error(exc: BaseException) -> bool:
    """Return True if the exception looks like a transient DB connection failure.

    Args:
        exc: The exception raised by psycopg or psycopg_pool.

    Returns:
        True if the exception is a known transient connection error.

    """
    if isinstance(exc, (PoolTimeout, TimeoutError)):
        return True
    msg = str(exc).lower()
    patterns = (
        "ssl connection has been closed unexpectedly",
        "server closed the connection unexpectedly",
        "connection is closed",
        "connection not open",
        "terminating connection",
    )
    return any(p in msg for p in patterns)


async def _init_branch_and_targets(
    pool: AsyncConnectionPool,
    cfg: StressRunConfig,
    neon_cfg: NeonConfig,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Reset the Neon branch and discover bookable contention targets.

    Resets the configured Neon branch to its parent baseline, then queries
    the database to find bookable availability spans for the workload.

    Args:
        pool: The open database connection pool pointing at the Neon branch.
        cfg: The stress run configuration.
        neon_cfg: Neon project and branch settings from eval config.

    Returns:
        A tuple of ``(targets, hot_targets)``.

    Raises:
        RuntimeError: If no bookable targets can be found after the reset.

    """
    logger.info(
        "Resetting Neon branch %r in project %r for stress run.",
        neon_cfg.branch_name,
        neon_cfg.project_id,
    )
    await reset_neon_branch(
        project_id=neon_cfg.project_id,
        branch_name=neon_cfg.branch_name,
    )

    targets: list[dict[str, object]] = []
    for attempt in range(cfg.db_retry_attempts):
        try:
            async with pool.connection() as conn:
                await _set_search_path(conn, "public")
                targets = await find_available_targets(
                    conn,
                    num_targets=cfg.num_targets,
                    stay_nights=cfg.stay_nights,
                    start_date=cfg.start_date,
                    horizon_days=cfg.horizon_days,
                )
            break
        except (
            psycopg.OperationalError,
            psycopg.InterfaceError,
            PoolTimeout,
            TimeoutError,
        ) as exc:
            if attempt < cfg.db_retry_attempts - 1 and _is_transient_db_error(exc):
                logger.warning(
                    "Transient DB error fetching targets (attempt %d/%d); retrying.",
                    attempt + 1,
                    cfg.db_retry_attempts,
                    exc_info=True,
                )
                await asyncio.sleep(cfg.db_retry_delay_s * (2**attempt))
            else:
                raise

    if not targets:
        msg = "No bookable targets found"
        raise RuntimeError(msg)

    hot_targets = targets[: max(0, min(cfg.hot_target_count, len(targets)))]
    logger.info(
        "Branch reset complete: %d targets available (%d hot).",
        len(targets),
        len(hot_targets),
    )
    return targets, hot_targets


def _pick_op_type(cfg: StressRunConfig) -> str:
    """Sample an operation type using the configured weights.

    Args:
        cfg: The stress run configuration supplying op-type weights.

    Returns:
        The chosen operation type string.

    """
    return random.choices(  # noqa: S311
        population=["BOOK", "MODIFY", "CANCEL"],
        weights=[cfg.book_weight, cfg.modify_weight, cfg.cancel_weight],
        k=1,
    )[0]


def _choose_target(
    targets: list[dict[str, object]],
    hot_targets: list[dict[str, object]],
    hot_target_probability: float,
) -> dict[str, object]:
    """Choose a contention target with hot-to-cold probability split.

    Args:
        targets: The full candidate target pool.
        hot_targets: The high-contention subset.
        hot_target_probability: Probability of selecting hot vs any target.

    Returns:
        A chosen target dictionary.

    """
    use_hot = bool(hot_targets) and random.random() < hot_target_probability  # noqa: S311
    pool = hot_targets if use_hot else targets
    return random.choice(pool)  # noqa: S311


def _choose_new_target(  # noqa: PLR0913
    *,
    targets: list[dict[str, object]],
    hot_targets: list[dict[str, object]],
    hot_target_probability: float,
    old_room: int | None,
    old_check_in: str | None,
    old_check_out: str | None,
) -> dict[str, object]:
    """Choose a new target that differs from the previous booking if possible.

    Applies the same hot/cold probability split as ``_choose_target`` before
    searching for a candidate that differs from the current booking.

    Args:
        targets: The full candidate target pool.
        hot_targets: The high-contention subset.
        hot_target_probability: Probability of selecting from hot targets.
        old_room: The prior room number, if known.
        old_check_in: The prior check-in date, if known.
        old_check_out: The prior check-out date, if known.

    Returns:
        A chosen target dictionary, preferring a different triplet.

    """
    use_hot = bool(hot_targets) and random.random() < hot_target_probability  # noqa: S311
    pool = hot_targets if use_hot else targets
    candidates = list(pool)
    random.shuffle(candidates)
    for cand in candidates:
        if (
            cand.get("room_number") != old_room
            or cand.get("check_in") != old_check_in
            or cand.get("check_out") != old_check_out
        ):
            return cand
    return random.choice(targets)  # noqa: S311


def _build_operation_request(
    op_type: str,
    state: UserState,
    *,
    targets: list[dict[str, object]],
    hot_targets: list[dict[str, object]],
    hot_target_probability: float,
) -> OperationBuildResult:
    """Build a prompt for the given operation and update user state.

    Two op-type promotions keep the single-booking-per-user invariant intact:

    * ``"BOOK"`` with an active booking → promoted to ``"MODIFY"``.
    * ``"MODIFY"`` or ``"CANCEL"`` with no active booking → normalised to
      ``"BOOK"``.

    In both cases ``OperationBuildResult.op_type`` reflects the effective
    operation that was actually executed.

    Args:
        op_type: The requested operation type label.
        state: The mutable per-user state to update.
        targets: The full target pool.
        hot_targets: The hot contention target subset.
        hot_target_probability: Probability of selecting hot vs any target.

    Returns:
        A structured operation build result.

    """
    if op_type == "BOOK" and state.has_booking:
        op_type = "MODIFY"

    old_room = state.last_room_number
    old_check_in = state.last_check_in
    old_check_out = state.last_check_out

    if op_type == "BOOK" or not state.last_room_number:
        op_type = "BOOK"  # normalise: MODIFY/CANCEL with no booking falls back to BOOK
        target_used = _choose_target(targets, hot_targets, hot_target_probability)
        book_room = cast("int", target_used["room_number"])
        book_check_in = target_used["check_in"]
        book_check_out = target_used["check_out"]
        prompt = (
            f"Book room {book_room} from {book_check_in} "
            f"to {book_check_out} for 2 adults."
        )
        state.last_room_number = book_room
        state.last_check_in = str(book_check_in)
        state.last_check_out = str(book_check_out)
        return OperationBuildResult(
            op_type=op_type,
            prompt=prompt,
            target_used=target_used,
            old_room=old_room,
            old_check_in=old_check_in,
            old_check_out=old_check_out,
        )

    if op_type == "MODIFY":
        new_target = _choose_new_target(
            targets=targets,
            hot_targets=hot_targets,
            hot_target_probability=hot_target_probability,
            old_room=old_room,
            old_check_in=old_check_in,
            old_check_out=old_check_out,
        )
        new_room = cast("int", new_target["room_number"])
        new_check_in = new_target["check_in"]
        new_check_out = new_target["check_out"]
        modify_old = (
            f"Modify my reservation for room {old_room} "
            f"from {old_check_in} to {old_check_out}. "
        )
        modify_new = (
            f"Change it to room {new_room} from {new_check_in} "
            f"to {new_check_out} for 2 adults."
        )
        prompt = f"{modify_old}{modify_new}"
        state.last_room_number = new_room
        state.last_check_in = str(new_check_in)
        state.last_check_out = str(new_check_out)
        return OperationBuildResult(
            op_type=op_type,
            prompt=prompt,
            target_used=new_target,
            old_room=old_room,
            old_check_in=old_check_in,
            old_check_out=old_check_out,
        )

    cancel_text = (
        f"Cancel my booking for room {old_room} from {old_check_in} to {old_check_out}."
    )
    target_used = {
        "room_number": old_room,
        "check_in": old_check_in,
        "check_out": old_check_out,
    }
    return OperationBuildResult(
        op_type=op_type,
        prompt=cancel_text,
        target_used=target_used,
        old_room=old_room,
        old_check_in=old_check_in,
        old_check_out=old_check_out,
    )


def _build_trace_context(
    *,
    run_id: str,
    user_idx: int,
    op_idx: int,
    op_type: str,
) -> tuple[list[str], dict[str, object]]:
    """Build LangSmith tags and metadata for a stress operation.

    Args:
        run_id: The unique identifier for the enclosing stress run, used to
            group all traces from the same invocation in LangSmith.
        user_idx: The user index for the operation.
        op_idx: The operation index for the user.
        op_type: The operation type label.

    Returns:
        A tuple of ``(tags, metadata)``.

    """
    tags = [
        "stress",
        f"run:{run_id}",
        f"user:{user_idx}",
        f"op:{op_idx}",
        f"op_type:{op_type}",
    ]
    metadata = {
        "run_id": run_id,
        "user_idx": user_idx,
        "op_idx": op_idx,
        "op_type": op_type,
    }
    return tags, metadata


async def _extract_last_assistant_text(result: object) -> str:
    """Extract the last assistant/AI message text from a graph result.

    The result structure varies across orchestrator and message types, so this
    function makes a best effort to find an assistant message and normalize
    list-based content.

    Args:
        result: The orchestration result object, typically a dict.

    Returns:
        The extracted assistant text, or an empty string if not found.

    """
    try:
        messages_obj: Sequence[object]
        if isinstance(result, Mapping):
            messages_obj = cast("Sequence[object]", result.get("messages", []))
        else:
            messages_obj = []
        for msg in reversed(messages_obj):
            role = getattr(msg, "type", None) or getattr(msg, "role", None)
            if role in {"ai", "assistant"}:
                content = getattr(msg, "content", "")
                if isinstance(content, list):
                    parts: list[str] = []
                    for part in content:
                        if isinstance(part, dict) and "text" in part:
                            parts.append(str(part["text"]))
                        else:
                            parts.append(str(part))
                    return " ".join(parts).strip()
                return str(content).strip()
    except (AttributeError, KeyError, TypeError, ValueError):
        return ""
    return ""


def _classify_outcome(text: str) -> str:
    """Classify an assistant response into a coarse outcome bucket.

    This heuristic is used only for stress-test statistics.

    Args:
        text: The assistant's response text.

    Returns:
        One of ``"success"``, ``"conflict"``, or ``"error"``.

    """
    lower = text.lower().replace("\u2019", "'").replace("\u2018", "'")
    conflict_markers = [
        "unavailable",
        "no availability",
        "already booked",
        "couldn't",
        "cannot",
        "not available",
        "isn't available",
        "i can't",
    ]
    if any(m in lower for m in conflict_markers):
        return "conflict"
    if any(m in lower for m in ["error", "failed", "exception", "try again"]):
        return "error"
    return "success"


async def _invoke_orchestration(  # noqa: PLR0913
    *,
    orchestration: OrchestrationManager,
    callback: EvalCaptureCallback,
    thread_id: str,
    prompt: str,
    tags: list[str],
    metadata: dict[str, object],
) -> tuple[str, str, str | None]:
    """Invoke the orchestrator and classify the assistant response.

    Args:
        orchestration: The ready orchestrator.
        callback: The eval capture callback to attach for this invocation.
        thread_id: The thread identifier for the user.
        prompt: The user prompt.
        tags: LangSmith tags to attach to the trace.
        metadata: LangSmith metadata for the trace.

    Returns:
        A tuple of ``(assistant_text, outcome, error_text)``.

    """
    try:
        result = await orchestration.ainvoke(
            thread_id=thread_id,
            user_text=prompt,
            callbacks=[callback],
            tags=tags,
            metadata=metadata,
        )
        assistant_text = await _extract_last_assistant_text(result)
        outcome = _classify_outcome(assistant_text or "")
    except Exception as exc:  # noqa: BLE001
        err_text = f"{type(exc).__name__}: {exc}"
        return err_text, "error", err_text
    return assistant_text, outcome, None


def _update_booking_state(
    state: UserState,
    build: OperationBuildResult,
    *,
    outcome: str,
) -> None:
    """Update booking state based on the operation outcome.

    ``_build_operation_request`` speculatively updates ``state`` with the new
    room and dates before the agent responds.  This function either confirms
    that speculative update (on success) or reverts it (on conflict or error).

    Args:
        state: The mutable per-user state to update.
        build: The operation build result carrying ``op_type`` and the
            pre-operation room/date values used for reverting on failure.
        outcome: The classified outcome label.

    """
    if build.op_type == "CANCEL":
        if outcome == "success":
            state.has_booking = False
            state.last_room_number = None
            state.last_check_in = None
            state.last_check_out = None
        # conflict/error: booking still exists; leave state unchanged
    elif outcome == "success":
        # BOOK/MODIFY succeeded; speculative room/dates in state are correct
        state.has_booking = True
    else:
        # BOOK/MODIFY conflict or error: revert speculative state update
        state.last_room_number = build.old_room
        state.last_check_in = build.old_check_in
        state.last_check_out = build.old_check_out


def _build_op_entry(  # noqa: PLR0913
    *,
    op_type: str,
    user_idx: int,
    op_idx: int,
    thread_id: str,
    prompt: str,
    latency_ms: float,
    outcome: str,
    assistant_text: str,
    err_text: str | None,
    target_used: dict[str, object] | None,
    old_room: int | None,
    old_check_in: str | None,
    old_check_out: str | None,
    sql_calls: list[dict[str, object]],
) -> dict[str, object]:
    """Create a normalized operation log entry.

    Args:
        op_type: The operation type label.
        user_idx: The user index.
        op_idx: The operation index for the user.
        thread_id: The thread identifier.
        prompt: The user prompt string.
        latency_ms: The end-to-end latency in milliseconds.
        outcome: The classified outcome label.
        assistant_text: The assistant response text.
        err_text: Any error text encountered during invocation.
        target_used: The target dict used for the operation, if any.  For
            MODIFY ops this holds the *attempted* new target, which is the
            correct source for ``new_room_number`` / ``new_check_in`` /
            ``new_check_out`` regardless of whether the op succeeded or failed.
        old_room: The prior room number before the operation.
        old_check_in: The prior check-in date before the operation.
        old_check_out: The prior check-out date before the operation.
        sql_calls: Compact summaries of run_sql tool calls from the callback.

    Returns:
        The operation log dictionary.

    """
    op_entry: dict[str, object] = {
        "ts": datetime.now(UTC).isoformat(),
        "user_idx": user_idx,
        "op_idx": op_idx,
        "thread_id": thread_id,
        "op_type": op_type,
        "prompt": prompt,
        "latency_ms": round(latency_ms, 2),
        "outcome": outcome,
        "agent_response": assistant_text or "",
        "assistant_text_trunc": _truncate(assistant_text or "", 300),
    }
    if sql_calls:
        op_entry["sql_calls"] = sql_calls

    if op_type == "MODIFY":
        op_entry.update(
            {
                "old_room_number": old_room,
                "old_check_in": old_check_in,
                "old_check_out": old_check_out,
                "new_room_number": (
                    target_used.get("room_number") if target_used else None
                ),
                "new_check_in": (
                    target_used.get("check_in") if target_used else None
                ),
                "new_check_out": (
                    target_used.get("check_out") if target_used else None
                ),
            },
        )
    elif target_used:
        op_entry.update(
            {
                "room_number": target_used.get("room_number"),
                "check_in": target_used.get("check_in"),
                "check_out": target_used.get("check_out"),
            },
        )

    if err_text:
        op_entry["error"] = err_text

    return op_entry


def _summarize_user_ops(
    *,
    user_idx: int,
    thread_id: str,
    state: UserState,
    local_ops: list[dict[str, object]],
) -> dict[str, object]:
    """Summarize per-user operation statistics and last state.

    Args:
        user_idx: The user index.
        thread_id: The thread identifier for the user.
        state: The final user state.
        local_ops: The list of operation logs for the user.

    Returns:
        A per-user summary dictionary.

    """
    latencies = [
        float(cast("float", op["latency_ms"])) for op in local_ops if "latency_ms" in op
    ]
    successes = sum(1 for op in local_ops if op.get("outcome") == "success")
    conflicts = sum(1 for op in local_ops if op.get("outcome") == "conflict")
    errors = sum(1 for op in local_ops if op.get("outcome") == "error")

    return {
        "user_idx": user_idx,
        "thread_id": thread_id,
        "ops": len(local_ops),
        "successes": successes,
        "conflicts": conflicts,
        "errors": errors,
        "latency_ms_mean": round(statistics.fmean(latencies), 2) if latencies else 0.0,
        "has_booking": state.has_booking,
        "last_room_number": state.last_room_number,
        "last_check_in": state.last_check_in,
        "last_check_out": state.last_check_out,
    }


async def _run_workload(
    orchestration: OrchestrationManager,
    cfg: StressRunConfig,
    targets: list[dict[str, object]],
    hot_targets: list[dict[str, object]],
    *,
    run_id: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Execute concurrent user workloads and collect operation logs.

    Args:
        orchestration: The ready orchestrator.
        cfg: The stress run configuration.
        targets: The full target pool.
        hot_targets: The hot contention target subset.
        run_id: The unique identifier for this stress run, threaded into every
            LangSmith trace tag so traces can be filtered by run in LangSmith.

    Returns:
        A tuple of ``(op_logs, user_logs)``.

    """
    op_logs: list[dict[str, object]] = []
    user_logs: list[dict[str, object]] = []
    semaphore = asyncio.Semaphore(cfg.max_concurrency)

    async def user_task(user_idx: int) -> dict[str, object]:
        """Run a single simulated user's workload under a semaphore.

        Args:
            user_idx: The integer user index for tagging and metadata.

        Returns:
            A per-user summary dictionary with counts and latency stats.

        """
        thread_id = uuid4().hex
        state = UserState()
        local_ops: list[dict[str, object]] = []

        async with semaphore:
            for op_idx in range(cfg.ops_per_user):
                op_type = _pick_op_type(cfg)
                build = _build_operation_request(
                    op_type,
                    state,
                    targets=targets,
                    hot_targets=hot_targets,
                    hot_target_probability=cfg.hot_target_probability,
                )
                tags, metadata = _build_trace_context(
                    run_id=run_id,
                    user_idx=user_idx,
                    op_idx=op_idx,
                    op_type=build.op_type,
                )
                t0 = time.perf_counter()
                callback = EvalCaptureCallback()
                assistant_text, outcome, err_text = await _invoke_orchestration(
                    orchestration=orchestration,
                    callback=callback,
                    thread_id=thread_id,
                    prompt=build.prompt,
                    tags=tags,
                    metadata=metadata,
                )

                latency_ms = (time.perf_counter() - t0) * 1000.0

                sql_calls = [
                    e for e in callback.tool_summary if e.get("tool") == "run_sql"
                ]
                _update_booking_state(state, build, outcome=outcome)
                op_entry = _build_op_entry(
                    op_type=build.op_type,
                    user_idx=user_idx,
                    op_idx=op_idx,
                    thread_id=thread_id,
                    prompt=build.prompt,
                    latency_ms=latency_ms,
                    outcome=outcome,
                    assistant_text=assistant_text,
                    err_text=err_text,
                    target_used=build.target_used,
                    old_room=build.old_room,
                    old_check_in=build.old_check_in,
                    old_check_out=build.old_check_out,
                    sql_calls=sql_calls,
                )

                local_ops.append(op_entry)
                op_logs.append(op_entry)

        return _summarize_user_ops(
            user_idx=user_idx,
            thread_id=thread_id,
            state=state,
            local_ops=local_ops,
        )

    tasks = [user_task(i) for i in range(cfg.users)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for i, res in enumerate(results):
        if isinstance(res, BaseException):
            user_logs.append(
                {
                    "user_idx": i,
                    "error": f"{type(res).__name__}: {res}",
                    "ops": 0,
                    "successes": 0,
                    "conflicts": 0,
                    "errors": 1,
                },
            )
        else:
            user_logs.append(cast("dict[str, object]", res))

    return op_logs, user_logs


async def _check_invariants(
    pool: AsyncConnectionPool,
    cfg: StressRunConfig,
) -> dict[str, object]:
    """Check deterministic database invariants after the stress run.

    Args:
        pool: The open database connection pool pointing at the Neon branch.
        cfg: The stress run configuration supplying retry settings.

    Returns:
        A dictionary describing invariant counts and pass/fail status.

    """
    double_booked_rows: list[Any] = []
    null_status_count = 0
    for attempt in range(cfg.db_retry_attempts):
        try:
            async with pool.connection() as conn:
                await _set_search_path(conn, "public")
                async with conn.cursor() as cur:
                    await cur.execute(
                        """
                        SELECT room_number, date, COUNT(*) AS c
                        FROM room_availability
                        WHERE status = 'Booked'
                        GROUP BY room_number, date
                        HAVING COUNT(*) > 1
                        """,
                    )
                    double_booked_rows = await cur.fetchall()

                    await cur.execute(
                        "SELECT COUNT(*) AS c"
                        " FROM room_availability WHERE status IS NULL",
                    )
                    row = await cur.fetchone()
                    if row is None:
                        msg = "Expected a count row for null status invariant query"
                        raise RuntimeError(msg)
                    null_status_count = cast("int", row[0])
            break
        except (
            psycopg.OperationalError,
            psycopg.InterfaceError,
            PoolTimeout,
            TimeoutError,
        ) as exc:
            if attempt < cfg.db_retry_attempts - 1 and _is_transient_db_error(exc):
                logger.warning(
                    "Transient DB error checking invariants (attempt %d/%d); retrying.",
                    attempt + 1,
                    cfg.db_retry_attempts,
                    exc_info=True,
                )
                await asyncio.sleep(cfg.db_retry_delay_s * (2**attempt))
            else:
                raise

    return {
        "double_booking_violations": len(double_booked_rows),
        "null_status_count": int(null_status_count),
        "passed": len(double_booked_rows) == 0 and int(null_status_count) == 0,
    }


def _generate_date_range(check_in: str, check_out: str) -> list[date]:
    """Generate every date in the half-open interval ``[check_in, check_out)``.

    Args:
        check_in: ISO-8601 check-in date string (inclusive).
        check_out: ISO-8601 check-out date string (exclusive).

    Returns:
        A list of ``date`` objects, one per night of the stay.

    """
    start = date.fromisoformat(check_in)
    end = date.fromisoformat(check_out)
    count = (end - start).days
    return [start + timedelta(days=i) for i in range(max(0, count))]


def _build_thread_final_states(
    op_logs: list[dict[str, object]],
) -> dict[str, dict[str, object] | None]:
    """Walk op_logs in order to compute each thread's expected final booking.

    Successful BOOK and MODIFY ops establish a booking; a successful CANCEL
    clears it.  Conflict and error outcomes leave the state unchanged.

    Args:
        op_logs: The per-operation log entries in chronological order.

    Returns:
        A mapping of thread_id to either a booking dict (keys
        ``room_number``, ``check_in``, ``check_out``) or ``None`` if the
        thread ended with no active booking.

    """
    thread_state: dict[str, dict[str, object] | None] = {}
    for op in op_logs:
        thread_id = str(op.get("thread_id", ""))
        op_type = str(op.get("op_type", ""))
        outcome = str(op.get("outcome", ""))
        if outcome != "success":
            continue
        if op_type == "BOOK":
            thread_state[thread_id] = {
                "room_number": op.get("room_number"),
                "check_in": str(op.get("check_in", "")),
                "check_out": str(op.get("check_out", "")),
            }
        elif op_type == "MODIFY":
            thread_state[thread_id] = {
                "room_number": op.get("new_room_number"),
                "check_in": str(op.get("new_check_in", "")),
                "check_out": str(op.get("new_check_out", "")),
            }
        elif op_type == "CANCEL":
            thread_state[thread_id] = None
    return thread_state


async def _fetch_booked_dates(
    pool: AsyncConnectionPool,
    cfg: StressRunConfig,
) -> frozenset[tuple[int, date]]:
    """Query the database for all (room_number, date) pairs with status='Booked'.

    Args:
        pool: The open database connection pool.
        cfg: The stress run configuration supplying retry settings.

    Returns:
        A frozenset of ``(room_number, date)`` tuples currently marked Booked.

    Raises:
        psycopg.OperationalError: If the query fails after all retry attempts.

    """
    rows: list[Any] = []
    for attempt in range(cfg.db_retry_attempts):
        try:
            async with pool.connection() as conn:
                await _set_search_path(conn, "public")
                async with conn.cursor() as cur:
                    await cur.execute(
                        "SELECT room_number, date"
                        " FROM room_availability WHERE status = 'Booked'",
                    )
                    rows = await cur.fetchall()
            break
        except (
            psycopg.OperationalError,
            psycopg.InterfaceError,
            PoolTimeout,
            TimeoutError,
        ) as exc:
            if attempt < cfg.db_retry_attempts - 1 and _is_transient_db_error(exc):
                logger.warning(
                    "Transient DB error fetching booked dates"
                    " (attempt %d/%d); retrying.",
                    attempt + 1,
                    cfg.db_retry_attempts,
                    exc_info=True,
                )
                await asyncio.sleep(cfg.db_retry_delay_s * (2**attempt))
            else:
                raise
    return frozenset((int(r[0]), r[1]) for r in rows)


def _find_suspicious_conflicts(
    op_logs: list[dict[str, object]],
    booked: frozenset[tuple[int, date]],
) -> list[dict[str, object]]:
    """Find conflict ops whose targeted room/dates have no booked night in the DB.

    A conflict where none of the targeted nights are booked in the final DB
    state may indicate the agent incorrectly rejected an available room.  It
    can also occur legitimately when a booking was made and later cancelled
    during the run, so the result is informational rather than definitive.

    Args:
        op_logs: The per-operation log entries in chronological order.
        booked: The frozenset of ``(room_number, date)`` pairs currently Booked.

    Returns:
        A list of conflict op summaries where no targeted night is in ``booked``.

    """
    suspicious: list[dict[str, object]] = []
    for op in op_logs:
        if op.get("outcome") != "conflict":
            continue
        op_type = str(op.get("op_type", ""))
        if op_type == "BOOK":
            room = op.get("room_number")
            check_in = str(op.get("check_in", ""))
            check_out = str(op.get("check_out", ""))
        elif op_type == "MODIFY":
            room = op.get("new_room_number")
            check_in = str(op.get("new_check_in", ""))
            check_out = str(op.get("new_check_out", ""))
        else:
            continue
        if not room or not check_in or not check_out:
            continue
        room_num = int(cast("int", room))
        dates = _generate_date_range(check_in, check_out)
        if dates and not any((room_num, d) in booked for d in dates):
            suspicious.append({
                "thread_id": op.get("thread_id"),
                "room_number": room,
                "check_in": check_in,
                "check_out": check_out,
                "op_type": op_type,
            })
    return suspicious


async def _reconcile_with_db(
    pool: AsyncConnectionPool,
    op_logs: list[dict[str, object]],
    cfg: StressRunConfig,
) -> dict[str, object]:
    """Reconcile op-log expected bookings against the final database state.

    Two checks are performed:

    1. **Missing bookings**: for each thread whose last successful operation
       was a BOOK or MODIFY, verify that every night in the booked range is
       ``status='Booked'`` in the database.  A missing booking that was not
       subsequently cancelled by the same thread indicates the agent may have
       reported a false success.

    2. **Suspicious conflicts**: for each BOOK or MODIFY op classified as
       ``"conflict"``, check whether the targeted room/dates contain *any*
       booked night in the final database.  If none are booked, the agent may
       have incorrectly rejected an available room — though this can also occur
       legitimately when a booking was made and then cancelled during the run.

    Args:
        pool: The open database connection pool.
        op_logs: The per-operation log entries in chronological order.
        cfg: The stress run configuration supplying retry and limit settings.

    Returns:
        A reconciliation dictionary with counts, detail lists capped at
        ``cfg.reconcile_max_detail`` entries, and a ``"passed"`` flag that is
        ``True`` only when ``missing_from_db == 0``.

    """
    booked = await _fetch_booked_dates(pool, cfg)
    thread_finals = _build_thread_final_states(op_logs)

    expected_count = 0
    confirmed_count = 0
    missing: list[dict[str, object]] = []

    for thread_id, booking in thread_finals.items():
        if booking is None:
            continue  # thread ended with a successful cancel — nothing to assert
        expected_count += 1
        room = booking.get("room_number")
        check_in = str(booking.get("check_in", ""))
        check_out = str(booking.get("check_out", ""))
        if not room or not check_in or not check_out:
            continue
        room_num = int(cast("int", room))
        dates = _generate_date_range(check_in, check_out)
        if dates and all((room_num, d) in booked for d in dates):
            confirmed_count += 1
        else:
            missing.append({
                "thread_id": thread_id,
                "room_number": room,
                "check_in": check_in,
                "check_out": check_out,
            })

    suspicious = _find_suspicious_conflicts(op_logs, booked)

    return {
        "expected_bookings": expected_count,
        "confirmed_in_db": confirmed_count,
        "missing_from_db": len(missing),
        "missing_from_db_detail": missing[:cfg.reconcile_max_detail],
        "suspicious_conflicts": len(suspicious),
        "suspicious_conflicts_detail": suspicious[:cfg.reconcile_max_detail],
        "passed": len(missing) == 0,
    }


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


async def run_stress() -> None:
    """Execute a full stress run against the Neon development branch.

    Resets the Neon branch to its parent baseline, initializes the orchestrator
    and database pool, discovers bookable contention targets, runs concurrent
    user workloads, asserts database invariants, and writes JSON artifacts.

    Raises:
        TimeoutError: If the orchestrator does not become ready within the
            configured ``orchestration.ready_timeout_s``.
        RuntimeError: If no bookable targets can be found after the branch reset.

    """
    cfg = _load_config()
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    _configure_logging(f"stress_{ts}", cfg.log_dir)
    neon_cfg = load_eval_config().neon
    pool: AsyncConnectionPool | None = None
    targets: list[dict[str, object]] = []
    hot_targets: list[dict[str, object]] = []
    op_logs: list[dict[str, object]] = []
    user_logs: list[dict[str, object]] = []
    invariants: dict[str, object] = {}
    reconciliation: dict[str, object] = {}

    start_time = time.perf_counter()
    try:
        orchestration = await _start_orchestration()
        pool = await open_schema_pool(cfg.db_url, max_size=cfg.pool_max)
        targets, hot_targets = await _init_branch_and_targets(
            pool, cfg, neon_cfg,
        )
        op_logs, user_logs = await _run_workload(
            orchestration,
            cfg,
            targets,
            hot_targets,
            run_id=f"stress_{ts}",
        )
        invariants = await _check_invariants(pool, cfg)
        reconciliation = await _reconcile_with_db(pool, op_logs, cfg)
    finally:
        if pool is not None:
            with suppress(Exception):
                await pool.close()

    elapsed_s = max(0.0, time.perf_counter() - start_time)
    summary = _build_summary(
        cfg,
        op_logs=op_logs,
        invariants=invariants,
        reconciliation=reconciliation,
        targets=targets,
        hot_targets=hot_targets,
        elapsed_s=elapsed_s,
    )
    _write_artifacts(
        cfg,
        run_id=f"stress_{ts}",
        op_logs=op_logs,
        user_logs=user_logs,
        summary=summary,
    )


if __name__ == "__main__":
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_stress())
