"""Stress and race testing suite for the LangGraph hotel agent.

This module creates a single shared Postgres schema for a stress run, loads
baseline data, discovers bookable contention targets, and then drives
concurrent end-to-end BOOK / MODIFY / CANCEL flows through the orchestrator.
It records per-operation and per-user JSONL artifacts, computes summary
statistics and database invariants, and emits LangSmith traces via tags and
metadata passed to the orchestrator.

"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import random
import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, cast
from uuid import uuid4

from dotenv import load_dotenv

from blue_horizon.agents.rooms import set_eval_schema
from blue_horizon.config import load_app_config
from eval._utils import truncate as _truncate
from eval.config import load_eval_config
from eval.langsmith_target import OrchestrationManager
from eval.rooms_schema_manager import (
    create_case_schema,
    generate_schema_name,
    open_schema_pool,
    teardown_schema,
)

if TYPE_CHECKING:
    from contextvars import Token
    from pathlib import Path

    from psycopg_pool import AsyncConnectionPool

load_dotenv()


@dataclass(frozen=True)
class StressRunConfig:
    """Configuration for a single stress run.

    Attributes:
        schema: The shared schema name for the run.
        users: The number of concurrent simulated users.
        ops_per_user: The number of operations per user.
        max_concurrency: The maximum concurrent users allowed at once.
        baseline_data_path: The baseline data directory passed to schema reset.
        output_dir: The base output directory for artifacts.
        stay_nights: The length of stay in nights for generated targets.
        num_targets: The number of available targets to precompute.
        hot_target_count: The size of the hot contention target subset.
        hot_target_probability: Probability of selecting a hot target vs any target.
        start_date: The search start date for available targets.
        horizon_days: The search horizon, in days, for available targets.
        pool_max: The maximum connection pool size.
        db_url: The Postgres connection URL.

    """

    schema: str
    users: int
    ops_per_user: int
    max_concurrency: int
    baseline_data_path: str
    output_dir: Path
    stay_nights: int
    num_targets: int
    hot_target_count: int
    hot_target_probability: float
    start_date: date
    horizon_days: int
    pool_max: int
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


async def run_stress() -> None:
    """Execute a full stress run against a shared evaluation schema.

    This routine initializes the orchestrator and database pool once, creates
    and loads a shared stress schema, discovers bookable contention targets,
    runs concurrent user workloads, asserts database invariants, writes JSON
    artifacts, and performs best-effort cleanup.

    Raises:
        TimeoutError: If the orchestrator does not become ready within 30s.
        RuntimeError: If no bookable targets can be found.

    """
    cfg = _load_config()
    pool: AsyncConnectionPool | None = None
    token: Token[str | None] | None = None
    targets: list[dict[str, object]] = []
    hot_targets: list[dict[str, object]] = []
    op_logs: list[dict[str, object]] = []
    user_logs: list[dict[str, object]] = []
    invariants: dict[str, object] = {}
    schema_drop_error: str | None = None

    start_time = time.perf_counter()
    try:
        orchestration = await _start_orchestration()
        pool = await open_schema_pool(cfg.db_url, max_size=cfg.pool_max)
        token, targets, hot_targets = await _init_schema_and_targets(pool, cfg)
        op_logs, user_logs = await _run_workload(
            orchestration,
            cfg,
            targets,
            hot_targets,
        )
        invariants = await _check_invariants(pool, cfg.schema)
    finally:
        schema_drop_error = await teardown_schema(token, pool, cfg.schema)

    elapsed_s = max(0.0, time.perf_counter() - start_time)
    summary = _build_summary(
        cfg,
        op_logs=op_logs,
        invariants=invariants,
        targets=targets,
        hot_targets=hot_targets,
        elapsed_s=elapsed_s,
        schema_drop_error=schema_drop_error,
    )
    _write_artifacts(
        cfg,
        op_logs=op_logs,
        user_logs=user_logs,
        summary=summary,
    )


def _load_config() -> StressRunConfig:
    """Load stress configuration from eval_config.toml.

    Returns:
        A fully populated ``StressRunConfig`` instance.

    Raises:
        RuntimeError: If a database URL cannot be determined.

    """
    cfg = load_eval_config().stress
    schema = cfg.db_schema or generate_schema_name("stress")
    users = cfg.users
    ops_per_user = cfg.ops_per_user
    max_concurrency = cfg.max_concurrency
    baseline_data_path = str(cfg.baseline_data_path)
    output_dir = cfg.output_dir
    stay_nights = cfg.stay_nights
    num_targets = cfg.num_targets
    hot_target_count = cfg.hot_target_count
    hot_target_probability = cfg.hot_target_probability
    start_date = cfg.start_date
    horizon_days = cfg.horizon_days
    pool_max = cfg.pool_max

    db_url = os.getenv("EVAL_DB_URL") or load_app_config().pgsql_db_url
    if not db_url:
        msg = "Database URL is required but was not found"
        raise RuntimeError(msg)

    return StressRunConfig(
        schema=schema,
        users=users,
        ops_per_user=ops_per_user,
        max_concurrency=max_concurrency,
        baseline_data_path=baseline_data_path,
        output_dir=output_dir,
        stay_nights=stay_nights,
        num_targets=num_targets,
        hot_target_count=hot_target_count,
        hot_target_probability=hot_target_probability,
        start_date=start_date,
        horizon_days=horizon_days,
        pool_max=pool_max,
        db_url=db_url,
    )


async def _start_orchestration() -> OrchestrationManager:
    """Start the orchestrator and wait for readiness with a timeout.

    Returns:
        A ready ``OrchestrationManager`` instance.

    Raises:
        TimeoutError: If readiness is not reached within 30 seconds.

    """
    orchestration = OrchestrationManager(prune_tool_messages=False)
    await orchestration.start()

    ready_deadline = time.perf_counter() + 30.0
    while not orchestration.is_ready:
        if time.perf_counter() > ready_deadline:
            msg = "OrchestrationManager did not become ready within 30s"
            raise TimeoutError(msg)
        await asyncio.sleep(0.1)

    return orchestration


async def _init_schema_and_targets(
    pool: AsyncConnectionPool,
    cfg: StressRunConfig,
) -> tuple[Token[str | None], list[dict[str, object]], list[dict[str, object]]]:
    """Create the stress schema, set search path, and find targets.

    Args:
        pool: The open database connection pool.
        cfg: The stress run configuration.

    Returns:
        A tuple of ``(token, targets, hot_targets)``.

    Raises:
        RuntimeError: If no bookable targets can be found.

    """
    await create_case_schema(
        pool=pool,
        schema=cfg.schema,
        data_path=cfg.baseline_data_path,
    )
    token = set_eval_schema(cfg.schema)

    async with pool.connection() as conn:
        await _set_search_path(conn, cfg.schema)
        targets = await find_available_targets(
            conn,
            num_targets=cfg.num_targets,
            stay_nights=cfg.stay_nights,
            start_date=cfg.start_date,
            horizon_days=cfg.horizon_days,
        )

    if not targets:
        msg = "No bookable targets found"
        raise RuntimeError(msg)

    hot_targets = targets[: max(0, min(cfg.hot_target_count, len(targets)))]
    return token, targets, hot_targets


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


async def _run_workload(
    orchestration: OrchestrationManager,
    cfg: StressRunConfig,
    targets: list[dict[str, object]],
    hot_targets: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Execute concurrent user workloads and collect operation logs.

    Args:
        orchestration: The ready orchestrator.
        cfg: The stress run configuration.
        targets: The full target pool.
        hot_targets: The hot contention target subset.

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
                op_type = _pick_op_type()
                build = _build_operation_request(
                    op_type,
                    state,
                    targets=targets,
                    hot_targets=hot_targets,
                    hot_target_probability=cfg.hot_target_probability,
                )
                tags, metadata = _build_trace_context(
                    cfg=cfg,
                    user_idx=user_idx,
                    op_idx=op_idx,
                    op_type=build.op_type,
                )
                t0 = time.perf_counter()
                assistant_text, outcome, err_text = await _invoke_orchestration(
                    orchestration=orchestration,
                    thread_id=thread_id,
                    prompt=build.prompt,
                    tags=tags,
                    metadata=metadata,
                )

                latency_ms = (time.perf_counter() - t0) * 1000.0

                _update_booking_state(state, op_type=build.op_type, outcome=outcome)
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
                    state=state,
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


def _pick_op_type() -> str:
    """Sample an operation type using the configured weights.

    Returns:
        The chosen operation type string.

    """
    return random.choices(  # noqa: S311
        population=["BOOK", "MODIFY", "CANCEL"],
        weights=[0.5, 0.25, 0.25],
        k=1,
    )[0]


def _build_operation_request(
    op_type: str,
    state: UserState,
    *,
    targets: list[dict[str, object]],
    hot_targets: list[dict[str, object]],
    hot_target_probability: float,
) -> OperationBuildResult:
    """Build a prompt for the given operation and update user state.

    Args:
        op_type: The requested operation type label.
        state: The mutable per-user state to update.
        targets: The full target pool.
        hot_targets: The hot contention target subset.
        hot_target_probability: Probability of selecting hot vs any target.

    Returns:
        A structured operation build result.

    """
    old_room = state.last_room_number
    old_check_in = state.last_check_in
    old_check_out = state.last_check_out

    if op_type == "BOOK" or not state.last_room_number:
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


def _choose_new_target(
    *,
    targets: list[dict[str, object]],
    hot_targets: list[dict[str, object]],
    old_room: int | None,
    old_check_in: str | None,
    old_check_out: str | None,
) -> dict[str, object]:
    """Choose a new target that differs from the previous booking if possible.

    Args:
        targets: The full candidate target pool.
        hot_targets: The high-contention subset.
        old_room: The prior room number, if known.
        old_check_in: The prior check-in date, if known.
        old_check_out: The prior check-out date, if known.

    Returns:
        A chosen target dictionary, preferring a different triplet.

    """
    candidates = list(hot_targets) if hot_targets else list(targets)
    random.shuffle(candidates)
    for cand in candidates:
        if (
            cand.get("room_number") != old_room
            or cand.get("check_in") != old_check_in
            or cand.get("check_out") != old_check_out
        ):
            return cand
    return random.choice(targets)  # noqa: S311


def _build_trace_context(
    *,
    cfg: StressRunConfig,
    user_idx: int,
    op_idx: int,
    op_type: str,
) -> tuple[list[str], dict[str, object]]:
    """Build LangSmith tags and metadata for a stress operation.

    Args:
        cfg: The stress run configuration.
        user_idx: The user index for the operation.
        op_idx: The operation index for the user.
        op_type: The operation type label.

    Returns:
        A tuple of ``(tags, metadata)``.

    """
    tags = [
        "stress",
        f"user:{user_idx}",
        f"op:{op_idx}",
        f"schema:{cfg.schema}",
        f"op_type:{op_type}",
    ]
    metadata = {
        "user_idx": user_idx,
        "op_idx": op_idx,
        "schema": cfg.schema,
        "op_type": op_type,
    }
    return tags, metadata


async def _invoke_orchestration(
    *,
    orchestration: OrchestrationManager,
    thread_id: str,
    prompt: str,
    tags: list[str],
    metadata: dict[str, object],
) -> tuple[str, str, str | None]:
    """Invoke the orchestrator and classify the assistant response.

    Args:
        orchestration: The ready orchestrator.
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
            tags=tags,
            metadata=metadata,
        )
        assistant_text = await _extract_last_assistant_text(result)
        outcome = _classify_outcome(assistant_text or "")
    except Exception as exc:  # noqa: BLE001
        err_text = f"{type(exc).__name__}: {exc}"
        return err_text, "error", err_text
    return assistant_text, outcome, None


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
    lower = text.lower()
    conflict_markers = [
        "unavailable",
        "no availability",
        "already booked",
        "couldn't",
        "cannot",
        "not available",
    ]
    if any(m in lower for m in conflict_markers):
        return "conflict"
    if any(m in lower for m in ["error", "failed", "exception"]):
        return "error"
    return "success"


def _update_booking_state(state: UserState, *, op_type: str, outcome: str) -> None:
    """Update booking state based on the operation outcome.

    Args:
        state: The mutable per-user state to update.
        op_type: The operation type label.
        outcome: The classified outcome label.

    """
    if op_type == "CANCEL":
        state.has_booking = False
    elif outcome != "error":
        state.has_booking = True


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
    state: UserState,
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
        target_used: The target dict used for the operation, if any.
        old_room: The prior room number before the operation.
        old_check_in: The prior check-in date before the operation.
        old_check_out: The prior check-out date before the operation.
        state: The current user state after the operation.

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
        "assistant_text_trunc": _truncate(assistant_text or "", 300),
    }

    if op_type == "MODIFY":
        op_entry.update(
            {
                "old_room_number": old_room,
                "old_check_in": old_check_in,
                "old_check_out": old_check_out,
                "new_room_number": state.last_room_number,
                "new_check_in": state.last_check_in,
                "new_check_out": state.last_check_out,
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


async def _check_invariants(
    pool: AsyncConnectionPool,
    schema: str,
) -> dict[str, object]:
    """Check deterministic database invariants in the stress schema.

    Args:
        pool: The open database connection pool.
        schema: The schema name to set as the search path.

    Returns:
        A dictionary describing invariant counts and pass/fail status.

    """
    async with pool.connection() as conn:
        await _set_search_path(conn, schema)
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
                "SELECT COUNT(*) AS c FROM room_availability WHERE status IS NULL",
            )
            row = await cur.fetchone()
            if row is None:
                msg = "Expected a count row for null status invariant query"
                raise RuntimeError(msg)
            null_status_count = cast("int", row[0])

    return {
        "double_booking_violations": len(double_booked_rows),
        "null_status_count": int(null_status_count),
        "passed": len(double_booked_rows) == 0 and int(null_status_count) == 0,
    }


def _build_summary(  # noqa: PLR0913
    cfg: StressRunConfig,
    *,
    op_logs: list[dict[str, object]],
    invariants: dict[str, object],
    targets: list[dict[str, object]],
    hot_targets: list[dict[str, object]],
    elapsed_s: float,
    schema_drop_error: str | None,
) -> dict[str, object]:
    """Compute summary statistics and status for the stress run.

    Args:
        cfg: The stress run configuration.
        op_logs: The per-operation logs.
        invariants: The invariant check results.
        targets: The full target pool.
        hot_targets: The hot contention subset.
        elapsed_s: The total elapsed time in seconds.
        schema_drop_error: Any schema drop error message.

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
        "schema": cfg.schema,
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
        "targets": {
            "total": len(targets),
            "hot_count": len(hot_targets),
            "hot_targets": hot_targets,
        },
        "cleanup": {
            "schema_drop_error": schema_drop_error,
        },
    }

    passed = bool(invariants.get("passed")) and schema_drop_error is None
    summary["status"] = "PASS" if passed else "FAIL"
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
    op_logs: list[dict[str, object]],
    user_logs: list[dict[str, object]],
    summary: dict[str, object],
) -> None:
    """Write JSON artifacts for operations, users, and the summary.

    Args:
        cfg: The stress run configuration.
        op_logs: The per-operation logs.
        user_logs: The per-user logs.
        summary: The computed summary dictionary.

    """
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = cfg.output_dir / f"stress_{ts}"
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


if __name__ == "__main__":
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_stress())
