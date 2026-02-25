"""Workload generation and execution for the stress test harness.

Provides operation-building helpers, orchestration invocation, booking-state
tracking, and the top-level concurrent workload runner.
"""

from __future__ import annotations

import asyncio
import logging
import random
import statistics
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import cast
from uuid import uuid4

from eval._utils import truncate as _truncate
from eval.langsmith_target import EvalCaptureCallback, OrchestrationManager
from eval.stress.models import OperationBuildResult, StressRunConfig, UserState

logger = logging.getLogger(__name__)


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
            f"to {book_check_out}."
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
            f"to {new_check_out}."
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


def _build_trace_context(
    *,
    run_id: str,
    user_idx: int,
    op_idx: int,
    op_type: str,
) -> tuple[list[str], dict[str, object]]:
    """Build LangSmith tags and metadata for a stress operation.

    The stress run identifier is stored as ``stress_run_id`` rather than
    ``run_id`` in the metadata dict.  LangGraph reserves ``run_id`` as an
    internal run-deduplication key: passing the same value across multiple
    ``ainvoke`` calls on the same thread causes subsequent calls to return the
    cached checkpoint without re-executing the graph.

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
        "stress_run_id": run_id,
        "user_idx": user_idx,
        "op_idx": op_idx,
        "op_type": op_type,
    }
    return tags, metadata
