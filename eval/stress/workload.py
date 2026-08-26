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
from datetime import UTC, datetime
from typing import cast
from uuid import uuid4

from blue_horizon.config import load_app_config
from eval._utils import truncate as _truncate
from eval.langsmith_target import EvalCaptureCallback, OrchestrationManager
from eval.langsmith_target._confirm import auto_confirm_pending_proposal
from eval.langsmith_target._text_utils import _extract_assistant_text_from_result
from eval.stress.models import OperationBuildResult, StressRunConfig, UserState

logger = logging.getLogger(__name__)

# Tool names that create a proposal -- mirrors eval.evaluators._booking and
# proposals.ProposalAction, though the stress harness only cares whether one
# was made, not which of BOOK/CANCEL/MODIFY it was.
_PROPOSE_TOOL_NAMES = frozenset(
    {"propose_booking", "propose_cancellation", "propose_modification"},
)


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
    # Simulated users beyond this count wrap around and share a guest
    # identity with an earlier user -- acceptable here, since contention
    # across users is the point of a stress run and the UI's own automated
    # assignment offers the same seeded guests.
    seeded_customer_count = (
        load_app_config().load_data.booking_pgsql.seeded_customer_count
    )

    async def user_task(user_idx: int) -> dict[str, object]:
        """Run a single simulated user's workload under a semaphore.

        Args:
            user_idx: The integer user index for tagging and metadata.

        Returns:
            A per-user summary dictionary with counts and latency stats.

        """
        thread_id = uuid4().hex
        customer_id = (user_idx % seeded_customer_count) + 1
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
                assistant_text, err_text = await _invoke_orchestration(
                    orchestration=orchestration,
                    callback=callback,
                    thread_id=thread_id,
                    customer_id=customer_id,
                    prompt=build.prompt,
                    tags=tags,
                    metadata=metadata,
                )

                latency_ms = (time.perf_counter() - t0) * 1000.0

                sql_calls = _collect_run_sql_calls(callback.tool_summary)
                outcome = _classify_outcome(
                    op_type=build.op_type,
                    assistant_text=assistant_text,
                    err_text=err_text,
                    tool_summary=callback.tool_summary,
                )
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


def _select_pool(
    targets: list[dict[str, object]],
    hot_targets: list[dict[str, object]],
    hot_target_probability: float,
) -> list[dict[str, object]]:
    """Select the hot or cold pool using the configured probability split.

    Args:
        targets: The full candidate target pool.
        hot_targets: The high-contention subset.
        hot_target_probability: Probability of selecting the hot pool.

    Returns:
        Either ``hot_targets`` or ``targets``.

    """
    use_hot = bool(hot_targets) and random.random() < hot_target_probability  # noqa: S311
    return hot_targets if use_hot else targets


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
    return random.choice(_select_pool(targets, hot_targets, hot_target_probability))  # noqa: S311


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
    candidates = list(_select_pool(targets, hot_targets, hot_target_probability))
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
    customer_id: int,
    prompt: str,
    tags: list[str],
    metadata: dict[str, object],
) -> tuple[str, str | None]:
    """Invoke the orchestrator, auto-confirm any proposal, and collect the response.

    The auto-confirm step -- the harness's stand-in for a human clicking
    Confirm, since a stress run drives no browser -- runs inside the same
    try/except as the turn itself, so an unexpected failure there (as
    opposed to the ordinary `BookingWriteError` refusals
    `auto_confirm_pending_proposal` already turns into a captured
    `confirm_booking` entry) still surfaces as an ``"error"`` outcome rather
    than escaping to the caller.

    Args:
        orchestration: The ready orchestrator.
        callback: The eval capture callback to attach for this invocation.
        thread_id: The thread identifier for the user.
        customer_id: The seeded guest identity this simulated user is
            impersonating for the whole run.
        prompt: The user prompt.
        tags: LangSmith tags to attach to the trace.
        metadata: LangSmith metadata for the trace.

    Returns:
        A tuple of ``(assistant_text, error_text)``.

    """
    try:
        result = await orchestration.ainvoke(
            thread_id=thread_id,
            user_text=prompt,
            customer_id=customer_id,
            callbacks=[callback],
            tags=tags,
            metadata=metadata,
        )
        assistant_text = _extract_assistant_text_from_result(result)
        await auto_confirm_pending_proposal(
            orchestration,
            thread_id=thread_id,
            customer_id=customer_id,
            callback=callback,
        )
    except Exception as exc:  # noqa: BLE001
        err_text = f"{type(exc).__name__}: {exc}"
        return "", err_text
    return assistant_text, None


def _collect_run_sql_calls(
    tool_summary: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Collect ``run_sql`` entries from a callback tool summary.

    Args:
        tool_summary: Tool summary entries captured during orchestration.

    Returns:
        List of ``run_sql`` tool entries in original order.

    """
    sql_calls: list[dict[str, object]] = []

    for entry in tool_summary:
        if not isinstance(entry, dict):
            continue
        if entry.get("tool") != "run_sql":
            continue
        sql_calls.append(entry)

    return sql_calls


def _classify_outcome(
    *,
    op_type: str,
    assistant_text: str,
    err_text: str | None,
    tool_summary: list[dict[str, object]],
) -> str:
    """Classify a stress-test operation outcome.

    The propose+confirm pair captured for the turn is preferred because it
    is more stable than natural-language response text. Assistant text is
    only used as a fallback when the turn made no ``propose_*`` call at all
    (for example, the agent refused or asked a clarifying question instead).

    Args:
        op_type: Effective operation type (``BOOK``, ``MODIFY``, or ``CANCEL``).
        assistant_text: Assistant response text.
        err_text: Python-level error text from orchestration invocation, if any.
        tool_summary: Captured tool summary entries for the turn, including
            any ``propose_*`` and ``confirm_booking`` entries.

    Returns:
        One of ``"success"``, ``"conflict"``, or ``"error"``.

    """
    _ = op_type
    if err_text:
        return "error"

    propose_confirm_outcome = _classify_propose_confirm_outcome(tool_summary)
    if propose_confirm_outcome is not None:
        return propose_confirm_outcome

    return _classify_text_outcome(assistant_text)


def _classify_propose_confirm_outcome(
    tool_summary: list[dict[str, object]],
) -> str | None:
    """Classify an outcome from the turn's propose_* and confirm_booking entries.

    `commit_booking`/`cancel_booking`/`modify_booking` are each atomic, so
    unlike the old SQL-CTE era there is no partial-success shape left to
    read out of a result row: a proposal that failed to price, or a confirm
    that lost a race to another thread, are both bucketed as ``"conflict"``
    -- the expected outcome under contention -- while a successful confirm
    is ``"success"``.

    Args:
        tool_summary: Captured tool summary entries for the turn.

    Returns:
        Structured outcome label, or ``None`` when the turn made no
        ``propose_*`` call so the caller can fall back to text heuristics.

    """
    propose_entry, confirm_entry = _last_propose_and_confirm(tool_summary)
    if propose_entry is None:
        return None
    if propose_entry.get("status") == "error":
        return "conflict"
    if confirm_entry is None:
        return None
    if confirm_entry.get("status") == "error":
        return "conflict"
    return "success"


def _last_propose_and_confirm(
    tool_summary: list[dict[str, object]],
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    """Find the last propose_* and confirm_booking entries in a tool summary.

    A turn has at most one pending proposal (the store supersedes any prior
    one), so the *last* entry of each kind is the pair that belongs together.

    Args:
        tool_summary: Captured tool summary entries for the turn.

    Returns:
        Tuple of ``(propose_entry, confirm_entry)``, either or both ``None``
        when not present.

    """
    propose_entry: dict[str, object] | None = None
    confirm_entry: dict[str, object] | None = None
    for entry in tool_summary:
        if not isinstance(entry, dict):
            continue
        tool = entry.get("tool")
        if tool in _PROPOSE_TOOL_NAMES:
            propose_entry = entry
        elif tool == "confirm_booking":
            confirm_entry = entry
    return propose_entry, confirm_entry


def _classify_text_outcome(text: str) -> str:
    """Classify an assistant response into a coarse outcome bucket.

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
        "you already have",  # "You already have room N booked for those dates."
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
