"""Post-run reconciliation logic for the stress test harness.

Compares the op-log expected bookings against the final database state to
detect missing bookings and suspicious conflict outcomes.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import TYPE_CHECKING, cast

from eval.stress.db import _fetch_booked_dates

if TYPE_CHECKING:
    from psycopg_pool import AsyncConnectionPool

    from eval.stress.models import StressRunConfig


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
