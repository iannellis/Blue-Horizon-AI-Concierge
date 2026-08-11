"""Dataclasses for stress-test configuration and per-user/operation state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import date
    from pathlib import Path


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
        db_url: The read-write Postgres connection URL (`bh_agent_rw`), used
            for the booking agent's write pool and the reconciliation/
            invariant-check pool.
        ro_db_url: The read-only Postgres connection URL (`bh_agent_ro`),
            used exclusively by the booking agent's `run_sql` tool.

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
    ro_db_url: str


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
    has_booking: bool = field(default=False)


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
