"""Entry point for the stress test harness.

Loads configuration, initializes the orchestrator and DB pool, runs the
workload, and writes artifacts.  Can be executed directly as a script.
"""

from __future__ import annotations

import asyncio
import platform
import time
from contextlib import suppress
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from dotenv import load_dotenv

from blue_horizon.config import load_app_config
from eval.config import load_stress_config
from eval.langsmith_target import OrchestrationManager
from eval.rooms_db_manager import open_schema_pool
from eval.stress.artifacts import _build_summary, _configure_logging, _write_artifacts
from eval.stress.db import _check_invariants, _init_branch_and_targets
from eval.stress.models import StressRunConfig
from eval.stress.reconciliation import _reconcile_with_db
from eval.stress.workload import _run_workload

if TYPE_CHECKING:
    from psycopg_pool import AsyncConnectionPool

load_dotenv()


def _load_config() -> StressRunConfig:
    """Load stress configuration from stress_config.toml.

    Returns:
        A fully populated ``StressRunConfig`` instance.

    Raises:
        RuntimeError: If a database URL cannot be determined.

    """
    cfg = load_stress_config().stress

    db_url = load_stress_config().pgsql_eval_db_url or load_app_config().pgsql_db_url
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


async def _start_orchestration(*, db_url: str) -> OrchestrationManager:
    """Start the orchestrator and wait for readiness with a configured timeout.

    The timeout is read from ``stress_config.toml`` via
    ``orchestration.ready_timeout_s``.

    The ``db_url`` is forwarded to :class:`OrchestrationManager` so that the
    rooms SQL agent writes to the same database that the reconciliation pool
    reads from.  When ``PGSQL_EVAL_DB_URL`` is set, ``db_url`` is that eval
    URL; without it, ``db_url`` is ``PGSQL_DB_URL``.  Either way, agent
    writes and reconciliation queries share one database.

    Args:
        db_url: Postgres connection URL used by the rooms SQL agent.

    Returns:
        A ready ``OrchestrationManager`` instance.

    Raises:
        TimeoutError: If readiness is not reached within the configured timeout.

    """
    ready_timeout_s = load_stress_config().orchestration.ready_timeout_s
    orchestration = OrchestrationManager(pgsql_db_url=db_url)
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
    stress_cfg = load_stress_config()
    neon_cfg = stress_cfg.neon
    neon_api_key = stress_cfg.neon_api_key
    pool: AsyncConnectionPool | None = None
    targets: list[dict[str, object]] = []
    hot_targets: list[dict[str, object]] = []
    op_logs: list[dict[str, object]] = []
    user_logs: list[dict[str, object]] = []
    invariants: dict[str, object] = {}
    reconciliation: dict[str, object] = {}

    start_time = time.perf_counter()
    try:
        orchestration = await _start_orchestration(db_url=cfg.db_url)
        pool = await open_schema_pool(cfg.db_url, max_size=cfg.pool_max)
        targets, hot_targets = await _init_branch_and_targets(
            pool, cfg, neon_cfg, api_key=neon_api_key,
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
