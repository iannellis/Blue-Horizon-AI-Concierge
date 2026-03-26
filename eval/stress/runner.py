"""Entry point for the stress test harness.

Loads configuration, initializes the orchestrator and DB pool, runs the
workload, and writes artifacts.  Can be executed directly as a script.
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import time
from contextlib import AsyncExitStack
from datetime import UTC, datetime

from dotenv import load_dotenv

from blue_horizon.config import load_app_config
from eval._utils import configure_logging as _configure_logging
from eval.config import load_stress_config
from eval.langsmith_target import OrchestrationManager
from eval.langsmith_target._orchestration_utils import wait_for_orchestration_ready
from eval.rooms_db_manager import open_schema_pool
from eval.stress.artifacts import _build_summary, _write_artifacts
from eval.stress.db import _check_invariants, _init_branch_and_targets
from eval.stress.models import StressRunConfig
from eval.stress.reconciliation import _reconcile_with_db
from eval.stress.workload import _run_workload

load_dotenv()

logger = logging.getLogger(__name__)


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


def _override_db_url(db_url: str) -> None:
    """Write ``db_url`` to ``PGSQL_DB_URL`` and clear the ``load_app_config`` cache.

    ``OrchestrationResources.__init__`` calls ``load_app_config()`` unconditionally,
    which requires ``PGSQL_DB_URL`` to be present in the environment.  In CI the
    stress test only receives ``PGSQL_EVAL_DB_URL``; this function bridges the gap
    by copying ``cfg.db_url`` (already resolved from ``PGSQL_EVAL_DB_URL``) into
    ``PGSQL_DB_URL`` before the orchestrator is created.

    Args:
        db_url: Resolved database URL to set as ``PGSQL_DB_URL``.

    """
    os.environ["PGSQL_DB_URL"] = db_url
    load_app_config.cache_clear()
    logger.info("PGSQL_EVAL_DB_URL override applied: PGSQL_DB_URL updated.")


async def _start_orchestration(
    *,
    db_url: str,
    ready_timeout_s: float,
) -> OrchestrationManager:
    """Start the orchestrator and wait for readiness with a configured timeout.

    The ``db_url`` is forwarded to :class:`OrchestrationManager` so that the
    rooms SQL agent writes to the same database that the reconciliation pool
    reads from.  When ``PGSQL_EVAL_DB_URL`` is set, ``db_url`` is that eval
    URL; without it, ``db_url`` is ``PGSQL_DB_URL``.  Either way, agent
    writes and reconciliation queries share one database.

    Args:
        db_url: Postgres connection URL used by the rooms SQL agent.
        ready_timeout_s: Maximum number of seconds to wait for readiness.

    Returns:
        A ready ``OrchestrationManager`` instance.

    Raises:
        TimeoutError: If readiness is not reached within the configured timeout.

    """
    orchestration = OrchestrationManager(pgsql_db_url=db_url)
    await orchestration.start()
    try:
        await wait_for_orchestration_ready(
            orchestration,
            timeout_s=ready_timeout_s,
            manager_name="OrchestrationManager",
            poll_interval_s=0.1,
        )
    except Exception:
        await orchestration.stop()
        raise

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
    run_id = f"stress_{ts}"
    _configure_logging(f"stress_{ts}", cfg.log_dir)
    _override_db_url(cfg.db_url)
    stress_cfg = load_stress_config()
    neon_cfg = stress_cfg.neon
    neon_api_key = stress_cfg.neon_api_key
    targets: list[dict[str, object]] = []
    hot_targets: list[dict[str, object]] = []
    op_logs: list[dict[str, object]] = []
    user_logs: list[dict[str, object]] = []
    invariants: dict[str, object] = {}
    reconciliation: dict[str, object] = {}
    caught_exc: Exception | None = None

    start_time = time.perf_counter()
    try:
        async with AsyncExitStack() as stack:
            orchestration = await _start_orchestration(
                db_url=cfg.db_url,
                ready_timeout_s=stress_cfg.orchestration.ready_timeout_s,
            )
            stack.push_async_callback(orchestration.stop)

            pool = await open_schema_pool(cfg.db_url, max_size=cfg.pool_max)
            stack.push_async_callback(pool.close)

            targets, hot_targets = await _init_branch_and_targets(
                pool,
                cfg,
                neon_cfg,
                api_key=neon_api_key,
            )
            op_logs, user_logs = await _run_workload(
                orchestration,
                cfg,
                targets,
                hot_targets,
                run_id=run_id,
            )
            invariants = await _check_invariants(pool, cfg)
            reconciliation = await _reconcile_with_db(pool, op_logs, cfg)
    except Exception as exc:  # noqa: BLE001
        caught_exc = exc

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
    if caught_exc is not None:
        summary["error"] = f"{type(caught_exc).__name__}: {caught_exc}"
        summary["status"] = "FAIL"
    _write_artifacts(
        cfg,
        run_id=run_id,
        op_logs=op_logs,
        user_logs=user_logs,
        summary=summary,
    )
    if caught_exc is not None:
        raise caught_exc


if __name__ == "__main__":
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_stress())
