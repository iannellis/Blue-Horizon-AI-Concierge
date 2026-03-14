"""Database utilities for the stress test harness.

Provides helpers for branch reset, target discovery, invariant checking,
and DB retry logic.
"""

from __future__ import annotations

import logging
import random
from datetime import date, timedelta
from typing import TYPE_CHECKING, Any, cast

from psycopg_pool import PoolTimeout
from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from eval.rooms_db_manager import reset_neon_branch

if TYPE_CHECKING:
    from psycopg_pool import AsyncConnectionPool

    from eval.config import NeonConfig
    from eval.stress.models import StressRunConfig

logger = logging.getLogger(__name__)


async def _init_branch_and_targets(
    pool: AsyncConnectionPool,
    cfg: StressRunConfig,
    neon_cfg: NeonConfig,
    *,
    api_key: str | None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Reset the Neon branch and discover bookable contention targets.

    Resets the configured Neon branch to its parent baseline, then queries
    the database to find bookable availability spans for the workload.

    Args:
        pool: The open database connection pool pointing at the Neon branch.
        cfg: The stress run configuration.
        neon_cfg: Neon project and branch settings from the stress config.
        api_key: Neon management API key sourced from the stress config.

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
    await reset_neon_branch(neon_cfg, api_key=api_key)

    targets: list[dict[str, object]] = []
    async for attempt in AsyncRetrying(
        retry=retry_if_exception(_is_transient_db_error),
        stop=stop_after_attempt(cfg.db_retry_attempts),
        wait=wait_exponential(multiplier=cfg.db_retry_delay_s, exp_base=2),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    ):
        with attempt:
            async with pool.connection() as conn:
                await _set_search_path(conn, "public")
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
    logger.info(
        "Branch reset complete: %d targets available (%d hot).",
        len(targets),
        len(hot_targets),
    )
    return targets, hot_targets


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
    async for attempt in AsyncRetrying(
        retry=retry_if_exception(_is_transient_db_error),
        stop=stop_after_attempt(cfg.db_retry_attempts),
        wait=wait_exponential(multiplier=cfg.db_retry_delay_s, exp_base=2),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    ):
        with attempt:
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

    return {
        "double_booking_violations": len(double_booked_rows),
        "null_status_count": int(null_status_count),
        "passed": len(double_booked_rows) == 0 and int(null_status_count) == 0,
    }


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
    async for attempt in AsyncRetrying(
        retry=retry_if_exception(_is_transient_db_error),
        stop=stop_after_attempt(cfg.db_retry_attempts),
        wait=wait_exponential(multiplier=cfg.db_retry_delay_s, exp_base=2),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    ):
        with attempt:
            async with pool.connection() as conn:
                await _set_search_path(conn, "public")
                async with conn.cursor() as cur:
                    await cur.execute(
                        "SELECT room_number, date"
                        " FROM room_availability WHERE status = 'Booked'",
                    )
                    rows = await cur.fetchall()
    return frozenset((int(r[0]), r[1]) for r in rows)


async def _set_search_path(conn: object, schema: str) -> None:
    """Set the connection's search path to the target schema safely.

    Args:
        conn: An async psycopg connection-like object.
        schema: The schema name to place at the front of ``search_path``.

    """
    conn_obj = cast("object", conn)
    await cast("object", conn_obj).execute(  # type: ignore[attr-defined]
        "SELECT set_config('search_path', %s, false)",
        (schema,),
    )


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
