"""Distributed schema slot management for evaluation runs.

This module provides a Postgres-backed coordination mechanism to cap the number
of concurrently active evaluation schemas across processes or hosts.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from psycopg import sql

if TYPE_CHECKING:
    from psycopg_pool import AsyncConnectionPool

_TABLE_NAME = "eval_schema_slots"


async def ensure_schema_slot_table(
    *,
    pool: AsyncConnectionPool[Any],
    max_slots: int,
) -> None:
    """Create the slot table and seed slot rows if needed.

    Args:
        pool: Async connection pool for Postgres.
        max_slots: Maximum number of slots to seed.

    Raises:
        ValueError: If ``max_slots`` is not positive.

    """
    if max_slots <= 0:
        msg = "max_slots must be a positive integer."
        raise ValueError(msg)

    async with pool.connection() as conn:
        await conn.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {} (
                    slot_id INT PRIMARY KEY,
                    acquired_at TIMESTAMPTZ NULL,
                    run_id TEXT NULL,
                    case_id TEXT NULL
                );
                """,
            ).format(sql.Identifier(_TABLE_NAME)),
        )

        await conn.execute(
            sql.SQL(
                """
                INSERT INTO {} (slot_id)
                SELECT slot_id
                FROM generate_series(1, {}) AS slot_id
                ON CONFLICT (slot_id) DO NOTHING;
                """,
            ).format(sql.Identifier(_TABLE_NAME), sql.Literal(max_slots)),
        )


async def acquire_schema_slot(    # noqa: PLR0913
    *,
    pool: AsyncConnectionPool[Any],
    run_id: str,
    case_id: str,
    max_slots: int,
    stale_after_s: int,
    wait_timeout_s: float,
    poll_interval_s: float,
) -> int:
    """Acquire a schema slot, blocking until available or timeout.

    Args:
        pool: Async connection pool for Postgres.
        run_id: Run identifier for diagnostic tracking.
        case_id: Case identifier for diagnostic tracking.
        max_slots: Maximum number of slots to seed.
        stale_after_s: Seconds after which a slot is considered stale.
        wait_timeout_s: Max seconds to wait for a slot before failing.
        poll_interval_s: Sleep duration between retries when no slot is free.

    Returns:
        The acquired slot id.

    Raises:
        RuntimeError: If no slot becomes available before timeout.

    """
    await ensure_schema_slot_table(pool=pool, max_slots=max_slots)

    if wait_timeout_s <= 0:
        msg = "wait_timeout_s must be positive."
        raise RuntimeError(msg)

    deadline = asyncio.get_running_loop().time() + wait_timeout_s
    while True:
        stale_before = datetime.now(UTC) - timedelta(seconds=stale_after_s)
        row = None
        async with pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
                await cur.execute(
                    sql.SQL(
                        """
                            WITH candidate AS (
                                SELECT slot_id
                                FROM {}
                                WHERE acquired_at IS NULL OR acquired_at < {}
                                ORDER BY slot_id
                                FOR UPDATE SKIP LOCKED
                                LIMIT 1
                            )
                            UPDATE {} AS s
                            SET acquired_at = NOW(),
                                run_id = {},
                                case_id = {}
                            WHERE s.slot_id = (SELECT slot_id FROM candidate)
                            RETURNING s.slot_id;
                            """,
                    ).format(
                        sql.Identifier(_TABLE_NAME),
                        sql.Literal(stale_before),
                        sql.Identifier(_TABLE_NAME),
                        sql.Literal(run_id),
                        sql.Literal(case_id),
                    ),
                )
                row = await cur.fetchone()

        if row and row[0] is not None:
            return int(row[0])

        if asyncio.get_running_loop().time() >= deadline:
            msg = "Timed out waiting for an evaluation schema slot."
            raise RuntimeError(msg)

        await asyncio.sleep(max(0.1, poll_interval_s))


async def release_schema_slot(
    *,
    pool: AsyncConnectionPool[Any],
    slot_id: int,
) -> None:
    """Release a previously acquired schema slot.

    Args:
        pool: Async connection pool for Postgres.
        slot_id: Slot identifier to release.

    """
    async with pool.connection() as conn:
        await conn.execute(
            sql.SQL(
                """
                UPDATE {}
                SET acquired_at = NULL,
                    run_id = NULL,
                    case_id = NULL
                WHERE slot_id = {};
                """,
            ).format(sql.Identifier(_TABLE_NAME), sql.Literal(slot_id)),
        )
