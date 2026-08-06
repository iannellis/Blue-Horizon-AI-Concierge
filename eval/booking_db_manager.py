"""Neon branch reset helpers and database pool utilities for eval and stress runs."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import platform
from typing import Any

from dotenv import load_dotenv
from psycopg_pool import AsyncConnectionPool

from blue_horizon.config import NeonConfig
from blue_horizon.neon import reset_branch as _reset_branch

logger = logging.getLogger(__name__)


async def open_schema_pool(
    db_url: str,
    *,
    min_size: int = 1,
    max_size: int = 2,
) -> AsyncConnectionPool[Any]:
    """Create and open an async Postgres connection pool.

    Args:
        db_url: The Postgres connection URL.
        min_size: The minimum number of connections in the pool.
        max_size: The maximum number of connections in the pool.

    Returns:
        An open ``AsyncConnectionPool``.

    """
    pool: AsyncConnectionPool[Any] = AsyncConnectionPool(
        conninfo=db_url,
        min_size=min_size,
        max_size=max_size,
        open=False,
    )
    await pool.open()
    return pool


async def reset_neon_branch(neon_cfg: NeonConfig, *, api_key: str | None) -> None:
    """Reset a Neon branch to its parent baseline state via the Neon API.

    Delegates to :func:`blue_horizon.neon.reset_branch`.

    Args:
        neon_cfg: Neon project and branch configuration including project ID,
            branch name, and retry settings.
        api_key: Neon management API key. Typically sourced from the
            ``NEON_API_KEY`` environment variable via the caller's config.

    Raises:
        RuntimeError: If ``api_key`` is not set, or the branch cannot be
            found, or the branch has no parent to restore from.
        httpx.HTTPStatusError: If the Neon API returns a non-2xx response.

    """
    await _reset_branch(neon_cfg, api_key=api_key)


async def _main() -> None:
    """Parse CLI args and reset the Neon branch for evaluation.

    Accepts ``--project-id`` and ``--branch-name`` to identify the branch.
    Retry settings use the ``NeonConfig`` defaults. Reads ``NEON_API_KEY``
    from the environment or a ``.env`` file.

    """
    load_dotenv()
    parser = argparse.ArgumentParser(description="Reset a Neon branch for evaluation.")
    parser.add_argument(
        "--project-id",
        default="",
        help="Neon project ID (from the console URL).",
    )
    parser.add_argument(
        "--branch-name",
        default="development",
        help="Neon branch name to reset.",
    )
    args = parser.parse_args()
    neon_cfg = NeonConfig(project_id=args.project_id, branch_name=args.branch_name)
    await reset_neon_branch(neon_cfg, api_key=os.environ.get("NEON_API_KEY"))


if __name__ == "__main__":
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(_main())
