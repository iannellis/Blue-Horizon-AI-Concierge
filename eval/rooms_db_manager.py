"""Neon branch reset helpers and database pool utilities for eval and stress runs."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import platform
from typing import Any, Final

import httpx
from dotenv import load_dotenv
from psycopg_pool import AsyncConnectionPool

from eval.config import NeonConfig

logger = logging.getLogger(__name__)

_HTTP_LOCKED: Final[int] = 423


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

    Finds the named branch and its parent within the given project, then issues
    a restore request so the branch returns to its parent's data state.

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
    if not api_key:
        msg = "api_key is required but was not provided (check NEON_API_KEY)."
        raise RuntimeError(msg)
    base_url = "https://console.neon.tech/api/v2"
    headers = {"Authorization": f"Bearer {api_key}"}
    async with httpx.AsyncClient(headers=headers) as client:
        branch_id, parent_id = await _find_neon_branch(
            client, base_url, neon_cfg.project_id, neon_cfg.branch_name,
        )
        await _restore_neon_branch(
            client,
            base_url,
            neon_cfg.project_id,
            branch_id,
            parent_id,
            lock_retry_attempts=neon_cfg.lock_retry_attempts,
            lock_retry_delay_s=neon_cfg.lock_retry_delay_s,
        )
    logger.info(
        "Neon branch %r in project %r reset to parent.",
        neon_cfg.branch_name,
        neon_cfg.project_id,
    )


async def _find_neon_branch(
    client: httpx.AsyncClient,
    base_url: str,
    project_id: str,
    branch_name: str,
) -> tuple[str, str]:
    """Resolve a branch name to its ID and parent branch ID.

    Args:
        client: Authenticated ``httpx.AsyncClient`` for the Neon API.
        base_url: Neon API v2 base URL.
        project_id: ID of the project that owns the branch.
        branch_name: Display name of the branch to locate.

    Returns:
        A tuple of ``(branch_id, parent_id)``.

    Raises:
        RuntimeError: If the branch is not found or has no parent.
        httpx.HTTPStatusError: If the Neon API returns a non-2xx response.

    """
    response = await client.get(f"{base_url}/projects/{project_id}/branches")
    response.raise_for_status()
    branches: list[dict[str, object]] = response.json().get("branches", [])
    for branch in branches:
        if branch.get("name") == branch_name:
            branch_id = str(branch["id"])
            parent_id = branch.get("parent_id")
            if not parent_id:
                msg = f"Branch {branch_name!r} has no parent to restore from."
                raise RuntimeError(msg)
            return branch_id, str(parent_id)
    msg = f"Neon branch not found: {branch_name!r}"
    raise RuntimeError(msg)


async def _restore_neon_branch(  # noqa: PLR0913
    client: httpx.AsyncClient,
    base_url: str,
    project_id: str,
    branch_id: str,
    parent_id: str,
    *,
    lock_retry_attempts: int,
    lock_retry_delay_s: float,
) -> None:
    """Call the Neon restore endpoint to reset a branch to a parent.

    Retries up to ``lock_retry_attempts`` times when the branch is locked
    (HTTP 423), waiting ``lock_retry_delay_s`` seconds between attempts.
    This handles ongoing Neon operations that block a restore request.

    Args:
        client: Authenticated ``httpx.AsyncClient`` for the Neon API.
        base_url: Neon API v2 base URL.
        project_id: ID of the project that owns the branch.
        branch_id: ID of the branch to restore.
        parent_id: ID of the parent branch to restore from.
        lock_retry_attempts: Maximum number of retry attempts on HTTP 423.
        lock_retry_delay_s: Seconds to wait between lock retry attempts.

    Raises:
        httpx.HTTPStatusError: If the Neon API returns a non-2xx response
            after all retry attempts are exhausted.

    """
    url = f"{base_url}/projects/{project_id}/branches/{branch_id}/restore"
    body: dict[str, str] = {"source_branch_id": parent_id}
    if lock_retry_attempts < 1:
        msg = "lock_retry_attempts must be at least 1"
        raise ValueError(msg)
    for attempt in range(lock_retry_attempts):
        response = await client.post(url, json=body)
        if response.status_code != _HTTP_LOCKED:
            response.raise_for_status()
            return
        is_last_attempt = attempt >= lock_retry_attempts - 1
        if is_last_attempt:
            response.raise_for_status()
            return
        logger.warning(
            "Neon branch %r is locked (attempt %d/%d); waiting %.0fs.",
            branch_id,
            attempt + 1,
            lock_retry_attempts,
            lock_retry_delay_s,
        )
        await asyncio.sleep(lock_retry_delay_s)


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
