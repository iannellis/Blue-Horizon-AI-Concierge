"""Neon branch reset helpers and database pool utilities for eval and stress runs."""

from __future__ import annotations

import argparse
import asyncio
import logging
import platform
from typing import Any, Final

import httpx
from psycopg_pool import AsyncConnectionPool

from eval.config import load_eval_config

logger = logging.getLogger(__name__)

_HTTP_LOCKED: Final[int] = 423
_LOCK_RETRY_ATTEMPTS: Final[int] = 8
_LOCK_RETRY_DELAY_S: Final[float] = 5.0


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


async def reset_neon_branch(*, project_id: str, branch_name: str) -> None:
    """Reset a Neon branch to its parent baseline state via the Neon API.

    Finds the named branch and its parent within the given project, then issues
    a restore request so the branch returns to its parent's data state.

    Args:
        project_id: Neon project ID (visible in the console URL:
            ``console.neon.tech/app/projects/<project_id>``).
        branch_name: Name of the branch to reset (e.g. ``"development"``).

    Raises:
        RuntimeError: If ``NEON_API_KEY`` is not set, or the branch cannot be
            found, or the branch has no parent to restore from.
        httpx.HTTPStatusError: If the Neon API returns a non-2xx response.

    """
    api_key = load_eval_config().neon_api_key
    if not api_key:
        msg = "NEON_API_KEY environment variable is not set."
        raise RuntimeError(msg)
    base_url = "https://console.neon.tech/api/v2"
    headers = {"Authorization": f"Bearer {api_key}"}
    async with httpx.AsyncClient(headers=headers) as client:
        branch_id, parent_id = await _find_neon_branch(
            client, base_url, project_id, branch_name,
        )
        await _restore_neon_branch(client, base_url, project_id, branch_id, parent_id)
    logger.info(
        "Neon branch %r in project %r reset to parent.", branch_name, project_id,
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


async def _restore_neon_branch(
    client: httpx.AsyncClient,
    base_url: str,
    project_id: str,
    branch_id: str,
    parent_id: str,
) -> None:
    """Call the Neon restore endpoint to reset a branch to a parent.

    Retries up to ``_LOCK_RETRY_ATTEMPTS`` times when the branch is locked
    (HTTP 423), waiting ``_LOCK_RETRY_DELAY_S`` seconds between attempts.
    This handles ongoing Neon operations that block a restore request.

    Args:
        client: Authenticated ``httpx.AsyncClient`` for the Neon API.
        base_url: Neon API v2 base URL.
        project_id: ID of the project that owns the branch.
        branch_id: ID of the branch to restore.
        parent_id: ID of the parent branch to restore from.

    Raises:
        httpx.HTTPStatusError: If the Neon API returns a non-2xx response
            after all retry attempts are exhausted.

    """
    url = f"{base_url}/projects/{project_id}/branches/{branch_id}/restore"
    body: dict[str, str] = {"source_branch_id": parent_id}
    for attempt in range(_LOCK_RETRY_ATTEMPTS):
        response = await client.post(url, json=body)
        if response.status_code != _HTTP_LOCKED:
            response.raise_for_status()
            return
        if attempt < _LOCK_RETRY_ATTEMPTS - 1:
            logger.warning(
                "Neon branch %r is locked (attempt %d/%d); waiting %.0fs.",
                branch_id,
                attempt + 1,
                _LOCK_RETRY_ATTEMPTS,
                _LOCK_RETRY_DELAY_S,
            )
            await asyncio.sleep(_LOCK_RETRY_DELAY_S)
    response.raise_for_status()


async def _main() -> None:
    """Parse CLI args and reset the Neon branch for evaluation.

    Accepts ``--project-id`` and ``--branch-name`` to override the defaults.
    Reads ``NEON_API_KEY`` from the environment.

    """
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
    await reset_neon_branch(
        project_id=args.project_id,
        branch_name=args.branch_name,
    )


if __name__ == "__main__":
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(_main())
