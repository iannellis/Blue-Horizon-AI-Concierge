"""Neon branch reset utility for production use.

Resets a named Neon branch to its parent's data state via the Neon
management API.  Used by the API's reset endpoint so users can clear
their bookings and return the database to its baseline state.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from blue_horizon.config import NeonConfig

_log = logging.getLogger(__name__)

_BASE_URL: str = "https://console.neon.tech/api/v2"
_HTTP_LOCKED: int = 423


async def reset_branch(neon_cfg: NeonConfig, *, api_key: str | None) -> None:
    """Reset a Neon branch to its parent baseline state via the Neon API.

    Looks up the named branch within the project, then issues a restore
    request so it returns to its parent's data state.

    Args:
        neon_cfg: Neon branch configuration (project ID, branch name, retry
            settings).
        api_key: Neon management API key.

    Raises:
        RuntimeError: If ``api_key`` is not provided, the branch is not
            found, or the branch has no parent to restore from.
        httpx.HTTPStatusError: If the Neon API returns a non-2xx response
            after all retry attempts are exhausted.

    """
    if not api_key:
        msg = "api_key is required but was not provided (check NEON_API_KEY)."
        raise RuntimeError(msg)
    headers = {"Authorization": f"Bearer {api_key}"}
    async with httpx.AsyncClient(headers=headers) as client:
        branch_id, parent_id = await _find_branch(
            client, neon_cfg.project_id, neon_cfg.branch_name,
        )
        await _restore_branch(
            client,
            neon_cfg.project_id,
            branch_id,
            parent_id,
            lock_retry_attempts=neon_cfg.lock_retry_attempts,
            lock_retry_delay_s=neon_cfg.lock_retry_delay_s,
        )
    _log.info(
        "Neon branch %r in project %r reset to parent.",
        neon_cfg.branch_name,
        neon_cfg.project_id,
    )


async def _find_branch(
    client: httpx.AsyncClient,
    project_id: str,
    branch_name: str,
) -> tuple[str, str]:
    """Resolve a branch name to its ID and parent branch ID.

    Args:
        client: Authenticated ``httpx.AsyncClient`` for the Neon API.
        project_id: ID of the project that owns the branch.
        branch_name: Display name of the branch to locate.

    Returns:
        A tuple of ``(branch_id, parent_id)``.

    Raises:
        RuntimeError: If the branch is not found or has no parent.
        httpx.HTTPStatusError: If the Neon API returns a non-2xx response.

    """
    response = await client.get(f"{_BASE_URL}/projects/{project_id}/branches")
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


async def _restore_branch(  # noqa: PLR0913
    client: httpx.AsyncClient,
    project_id: str,
    branch_id: str,
    parent_id: str,
    *,
    lock_retry_attempts: int,
    lock_retry_delay_s: float,
) -> None:
    """Call the Neon restore endpoint to reset a branch to its parent.

    Retries up to ``lock_retry_attempts`` times when the branch is locked
    (HTTP 423), waiting ``lock_retry_delay_s`` seconds between attempts.

    Args:
        client: Authenticated ``httpx.AsyncClient`` for the Neon API.
        project_id: ID of the project that owns the branch.
        branch_id: ID of the branch to restore.
        parent_id: ID of the parent branch to restore from.
        lock_retry_attempts: Maximum retry attempts on HTTP 423.
        lock_retry_delay_s: Seconds to wait between retry attempts.

    Raises:
        httpx.HTTPStatusError: If the Neon API returns a non-2xx response
            after all retry attempts are exhausted.

    """
    url = f"{_BASE_URL}/projects/{project_id}/branches/{branch_id}/restore"
    body: dict[str, str] = {"source_branch_id": parent_id}
    for attempt in range(lock_retry_attempts):
        response = await client.post(url, json=body)
        if response.status_code != _HTTP_LOCKED:
            response.raise_for_status()
            return
        is_last_attempt = attempt >= lock_retry_attempts - 1
        if is_last_attempt:
            response.raise_for_status()
            return
        _log.warning(
            "Neon branch %r is locked (attempt %d/%d); waiting %.0fs.",
            branch_id,
            attempt + 1,
            lock_retry_attempts,
            lock_retry_delay_s,
        )
        await asyncio.sleep(lock_retry_delay_s)
