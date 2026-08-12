"""Neon branch reset utility for production use.

Resets a named Neon branch to its parent's data state via the Neon
management API.  Used by the API's reset endpoint so users can clear
their bookings and return the database to its baseline state.

The restore API call itself is asynchronous on Neon's side: it returns as
soon as the request is accepted, before the branch's data (and anything
stored in it, including role grants) has actually been copied over and the
compute has caught up. `reset_branch` polls the returned operation(s) until
Neon reports them "finished" so callers can trust the branch is actually
ready once this returns.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from blue_horizon.config import NeonConfig

_log = logging.getLogger(__name__)

_BASE_URL: str = "https://console.neon.tech/api/v2"
_HTTP_LOCKED: int = 423
_OPERATION_STATUS_FINISHED: str = "finished"
_OPERATION_STATUSES_FAILED: frozenset[str] = frozenset({"failed", "cancelled"})


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
            found, the branch has no parent to restore from, the restore
            operation fails, or it does not finish within
            ``neon_cfg.operation_poll_timeout_s``.
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
            operation_poll_interval_s=neon_cfg.operation_poll_interval_s,
            operation_poll_timeout_s=neon_cfg.operation_poll_timeout_s,
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
    operation_poll_interval_s: float,
    operation_poll_timeout_s: float,
) -> None:
    """Call the Neon restore endpoint to reset a branch to its parent.

    Retries up to ``lock_retry_attempts`` times when the branch is locked
    (HTTP 423), waiting ``lock_retry_delay_s`` seconds between attempts. Once
    Neon accepts the restore request, waits for every operation it returns to
    reach a terminal status before returning -- the branch is not actually
    restored yet at the moment the request is accepted.

    Args:
        client: Authenticated ``httpx.AsyncClient`` for the Neon API.
        project_id: ID of the project that owns the branch.
        branch_id: ID of the branch to restore.
        parent_id: ID of the parent branch to restore from.
        lock_retry_attempts: Maximum retry attempts on HTTP 423.
        lock_retry_delay_s: Seconds to wait between retry attempts.
        operation_poll_interval_s: Seconds to wait between operation-status
            polls.
        operation_poll_timeout_s: Maximum seconds to wait for the restore
            operation(s) to finish.

    Raises:
        RuntimeError: If a restore operation fails or does not finish within
            ``operation_poll_timeout_s``.
        httpx.HTTPStatusError: If the Neon API returns a non-2xx response
            after all retry attempts are exhausted.

    """
    url = f"{_BASE_URL}/projects/{project_id}/branches/{branch_id}/restore"
    body: dict[str, str] = {"source_branch_id": parent_id}
    for attempt in range(lock_retry_attempts):
        response = await client.post(url, json=body)
        if response.status_code != _HTTP_LOCKED:
            response.raise_for_status()
            await _wait_for_operations(
                client,
                project_id,
                response.json().get("operations", []),
                poll_interval_s=operation_poll_interval_s,
                timeout_s=operation_poll_timeout_s,
            )
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


async def _wait_for_operations(
    client: httpx.AsyncClient,
    project_id: str,
    operations: list[dict[str, Any]],
    *,
    poll_interval_s: float,
    timeout_s: float,
) -> None:
    """Wait for every operation from a Neon API response to finish.

    Args:
        client: Authenticated ``httpx.AsyncClient`` for the Neon API.
        project_id: ID of the project that owns the operations.
        operations: The ``operations`` list from a Neon API response, each a
            dict expected to have an ``id`` key. Entries without an ``id``
            are skipped.
        poll_interval_s: Seconds to wait between polls of each operation.
        timeout_s: Maximum seconds to wait for each operation to finish.

    Raises:
        RuntimeError: If an operation fails or does not finish within
            ``timeout_s``.
        httpx.HTTPStatusError: If the Neon API returns a non-2xx response.

    """
    for operation in operations:
        operation_id = operation.get("id")
        if not operation_id:
            continue
        await _wait_for_operation(
            client,
            project_id,
            str(operation_id),
            poll_interval_s=poll_interval_s,
            timeout_s=timeout_s,
        )


async def _wait_for_operation(
    client: httpx.AsyncClient,
    project_id: str,
    operation_id: str,
    *,
    poll_interval_s: float,
    timeout_s: float,
) -> None:
    """Poll a single Neon operation until it reaches a terminal status.

    Args:
        client: Authenticated ``httpx.AsyncClient`` for the Neon API.
        project_id: ID of the project that owns the operation.
        operation_id: ID of the operation to poll.
        poll_interval_s: Seconds to wait between polls.
        timeout_s: Maximum seconds to wait before giving up.

    Raises:
        RuntimeError: If the operation reaches a failed/cancelled status, or
            does not finish within ``timeout_s``.
        httpx.HTTPStatusError: If the Neon API returns a non-2xx response.

    """
    url = f"{_BASE_URL}/projects/{project_id}/operations/{operation_id}"
    deadline = time.monotonic() + timeout_s
    while True:
        response = await client.get(url)
        response.raise_for_status()
        status = response.json().get("operation", {}).get("status")
        if status == _OPERATION_STATUS_FINISHED:
            return
        if status in _OPERATION_STATUSES_FAILED:
            msg = f"Neon operation {operation_id!r} ended with status {status!r}."
            raise RuntimeError(msg)
        if time.monotonic() >= deadline:
            msg = (
                f"Timed out after {timeout_s}s waiting for Neon operation "
                f"{operation_id!r} to finish (last status: {status!r})."
            )
            raise RuntimeError(msg)
        await asyncio.sleep(poll_interval_s)
