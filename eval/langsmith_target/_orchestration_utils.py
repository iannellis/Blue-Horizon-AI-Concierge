"""Shared orchestration helpers for eval and stress harnesses.

This module centralizes readiness polling for ``OrchestrationManager``-like
objects so the evaluation target and the stress runner use identical timeout
behavior.
"""

from __future__ import annotations

import asyncio
from typing import Protocol


class SupportsReadyState(Protocol):
    """Protocol for managers exposing an ``is_ready`` property."""

    @property
    def is_ready(self) -> bool:
        """Return whether the manager is ready to accept requests."""
        ...


async def wait_for_orchestration_ready(
    orchestration: SupportsReadyState,
    *,
    timeout_s: float,
    manager_name: str = "OrchestrationManager",
    poll_interval_s: float = 0.2,
) -> None:
    """Wait until an orchestration manager reports readiness.

    Args:
        orchestration: Manager-like object exposing an ``is_ready`` property.
        timeout_s: Maximum number of seconds to wait before failing.
        manager_name: Human-readable manager name used in timeout errors.
        poll_interval_s: Delay between readiness checks.

    Raises:
        TimeoutError: If readiness is not reached within ``timeout_s`` seconds.

    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s

    while not orchestration.is_ready:
        if loop.time() >= deadline:
            msg = f"{manager_name} did not become ready within {timeout_s:.1f}s."
            raise TimeoutError(msg)
        await asyncio.sleep(poll_interval_s)
