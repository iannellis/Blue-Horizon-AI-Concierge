"""Shared lifecycle helper for agent resource classes.

`InfoRagResources`, `BookingSqlResources`, and `OrchestrationResources` each
do lightweight setup in `__init__` and defer expensive work (network
connections, agent compilation) to an async `startup_check()`. Until that has
completed -- or if it failed -- several attributes are `None`. `require()` is
the one place that turns "attribute is still None" into a consistent,
user-facing `RuntimeError`, replacing the dozen near-identical
`if x is None: raise RuntimeError(...)` guards this used to be spelled out as.
"""

from __future__ import annotations


def require[T](value: T | None, what: str) -> T:
    """Return `value`, narrowed to non-None, or raise if not initialized yet.

    Args:
        value: The resource to check. `None` means `startup_check()` has not
            been called yet, or a prior call failed.
        what: Human-readable name of the resource, used in the error message
            (e.g. "System prompt", "Read pool").

    Returns:
        T: `value`, guaranteed non-None.

    Raises:
        RuntimeError: If `value` is None.

    """
    if value is None:
        msg = f"{what} is not initialized; call await startup_check() first"
        raise RuntimeError(msg)
    return value
