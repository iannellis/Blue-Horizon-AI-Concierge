"""Shared operational errors used by the agents."""


class OperationalError(RuntimeError):
    """Raised for expected operational failures in agent resources.

    These errors represent transient, recoverable issues such as dependency
    outages or connectivity problems. Consumers should generally log the
    exception and return a safe response instead of terminating the process.

    """


class ConfigurationError(RuntimeError):
    """Raised for permanent misconfiguration that retrying cannot fix.

    Deliberately a **sibling** of `OperationalError`, not a subclass: broad
    `except OperationalError` handlers exist specifically to catch transient
    failures and keep retrying, and must not silently swallow a defect that
    no amount of retrying resolves (e.g. `PGSQL_RO_DB_URL` pointed at a
    writable role, per invariant 7, or a missing prompt resource). Consumers
    that genuinely want to treat both the same way still can, via
    `except (OperationalError, ConfigurationError)`.

    """


class ThreadCustomerMismatchError(RuntimeError):
    """Raised when a `thread_id` is reused with a different `customer_id`.

    Each conversation thread is bound to whichever guest first uses it. A
    mismatch means a client is replaying or guessing another guest's
    `thread_id`, which -- once guests are distinct people holding real
    reservations -- would otherwise expose their room numbers and dates.

    """
