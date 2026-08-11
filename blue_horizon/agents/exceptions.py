"""Shared operational errors used by the agents."""


class OperationalError(RuntimeError):
    """Raised for expected operational failures in agent resources.

    These errors represent transient, recoverable issues such as dependency
    outages or connectivity problems. Consumers should generally log the
    exception and return a safe response instead of terminating the process.

    """


class ThreadCustomerMismatchError(RuntimeError):
    """Raised when a `thread_id` is reused with a different `customer_id`.

    Each conversation thread is bound to whichever guest first uses it. A
    mismatch means a client is replaying or guessing another guest's
    `thread_id`, which -- once guests are distinct people holding real
    reservations -- would otherwise expose their room numbers and dates.

    """
