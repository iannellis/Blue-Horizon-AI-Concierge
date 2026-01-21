"""Shared operational errors used by the agents."""


class OperationalError(RuntimeError):
    """Raised for expected operational failures in agent resources.

    These errors represent transient, recoverable issues such as dependency
    outages or connectivity problems. Consumers should generally log the
    exception and return a safe response instead of terminating the process.

    """
