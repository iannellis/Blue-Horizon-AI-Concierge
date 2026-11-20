"""Run a room search against a read-only pool, with retry and error classification.

`search.py` owns what a search is: its argument model and its fixed queries.
This module owns how one is run: borrowing a connection, retrying a transient
connection failure, logging the outcome, and turning a failure into a bounded
error result instead of an exception. It takes the pool and its settings as
arguments rather than reading them from `BookingSqlResources`, so the booking
agent and an MCP server share one implementation.

It imports no LangChain code, for the same reason `search.py` does not.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Literal

import psycopg
from psycopg_pool import PoolTimeout
from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from blue_horizon.agents.booking.db_utils import is_transient_conn_error
from blue_horizon.agents.booking.search import run_room_search

if TYPE_CHECKING:
    from psycopg_pool import AsyncConnectionPool
    from pydantic import BaseModel

    from blue_horizon.config import DbRetryConfig

logger = logging.getLogger(__name__)

# Coarse classification of a search_rooms failure, independent of message
# text, so evaluators (see eval/stress/workload.py) can assert on structure
# rather than matching prose that guest-facing or model-facing copy changes
# could move. "invalid_arguments" is produced by the tool layer, which rejects
# arguments before this module sees them.
SqlErrorKind = Literal["unavailable", "invalid_arguments", "unexpected"]

# Exceptions that mean the connection, not the query, failed. Only these are
# candidates for a retry, and only when `is_transient_conn_error` agrees.
_CONN_ERRORS = (
    psycopg.OperationalError,
    psycopg.InterfaceError,
    PoolTimeout,
    TimeoutError,
)


async def search_rooms_with_retry(  # noqa: PLR0913
    pool: AsyncConnectionPool[Any],
    args: BaseModel,
    *,
    retry: DbRetryConfig,
    checkout_timeout_s: float,
    max_results: int,
    failure_message: str,
) -> dict[str, Any]:
    """Run a room search and return a bounded result.

    Each attempt borrows a connection from *pool* and runs the fixed query
    `search.run_room_search` selects for *args*. A search is a read, so it
    is always safe to retry on a transient connection error.

    Args:
        pool: An open pool authenticating as the read-only role.
        args: A validated instance of the model `build_search_args_model`
            returns.
        retry: Transient retry policy: attempt count and backoff.
        checkout_timeout_s: Seconds to wait for a connection from *pool*.
        max_results: Hard cap on the number of rooms returned.
        failure_message: The ``error`` text of a failed search. It is
            written for whoever reads the result: the booking agent's text
            carries a marker its system prompt keys off, which a client
            bringing its own prompt has never seen. ``error_kind``
            distinguishes the failures without depending on this text.

    Returns:
        On success, `run_room_search`'s result: ``status="ok"``,
        ``matching_count``, ``rooms``, and ``limit_note`` when the
        requested ``limit`` was clamped. On failure, ``status="error"``
        with ``error`` set to *failure_message* and ``error_kind``.

    """
    # Spans every attempt and backoff, so a cold Neon start shows up here.
    started = time.perf_counter()
    try:
        async for attempt in AsyncRetrying(
            retry=retry_if_exception(_is_retryable),
            stop=stop_after_attempt(retry.max_transient_retries + 1),
            wait=wait_exponential(multiplier=retry.transient_retry_backoff_s),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        ):
            with attempt:
                result = await _search_once(
                    pool,
                    args,
                    checkout_timeout_s=checkout_timeout_s,
                    max_results=max_results,
                )
                # The arguments, never the rooms: the counts are all an
                # audit needs to see what the model looked at.
                timing = _timing(started)
                logger.info(
                    "search_rooms ok: matching_count=%s returned=%s "
                    "duration_ms=%s args=%r",
                    result["matching_count"],
                    len(result["rooms"]),
                    timing["duration_ms"],
                    args,
                    extra=timing,
                )
                return result

    except _CONN_ERRORS as exc:
        # One line, no traceback: the stack under a pool checkout is
        # psycopg_pool internals and says nothing the exception does not.
        timing = _timing(started)
        logger.warning(
            "search_rooms connection error after retries: duration_ms=%s %r",
            timing["duration_ms"],
            exc,
            extra=timing,
        )
        return _sql_error_result(failure_message, error_kind="unavailable")

    except Exception:
        # The SQL is fixed and the arguments validated, so anything else
        # (a psycopg.Error included) is a bug, not something the caller
        # can fix by searching differently.
        timing = _timing(started)
        logger.exception(
            "search_rooms unexpected failure: duration_ms=%s args=%r",
            timing["duration_ms"],
            args,
            extra=timing,
        )
        return _sql_error_result(failure_message, error_kind="unexpected")

    # Unreachable: AsyncRetrying either returns from the loop body or
    # reraises. Kept so a broken assumption fails closed, not silently.
    return _sql_error_result(failure_message, error_kind="unexpected")


def _is_retryable(exc: BaseException) -> bool:
    """Decide whether a failed attempt is worth retrying.

    Args:
        exc: The exception the attempt raised.

    Returns:
        bool: True for a connection-level error that `is_transient_conn_error`
        classifies as transient.

    """
    return isinstance(exc, _CONN_ERRORS) and is_transient_conn_error(exc)


async def _search_once(
    pool: AsyncConnectionPool[Any],
    args: BaseModel,
    *,
    checkout_timeout_s: float,
    max_results: int,
) -> dict[str, Any]:
    """Run the search exactly once without any retry logic.

    Args:
        pool: An open pool authenticating as the read-only role.
        args: Validated search arguments.
        checkout_timeout_s: Seconds to wait for a connection from *pool*.
        max_results: Hard cap on the number of rooms returned.

    Returns:
        `run_room_search`'s result.

    Raises:
        psycopg.OperationalError: On connection-level failures.
        psycopg.InterfaceError: On connection-level failures.
        psycopg_pool.PoolTimeout: When a pool connection cannot be acquired.
        TimeoutError: On network timeout.
        psycopg.Error: On any other database error.

    """
    async with (
        pool.connection(timeout=checkout_timeout_s) as conn,
        conn.transaction(),
    ):
        # Belt-and-braces: with a genuinely read-only role this can never
        # engage, but it costs nothing and covers the window between a
        # misconfiguration and the next restart's startup assertion.
        await conn.execute("SET TRANSACTION READ ONLY")
        return await run_room_search(conn, args, max_results=max_results)


def _timing(started: float) -> dict[str, int]:
    """Measure the time since `started`, as a log record's extra fields.

    Args:
        started: `time.perf_counter` reading when the work began.

    Returns:
        dict[str, int]: ``duration_ms``, whole milliseconds elapsed.

    """
    return {"duration_ms": round((time.perf_counter() - started) * 1000)}


def _sql_error_result(error: str, *, error_kind: SqlErrorKind) -> dict[str, Any]:
    """Build a standard `search_rooms` error result dict.

    Args:
        error: The error message to include in the result.
        error_kind: Coarse, message-independent classification of the
            failure.

    Returns:
        A result dict with ``status="error"`` and no rooms.

    """
    return {
        "status": "error",
        "matching_count": 0,
        "rooms": [],
        "error": error,
        "error_kind": error_kind,
    }
