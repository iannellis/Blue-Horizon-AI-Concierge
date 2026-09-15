"""Tests for `BookingSqlResources.execute_sql`'s retry and error classification.

No real database connection is required for most of this file: the read pool
is a fake whose `.connection()` replays a scripted sequence of outcomes, which
exercises the retry loop, error-kind classification (`error_kind`, added
alongside this file so evaluators can assert on failure *shape* rather than
matching prose -- see `blue_horizon.agents.booking.resources.SqlErrorKind`),
and the guardrail short-circuit in `execute_sql()` directly.

`TestConnectFailureSurfacesAsPoolTimeout` is the one exception: it drives a
real `psycopg_pool.AsyncConnectionPool` against a connection class that always
fails to connect, to pin (per the plan's phase 3 note on the cold-start
classifier gap) which exception a checkout actually sees when a connection
cannot be *established* at all -- the Neon wake-up case, as opposed to a
connection dying mid-use (the Neon suspend case, already covered by the
message-pattern tests in `tests/booking/test_db_utils.py`).
"""

# ruff: noqa: S101

from __future__ import annotations

import asyncio
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import psycopg
import pytest
from psycopg_pool import AsyncConnectionPool, PoolTimeout

from blue_horizon.agents.booking.resources import (
    BookingSqlResources,
    _check_credentials,
)
from blue_horizon.agents.exceptions import ConfigurationError
from blue_horizon.config import BookingSqlConfig

_BOOKING_CONFIG_DICT: dict[str, Any] = {
    "llm": {
        "model": "gpt-5-mini",
        "reasoning_effort": "low",
        "timeout_s": 20.0,
        "max_retries": 2,
    },
    "agent": {"top_k": 4},
    "prompts": {
        "folder": "system_prompts",
        "system_prompt_filename": "rooms_sql_prompt.txt",
    },
    "db": {
        "pool": {
            "min_size": 0,
            "max_size": 10,
            "timeout_s": 5.0,
            "max_idle_s": 240.0,
            "reconnect_timeout_s": 30.0,
        },
        "guardrails": {"max_rows": 50, "allow_only_hotel_tables": True},
        "retry": {"max_transient_retries": 2, "transient_retry_backoff_s": 0.001},
    },
    "proposals": {"ttl_s": 1800.0},
}

_RESOURCES_LOGGER = "blue_horizon.agents.booking.resources"
_SELECT_QUERY = "SELECT room_number FROM rooms"


def _make_resources() -> BookingSqlResources:
    """Build a `BookingSqlResources` with no DB access performed yet.

    `__init__` only stores config and resolves the prompt resource path (pure
    string joining, no filesystem access), so this is safe to call directly
    in a unit test. The pool is attached separately by each test.

    Returns:
        A `BookingSqlResources` instance with `pool` still `None`.

    """
    config = BookingSqlConfig.model_validate(_BOOKING_CONFIG_DICT)
    return BookingSqlResources(
        config=config,
        pgsql_ro_db_url="postgresql://example/ro",
        pgsql_rw_db_url="postgresql://example/rw",
    )


def _successful_connection_cm() -> MagicMock:
    """Build a connection context manager that returns zero rows.

    Returns:
        A mock usable as the return value of `pool.connection(timeout=...)`.

    """
    cursor = MagicMock()
    cursor.__aenter__ = AsyncMock(return_value=cursor)
    cursor.__aexit__ = AsyncMock(return_value=False)
    cursor.execute = AsyncMock()
    cursor.fetchall = AsyncMock(return_value=[])
    # A non-None description routes _execute_once() to the fetchall() branch,
    # matching a real SELECT rather than a description-less statement.
    cursor.description = ["room_number"]

    transaction_cm = MagicMock()
    transaction_cm.__aenter__ = AsyncMock(return_value=None)
    transaction_cm.__aexit__ = AsyncMock(return_value=False)

    conn = MagicMock()
    conn.__aenter__ = AsyncMock(return_value=conn)
    conn.__aexit__ = AsyncMock(return_value=False)
    conn.execute = AsyncMock()
    conn.transaction = MagicMock(return_value=transaction_cm)
    conn.cursor = MagicMock(return_value=cursor)

    connection_cm = MagicMock()
    connection_cm.__aenter__ = AsyncMock(return_value=conn)
    connection_cm.__aexit__ = AsyncMock(return_value=False)
    return connection_cm


def _failing_connection_cm(exc: BaseException) -> MagicMock:
    """Build a connection context manager whose `__aenter__` raises.

    Simulates a failure at the pool-acquisition boundary itself, mirroring
    the pattern already used in `tests/booking/test_write_ops_unavailable.py`.

    Args:
        exc: The exception `.connection().__aenter__()` should raise.

    Returns:
        A mock usable as the return value of `pool.connection(timeout=...)`.

    """
    connection_cm = MagicMock()
    connection_cm.__aenter__ = AsyncMock(side_effect=exc)
    connection_cm.__aexit__ = AsyncMock(return_value=False)
    return connection_cm


def _fake_pool_with_outcomes(outcomes: list[BaseException | None]) -> MagicMock:
    """Build a fake read pool whose successive `.connection()` calls replay outcomes.

    Args:
        outcomes: One entry per expected `execute_sql` attempt, consumed in
            order. `None` means the attempt succeeds with zero rows; an
            exception instance means that attempt's checkout raises it.

    Returns:
        A mock pool. Its `connection` mock's `call_count` is the number of
        checkout attempts actually made, for asserting retry counts.

    """
    remaining = list(outcomes)

    def _connection(*, timeout: float) -> MagicMock:  # noqa: ARG001
        outcome = remaining.pop(0)
        if outcome is None:
            return _successful_connection_cm()
        return _failing_connection_cm(outcome)

    pool = MagicMock()
    pool.connection = MagicMock(side_effect=_connection)
    return pool


# ---------------------------------------------------------------------------
# Guardrail rejection: no DB call at all
# ---------------------------------------------------------------------------


class TestExecuteSqlGuardrailRejection:
    """A guardrail-rejected statement never reaches the pool."""

    def test_non_select_is_rejected_without_a_db_call(self) -> None:
        """`DELETE` is rejected by the statement-type guardrail, not executed."""
        resources = _make_resources()
        resources.pool = _fake_pool_with_outcomes([])  # any call is a failure

        result = asyncio.run(resources.execute_sql("DELETE FROM rooms"))

        assert result["status"] == "error"
        assert result["error_kind"] == "guardrail"
        resources.pool.connection.assert_not_called()


# ---------------------------------------------------------------------------
# Transient connection errors: retried, then classified "unavailable"
# ---------------------------------------------------------------------------


class TestExecuteSqlTransientRetry:
    """Transient connection errors are retried up to `max_transient_retries`."""

    @pytest.mark.parametrize("failures_before_success", [0, 1, 2])
    def test_succeeds_after_n_transient_failures(
        self, failures_before_success: int,
    ) -> None:
        """A transient failure on attempts before the last one is absorbed."""
        outcomes: list[BaseException | None] = [
            PoolTimeout("couldn't get a connection in time"),
        ] * failures_before_success
        outcomes.append(None)
        resources = _make_resources()
        resources.pool = _fake_pool_with_outcomes(outcomes)

        result = asyncio.run(resources.execute_sql(_SELECT_QUERY))

        assert result["status"] == "ok"
        assert resources.pool.connection.call_count == failures_before_success + 1

    def test_retries_exhausted_reports_unavailable(self) -> None:
        """Failing transiently on every attempt exhausts retries as "unavailable"."""
        max_retries = _BOOKING_CONFIG_DICT["db"]["retry"]["max_transient_retries"]
        outcomes: list[BaseException | None] = [
            PoolTimeout("couldn't get a connection in time"),
        ] * (max_retries + 1)
        resources = _make_resources()
        resources.pool = _fake_pool_with_outcomes(outcomes)

        result = asyncio.run(resources.execute_sql(_SELECT_QUERY))

        assert result["status"] == "error"
        assert result["error_kind"] == "unavailable"
        assert result["error"].startswith("DATABASE_UNAVAILABLE:")
        assert resources.pool.connection.call_count == max_retries + 1

    def test_retries_exhausted_logs_one_line_without_traceback(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """An unreachable database logs one warning naming the error, no traceback."""
        max_retries = _BOOKING_CONFIG_DICT["db"]["retry"]["max_transient_retries"]
        outcomes: list[BaseException | None] = [
            PoolTimeout("couldn't get a connection in time"),
        ] * (max_retries + 1)
        resources = _make_resources()
        resources.pool = _fake_pool_with_outcomes(outcomes)

        with caplog.at_level(logging.WARNING, logger=_RESOURCES_LOGGER):
            asyncio.run(resources.execute_sql(_SELECT_QUERY))

        [record] = [
            r for r in caplog.records if "after retries" in r.getMessage()
        ]
        assert "couldn't get a connection in time" in record.getMessage()
        assert record.exc_info is None
        assert isinstance(record.__dict__["duration_ms"], int)

    @pytest.mark.parametrize(
        ("outcome", "prefix"),
        [
            (None, "run_sql ok"),
            (psycopg.errors.SyntaxError("syntax error at or near ..."), "run_sql SQL"),
        ],
    )
    def test_statement_that_reached_the_pool_logs_its_duration(
        self,
        caplog: pytest.LogCaptureFixture,
        outcome: BaseException | None,
        prefix: str,
    ) -> None:
        """A statement's line carries its duration, as text and as an attribute."""
        resources = _make_resources()
        resources.pool = _fake_pool_with_outcomes([outcome])

        with caplog.at_level(logging.INFO, logger=_RESOURCES_LOGGER):
            asyncio.run(resources.execute_sql(_SELECT_QUERY))

        [record] = [r for r in caplog.records if r.getMessage().startswith(prefix)]
        duration_ms = record.__dict__["duration_ms"]
        assert isinstance(duration_ms, int)
        assert f"duration_ms={duration_ms}" in record.getMessage()

    def test_non_retryable_error_is_not_retried(self) -> None:
        """A plain SQL error isn't transient, so only one attempt is made."""
        resources = _make_resources()
        resources.pool = _fake_pool_with_outcomes(
            [psycopg.errors.SyntaxError("syntax error at or near ...")],
        )

        result = asyncio.run(resources.execute_sql(_SELECT_QUERY))

        assert result["status"] == "error"
        assert result["error_kind"] == "sql"
        assert resources.pool.connection.call_count == 1


# ---------------------------------------------------------------------------
# Privilege errors: a blocked write, not a SQL error to diagnose and retry
# ---------------------------------------------------------------------------


class TestExecuteSqlPrivilegeError:
    """A write blocked by the read-only role is classified `privilege`."""

    @pytest.mark.parametrize(
        "exc",
        [
            psycopg.errors.ReadOnlySqlTransaction(
                "cannot execute UPDATE in a read-only transaction",
            ),
            psycopg.errors.InsufficientPrivilege(
                "permission denied for table bookings",
            ),
        ],
    )
    def test_blocked_write_is_privilege_not_sql(self, exc: BaseException) -> None:
        """The result is `error_kind="privilege"`, not the generic SQL-error text.

        Distinguishing this from `error_kind="sql"` matters because the
        prompt's retry rule ("if you get a query error, rewrite and retry")
        applies to SQL errors, not to a blocked write -- see the comment at
        this branch in `resources.py`.
        """
        resources = _make_resources()
        resources.pool = _fake_pool_with_outcomes([exc])

        result = asyncio.run(resources.execute_sql(_SELECT_QUERY))

        assert result["status"] == "error"
        assert result["error_kind"] == "privilege"
        assert not result["error"].startswith("SQL error:")
        assert resources.pool.connection.call_count == 1


# ---------------------------------------------------------------------------
# Plain SQL errors: text preserved for the model to diagnose
# ---------------------------------------------------------------------------


class TestExecuteSqlSqlError:
    """A generic SQL-level error is classified `sql` with its text preserved."""

    def test_sql_error_text_is_preserved(self) -> None:
        """The original driver message survives into the tool result."""
        resources = _make_resources()
        resources.pool = _fake_pool_with_outcomes(
            [psycopg.errors.UndefinedColumn('column "nope" does not exist')],
        )

        result = asyncio.run(resources.execute_sql(_SELECT_QUERY))

        assert result["status"] == "error"
        assert result["error_kind"] == "sql"
        assert "nope" in result["error"]


# ---------------------------------------------------------------------------
# Cold-start pin: a connect failure surfaces as PoolTimeout, not raw
# ---------------------------------------------------------------------------


class _NeverConnects:
    """A psycopg-connection-shaped class whose `connect()` always fails.

    Stands in for a Neon compute that has not resumed yet: every attempt to
    *establish* a connection fails, as opposed to a connection dying after it
    was already open.
    """

    @classmethod
    async def connect(cls, conninfo: str, **kwargs: object) -> object:  # noqa: ARG003
        """Raise, simulating a compute that refuses every new connection.

        Args:
            conninfo: Ignored.
            kwargs: Ignored.

        Raises:
            ConnectionRefusedError: Unconditionally.

        """
        msg = "simulated: compute not reachable yet"
        raise ConnectionRefusedError(msg)


class TestConnectFailureSurfacesAsPoolTimeout:
    """A checkout against a pool that can never connect raises `PoolTimeout`.

    This is the behavior `blue_horizon.agents.booking.db_utils.is_transient_conn_error`
    relies on to cover the Neon wake-up case without a connect-side message
    pattern: `psycopg_pool.AsyncConnectionPool` retries a failed background
    connect internally (`reconnect_timeout` defaults to five minutes, far
    longer than this project's `pool.timeout_s`), so the checkout's own wait
    times out and raises `PoolTimeout` before the raw connect exception ever
    could. If a future `psycopg_pool` release changes that internal
    behaviour, this test fails first, at the boundary, rather than as a
    silent retry gap in production.
    """

    def test_checkout_raises_pool_timeout_not_the_raw_connect_error(self) -> None:
        """The checkout sees `PoolTimeout`, never the raw connect error."""

        async def _checkout_once() -> None:
            pool = AsyncConnectionPool(
                conninfo="postgresql://example/ro",
                min_size=0,
                max_size=1,
                timeout=0.3,
                max_idle=240.0,
                check=None,
                open=False,
                # _NeverConnects duck-types AsyncConnection's `.connect()`
                # classmethod at runtime; it isn't a real subclass, which is
                # the only thing pyright objects to here.
                connection_class=_NeverConnects,  # pyright: ignore[reportArgumentType]
            )
            await pool.open()
            try:
                async with pool.connection(timeout=0.3):
                    pass
            finally:
                await pool.close()

        with pytest.raises(PoolTimeout):
            asyncio.run(_checkout_once())


# ---------------------------------------------------------------------------
# Startup credential check: a rejected password is permanent, not transient
# ---------------------------------------------------------------------------


class TestCheckCredentials:
    """`_check_credentials` separates a rejected password from an outage.

    Through the pool both look like `PoolTimeout`, so without this a wrong
    password left the guest told the system was merely starting up.
    """

    def test_rejected_password_raises_configuration_error(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A password rejection reported by libpq becomes `ConfigurationError`."""
        rejected = psycopg.OperationalError(
            'connection to server failed: FATAL:  password authentication failed '
            'for user "bh_agent_ro"',
        )
        monkeypatch.setattr(
            psycopg.AsyncConnection, "connect", AsyncMock(side_effect=rejected),
        )
        with pytest.raises(ConfigurationError, match="PGSQL_RO_DB_URL"):
            asyncio.run(
                _check_credentials("postgresql://x", "PGSQL_RO_DB_URL", timeout_s=1.0),
            )

    def test_unreachable_database_propagates_unchanged(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Any other connect failure stays a `psycopg.OperationalError`."""
        unreachable = psycopg.OperationalError("connection timeout expired")
        monkeypatch.setattr(
            psycopg.AsyncConnection, "connect", AsyncMock(side_effect=unreachable),
        )
        with pytest.raises(psycopg.OperationalError) as exc_info:
            asyncio.run(
                _check_credentials("postgresql://x", "PGSQL_RO_DB_URL", timeout_s=1.0),
            )
        assert exc_info.value is unreachable

    def test_success_closes_the_connection(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A successful check leaves no connection open."""
        conn = MagicMock()
        conn.close = AsyncMock()
        monkeypatch.setattr(
            psycopg.AsyncConnection, "connect", AsyncMock(return_value=conn),
        )
        asyncio.run(
            _check_credentials("postgresql://x", "PGSQL_RO_DB_URL", timeout_s=1.0),
        )
        conn.close.assert_awaited_once()
