"""Tests for booking database utility helpers.

No real database connection is required. psycopg_pool.PoolTimeout is
instantiated directly, and database metadata tests use async mocks.
"""

# ruff: noqa: S101

import asyncio
import datetime as dt
from unittest.mock import AsyncMock, MagicMock, patch

import psycopg
import pytest
from psycopg_pool import PoolTimeout

from blue_horizon.agents.booking.db_utils import (
    _tool_error_message_for_model,
    check_credentials,
    fetch_rooms_metadata,
    is_transient_conn_error,
)
from blue_horizon.agents.exceptions import ConfigurationError, OperationalError

# ---------------------------------------------------------------------------
# fetch_rooms_metadata
# ---------------------------------------------------------------------------


class TestFetchRoomsMetadata:
    """fetch_rooms_metadata builds metadata from PostgreSQL result rows."""

    @staticmethod
    def _connect_returning(
        *,
        amenity_rows: list[tuple[str | None]],
        floor_row: tuple[int | None, int | None],
        date_row: tuple[dt.date | None, dt.date | None],
    ) -> AsyncMock:
        """Build a patched `AsyncConnection.connect` over canned results.

        Args:
            amenity_rows: Rows for the combined amenities query.
            floor_row: Row for the floor-range query.
            date_row: Row for the availability-window query.

        Returns:
            An async mock whose connection's cursor replays the results in
            `fetch_rooms_metadata`'s query order.

        """
        cursor = MagicMock()
        cursor.__aenter__ = AsyncMock(return_value=cursor)
        cursor.__aexit__ = AsyncMock(return_value=False)
        cursor.execute = AsyncMock()
        cursor.fetchall = AsyncMock(
            side_effect=[
                [("available",), ("booked",)],
                [("King, Split",), ("Queen",)],
                [("clean",), ("dirty",)],
                [("suite",), ("standard",)],
                amenity_rows,
                [("Ocean View",), ("City View",)],
            ],
        )
        cursor.fetchone = AsyncMock(side_effect=[floor_row, date_row])

        conn = MagicMock()
        conn.__aenter__ = AsyncMock(return_value=conn)
        conn.__aexit__ = AsyncMock(return_value=False)
        conn.cursor.return_value = cursor
        return AsyncMock(return_value=conn)

    def test_builds_rooms_metadata(self) -> None:
        """Enum labels keep commas, vocabularies are sorted, NULLs dropped."""
        connect = self._connect_returning(
            amenity_rows=[("wifi",), (None,), ("balcony",)],
            floor_row=(1, 20),
            date_row=(dt.date(2025, 1, 4), dt.date(2026, 1, 3)),
        )
        with patch(
            "blue_horizon.agents.booking.db_utils.psycopg.AsyncConnection.connect",
            new=connect,
        ):
            meta = asyncio.run(fetch_rooms_metadata("postgresql://example"))

        assert meta.enum_values["room_bed_type"] == ["King, Split", "Queen"]
        assert meta.amenities == ("balcony", "wifi")
        assert meta.view_types == ("City View", "Ocean View")
        assert (meta.min_floor, meta.max_floor) == (1, 20)
        assert meta.first_night == dt.date(2025, 1, 4)
        assert meta.last_check_out == dt.date(2026, 1, 4)

    def test_empty_availability_raises(self) -> None:
        """An empty availability table cannot define a search window."""
        connect = self._connect_returning(
            amenity_rows=[("wifi",)],
            floor_row=(1, 20),
            date_row=(None, None),
        )
        with (
            patch(
                "blue_horizon.agents.booking.db_utils.psycopg.AsyncConnection.connect",
                new=connect,
            ),
            pytest.raises(OperationalError),
        ):
            asyncio.run(fetch_rooms_metadata("postgresql://example"))


# ---------------------------------------------------------------------------
# is_transient_conn_error
# ---------------------------------------------------------------------------


class TestIsTransientConnError:
    """is_transient_conn_error correctly classifies exceptions."""

    # --- type-based matches ---

    def test_pool_timeout_is_transient(self) -> None:
        """PoolTimeout is always considered transient."""
        assert is_transient_conn_error(PoolTimeout("pool exhausted")) is True

    def test_builtin_timeout_error_is_transient(self) -> None:
        """Python's built-in TimeoutError is transient."""
        assert is_transient_conn_error(TimeoutError("timed out")) is True

    # --- message-based matches ---

    @pytest.mark.parametrize(
        "message",
        [
            "SSL connection has been closed unexpectedly",
            "ssl connection has been closed unexpectedly",  # lowercase variant
            "server closed the connection unexpectedly",
            "connection is closed",
            "connection not open",
            "terminating connection due to administrator command",
        ],
    )
    def test_known_transient_message_is_transient(self, message: str) -> None:
        """Standard transient-error substrings are detected case-insensitively."""
        assert is_transient_conn_error(RuntimeError(message)) is True

    # --- non-transient ---

    @pytest.mark.parametrize(
        "exc",
        [
            ValueError("bad query"),
            KeyError("missing key"),
            RuntimeError("some unrelated runtime error"),
            TypeError("wrong type"),
            Exception("generic exception"),
        ],
    )
    def test_non_transient_exception_returns_false(self, exc: BaseException) -> None:
        """Unrelated exceptions are not flagged as transient."""
        assert is_transient_conn_error(exc) is False

    def test_empty_message_returns_false(self) -> None:
        """An exception with no message text is not transient."""
        assert is_transient_conn_error(RuntimeError()) is False


# ---------------------------------------------------------------------------
# _tool_error_message_for_model
# ---------------------------------------------------------------------------


class TestToolErrorMessageForModel:
    """_tool_error_message_for_model returns a stable, marked instruction.

    This string is a `search_rooms` tool result the model consumes, not guest-
    facing copy -- see `booking.txt`'s ``DATABASE_UNAVAILABLE`` retry rule,
    which keys off the exact marker asserted here.
    """

    def test_returns_string(self) -> None:
        """Return value is a non-empty string."""
        msg = _tool_error_message_for_model()
        assert isinstance(msg, str)
        assert len(msg) > 0

    def test_starts_with_stable_marker(self) -> None:
        """The leading marker is exact, since the prompt's retry rule keys off it."""
        msg = _tool_error_message_for_model()
        assert msg.startswith("DATABASE_UNAVAILABLE:")

    def test_instructs_against_retrying(self) -> None:
        """The text tells the model not to retry, unlike rejected arguments."""
        msg = _tool_error_message_for_model()
        assert "retry" in msg.lower()

    def test_message_is_deterministic(self) -> None:
        """Successive calls return the same string."""
        assert _tool_error_message_for_model() == _tool_error_message_for_model()


# ---------------------------------------------------------------------------
# Startup credential check: a rejected password is permanent, not transient
# ---------------------------------------------------------------------------


class TestCheckCredentials:
    """`check_credentials` separates a rejected password from an outage.

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
        with pytest.raises(ConfigurationError, match="read-only database URL"):
            asyncio.run(
                check_credentials("postgresql://x", "read-only", timeout_s=1.0),
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
                check_credentials("postgresql://x", "read-only", timeout_s=1.0),
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
            check_credentials("postgresql://x", "read-only", timeout_s=1.0),
        )
        conn.close.assert_awaited_once()
