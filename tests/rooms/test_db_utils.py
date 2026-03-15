"""Tests for rooms database utility helpers.

All tested functions are pure (no I/O).  psycopg_pool.PoolTimeout is
instantiated directly; no real database connection is required.
"""

# ruff: noqa: S101

import pytest
from psycopg_pool import PoolTimeout

from blue_horizon.agents.rooms.db_utils import (
    _is_transient_conn_error,
    _truncate_rows,
    _user_facing_db_message,
)

# ---------------------------------------------------------------------------
# _is_transient_conn_error
# ---------------------------------------------------------------------------


class TestIsTransientConnError:
    """_is_transient_conn_error correctly classifies exceptions."""

    # --- type-based matches ---

    def test_pool_timeout_is_transient(self) -> None:
        """PoolTimeout is always considered transient."""
        assert _is_transient_conn_error(PoolTimeout("pool exhausted")) is True

    def test_builtin_timeout_error_is_transient(self) -> None:
        """Python's built-in TimeoutError is transient."""
        assert _is_transient_conn_error(TimeoutError("timed out")) is True

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
        assert _is_transient_conn_error(RuntimeError(message)) is True

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
        assert _is_transient_conn_error(exc) is False

    def test_empty_message_returns_false(self) -> None:
        """An exception with no message text is not transient."""
        assert _is_transient_conn_error(RuntimeError()) is False


# ---------------------------------------------------------------------------
# _truncate_rows
# ---------------------------------------------------------------------------


class TestTruncateRows:
    """_truncate_rows returns correct (rows, truncated) pairs."""

    def _make_rows(self, n: int) -> list[dict[str, int]]:
        """Build a list of n trivial row dicts."""
        return [{"id": i} for i in range(n)]

    def test_empty_rows_not_truncated(self) -> None:
        """Empty input is returned unchanged and truncated=False."""
        rows, truncated = _truncate_rows([], max_rows=10)
        assert rows == []
        assert truncated is False

    def test_rows_under_limit_not_truncated(self) -> None:
        """Fewer rows than max_rows passes through unchanged."""
        rows = self._make_rows(3)
        result, truncated = _truncate_rows(rows, max_rows=10)
        assert result == rows
        assert truncated is False

    def test_rows_exactly_at_limit_not_truncated(self) -> None:
        """Exactly max_rows rows are not considered truncated."""
        rows = self._make_rows(5)
        result, truncated = _truncate_rows(rows, max_rows=5)
        assert result == rows
        assert truncated is False

    def test_rows_over_limit_truncated(self) -> None:
        """More rows than max_rows are sliced and truncated=True."""
        rows = self._make_rows(7)
        result, truncated = _truncate_rows(rows, max_rows=4)
        assert result == rows[:4]
        assert truncated is True

    def test_truncation_preserves_order(self) -> None:
        """The first max_rows rows are kept (order is preserved)."""
        rows = [{"id": i} for i in [10, 20, 30, 40, 50]]
        result, _ = _truncate_rows(rows, max_rows=3)
        assert [r["id"] for r in result] == [10, 20, 30]

    def test_max_rows_one(self) -> None:
        """max_rows=1 keeps only the first row."""
        rows = self._make_rows(5)
        result, truncated = _truncate_rows(rows, max_rows=1)
        assert len(result) == 1
        assert truncated is True


# ---------------------------------------------------------------------------
# _user_facing_db_message
# ---------------------------------------------------------------------------


class TestUserFacingDbMessage:
    """_user_facing_db_message returns a non-empty, user-safe string."""

    def test_returns_string(self) -> None:
        """Return value is a non-empty string."""
        msg = _user_facing_db_message()
        assert isinstance(msg, str)
        assert len(msg) > 0

    def test_message_mentions_booking_system(self) -> None:
        """Message references the booking system so users understand context."""
        msg = _user_facing_db_message()
        assert "booking" in msg.lower()

    def test_message_is_deterministic(self) -> None:
        """Successive calls return the same string."""
        assert _user_facing_db_message() == _user_facing_db_message()
