"""Tests for the Streamlit UI helper functions.

_check_health and _send_message contain all the non-trivial logic in ui/app.py
and do not use the Streamlit runtime — they are fully testable with mocked httpx
calls.

The entire module is skipped if streamlit is not installed in the current
environment (it lives in the optional ``ui`` dependency group).
"""

# ruff: noqa: S101

from unittest.mock import MagicMock, patch

import httpx
import pytest

streamlit = pytest.importorskip("streamlit", reason="streamlit not installed")

from ui.app import _call_reset, _check_health, _send_message  # noqa: E402

# ---------------------------------------------------------------------------
# _check_health
# ---------------------------------------------------------------------------


class TestCheckHealth:
    """_check_health polls /v1/health and returns True only on HTTP 200."""

    def test_returns_true_on_200(self) -> None:
        """HTTP 200 response → True."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        with patch("ui.app.httpx.get", return_value=mock_response):
            assert _check_health() is True

    def test_returns_false_on_503(self) -> None:
        """HTTP 503 response → False (agent still starting up)."""
        mock_response = MagicMock()
        mock_response.status_code = 503
        with patch("ui.app.httpx.get", return_value=mock_response):
            assert _check_health() is False

    def test_returns_false_on_500(self) -> None:
        """Any non-200 status code → False."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        with patch("ui.app.httpx.get", return_value=mock_response):
            assert _check_health() is False

    def test_returns_false_on_connection_error(self) -> None:
        """Network errors (ConnectError) → False."""
        with patch(
            "ui.app.httpx.get",
            side_effect=httpx.ConnectError("refused"),
        ):
            assert _check_health() is False

    def test_returns_false_on_timeout(self) -> None:
        """Request timeout → False."""
        with patch(
            "ui.app.httpx.get",
            side_effect=httpx.TimeoutException("timeout"),
        ):
            assert _check_health() is False


# ---------------------------------------------------------------------------
# _send_message
# ---------------------------------------------------------------------------


class TestSendMessage:
    """_send_message posts to /v1/chat and extracts the AI reply."""

    def _mock_post(self, *, ai_messages: list[dict] | None = None) -> MagicMock:
        """Build a mock httpx.post that returns a successful chat response.

        Args:
            ai_messages: AI message dicts to include in the response body.

        Returns:
            MagicMock configured to return the given messages.

        """
        if ai_messages is None:
            ai_messages = [{"type": "ai", "content": "Hello!"}]
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "messages": [{"type": "human", "content": "hi"}, *ai_messages],
        }
        return mock_response

    def test_returns_last_ai_message_content(self) -> None:
        """The content of the last AI message is returned."""
        response = self._mock_post(
            ai_messages=[{"type": "ai", "content": "Room booked!"}],
        )
        with patch("ui.app.httpx.post", return_value=response):
            result = _send_message("thread-1", "book a room")
        assert result == "Room booked!"

    def test_returns_last_of_multiple_ai_messages(self) -> None:
        """When multiple AI messages are present, the last one is returned."""
        response = self._mock_post(
            ai_messages=[
                {"type": "ai", "content": "First reply"},
                {"type": "ai", "content": "Final reply"},
            ],
        )
        with patch("ui.app.httpx.post", return_value=response):
            result = _send_message("thread-1", "question")
        assert result == "Final reply"

    def test_no_ai_messages_returns_fallback(self) -> None:
        """A response with no AI messages returns the no-response fallback."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "messages": [{"type": "human", "content": "hi"}],
        }
        with patch("ui.app.httpx.post", return_value=mock_response):
            result = _send_message("thread-1", "hi")
        assert result == "No response received."

    def test_503_returns_starting_up_message(self) -> None:
        """HTTP 503 returns the 'still starting up' user message."""
        mock_response = MagicMock()
        mock_response.status_code = 503
        error = httpx.HTTPStatusError(
            "Service Unavailable",
            request=MagicMock(),
            response=mock_response,
        )
        mock_response.raise_for_status.side_effect = error
        with patch("ui.app.httpx.post", return_value=mock_response):
            result = _send_message("thread-1", "hello")
        assert "starting up" in result.lower()

    def test_other_http_error_returns_server_error_message(self) -> None:
        """Non-503 HTTP errors return a generic server-error message."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        error = httpx.HTTPStatusError(
            "Internal Server Error",
            request=MagicMock(),
            response=mock_response,
        )
        mock_response.raise_for_status.side_effect = error
        with patch("ui.app.httpx.post", return_value=mock_response):
            result = _send_message("thread-1", "hello")
        assert "500" in result

    def test_timeout_exception_returns_timeout_message(self) -> None:
        """Request timeout returns the timeout user message."""
        with patch(
            "ui.app.httpx.post",
            side_effect=httpx.TimeoutException("timed out"),
        ):
            result = _send_message("thread-1", "slow query")
        assert "timed out" in result.lower() or "timeout" in result.lower()

    def test_connection_error_returns_api_unreachable_message(self) -> None:
        """Network errors return the 'could not reach API' message."""
        with patch(
            "ui.app.httpx.post",
            side_effect=httpx.ConnectError("connection refused"),
        ):
            result = _send_message("thread-1", "hello")
        assert "api" in result.lower() or "reach" in result.lower()


# ---------------------------------------------------------------------------
# _call_reset
# ---------------------------------------------------------------------------


class TestCallReset:
    """_call_reset posts to /v1/reset and returns None on success or an error string."""

    def test_returns_none_on_200(self) -> None:
        """HTTP 200 response → None (success)."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        with patch("ui.app.httpx.post", return_value=mock_response):
            assert _call_reset() is None

    def test_returns_error_on_503(self) -> None:
        """HTTP 503 → user-facing 'not configured' message."""
        mock_response = MagicMock()
        mock_response.status_code = 503
        error = httpx.HTTPStatusError(
            "Service Unavailable",
            request=MagicMock(),
            response=mock_response,
        )
        mock_response.raise_for_status.side_effect = error
        with patch("ui.app.httpx.post", return_value=mock_response):
            result = _call_reset()
        assert result is not None
        assert "not configured" in result.lower()

    def test_returns_error_on_500(self) -> None:
        """Non-503 HTTP errors → error string containing the status code."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        error = httpx.HTTPStatusError(
            "Internal Server Error",
            request=MagicMock(),
            response=mock_response,
        )
        mock_response.raise_for_status.side_effect = error
        with patch("ui.app.httpx.post", return_value=mock_response):
            result = _call_reset()
        assert result is not None
        assert "500" in result

    def test_returns_error_on_connection_error(self) -> None:
        """Network errors → error string mentioning the API."""
        with patch(
            "ui.app.httpx.post",
            side_effect=httpx.ConnectError("connection refused"),
        ):
            result = _call_reset()
        assert result is not None
        assert "api" in result.lower() or "reach" in result.lower()
