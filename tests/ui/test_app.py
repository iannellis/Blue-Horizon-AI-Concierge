"""Tests for the Streamlit UI helper functions.

Covers the pure/semi-pure functions in ui/app.py that do not require a live
Streamlit runtime: health polling, the unified chat stream (including
its `proposal` and `error` events), SSE line parsing, HTTP error translation,
and the proposal-summary renderers that back the confirmation dialog.

Not covered here: `st.dialog`-decorated flows, session-state-driven widgets
(`_render_customer_picker`, `_render_reservations`, `_render_chat`), and
button click handling -- these need a full Streamlit AppTest harness to
exercise meaningfully, which is out of scope for this module.

The entire module is skipped if streamlit is not installed in the current
environment (it lives in the optional ``ui`` dependency group).
"""

# ruff: noqa: S101

from unittest.mock import MagicMock, patch

import httpx
import pytest

streamlit = pytest.importorskip("streamlit", reason="streamlit not installed")

from ui.app import (  # noqa: E402
    _check_health,
    _handle_stream_event,
    _http_error_message,
    _parse_sse_line,
    _render_book_summary,
    _render_cancel_summary,
    _render_modify_summary,
    _stream_message,
)

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
# _parse_sse_line
# ---------------------------------------------------------------------------


class TestParseSseLine:
    """_parse_sse_line decodes 'data: ' lines and ignores everything else."""

    def test_decodes_data_line(self) -> None:
        """A well-formed data line decodes to its JSON payload."""
        assert _parse_sse_line('data: {"type": "done", "response": "ok"}') == {
            "type": "done",
            "response": "ok",
        }

    def test_non_data_line_returns_none(self) -> None:
        """A keepalive comment line (no 'data: ' prefix) returns None."""
        assert _parse_sse_line(": keepalive") is None

    def test_blank_line_returns_none(self) -> None:
        """A blank line returns None."""
        assert _parse_sse_line("") is None

    def test_malformed_json_returns_none(self) -> None:
        """A 'data: ' line with invalid JSON returns None rather than raising."""
        assert _parse_sse_line("data: not-json") is None


# ---------------------------------------------------------------------------
# _handle_stream_event
# ---------------------------------------------------------------------------


class TestHandleStreamEvent:
    """_handle_stream_event applies one SSE event to the status widget."""

    def test_stage_event_updates_status_and_continues(self) -> None:
        """A stage event updates the label and returns no result yet."""
        status = MagicMock()
        pending, result = _handle_stream_event(
            {"type": "stage", "label": "Routing…"}, status, None,
        )
        status.update.assert_called_with(label="Routing…", state="running")
        assert pending is None
        assert result is None

    def test_proposal_event_is_carried_forward(self) -> None:
        """A proposal event is captured but does not end the turn."""
        status = MagicMock()
        pending, result = _handle_stream_event(
            {
                "type": "proposal",
                "proposal_id": "p1",
                "action": "book",
                "summary": {"total": "100.00"},
            },
            status,
            None,
        )
        assert pending == {
            "proposal_id": "p1",
            "action": "book",
            "summary": {"total": "100.00"},
        }
        assert result is None

    def test_done_event_returns_response_and_pending_proposal(self) -> None:
        """A done event ends the turn, returning the response and any proposal."""
        status = MagicMock()
        proposal = {"proposal_id": "p1", "action": "book", "summary": {}}
        pending, result = _handle_stream_event(
            {"type": "done", "response": "All set."}, status, proposal,
        )
        status.update.assert_called_with(label="Done", state="complete")
        assert pending == proposal
        assert result == ("All set.", proposal)

    def test_done_event_without_proposal(self) -> None:
        """A done event with no prior proposal returns None as the proposal half."""
        status = MagicMock()
        _pending, result = _handle_stream_event(
            {"type": "done", "response": "Hi!"}, status, None,
        )
        assert result == ("Hi!", None)

    def test_error_event_ends_the_turn(self) -> None:
        """An error event marks the status widget as errored and ends the turn."""
        status = MagicMock()
        pending, result = _handle_stream_event(
            {"type": "error", "message": "Something broke."}, status, None,
        )
        status.update.assert_called_with(label="Error", state="error")
        assert pending is None
        assert result == ("Something broke.", None)


# ---------------------------------------------------------------------------
# _http_error_message
# ---------------------------------------------------------------------------


class TestHttpErrorMessage:
    """_http_error_message translates HTTP errors into guest-facing copy."""

    def test_503_returns_starting_up_message(self) -> None:
        """HTTP 503 maps to the 'still starting up' message."""
        response = MagicMock()
        response.status_code = 503
        exc = httpx.HTTPStatusError(
            "unavailable", request=MagicMock(), response=response,
        )
        assert "starting up" in _http_error_message(exc).lower()

    def test_other_status_includes_code(self) -> None:
        """Any other status code is echoed back in the message."""
        response = MagicMock()
        response.status_code = 500
        exc = httpx.HTTPStatusError("error", request=MagicMock(), response=response)
        assert "500" in _http_error_message(exc)


# ---------------------------------------------------------------------------
# _stream_message
# ---------------------------------------------------------------------------


class TestStreamMessage:
    """_stream_message connects to the unified /v1/chat endpoint via SSE."""

    def _make_stream_ctx(
        self,
        lines: list[str],
        *,
        raise_for_status: Exception | None = None,
    ) -> MagicMock:
        """Build a mock for ``httpx.stream`` used as a context manager.

        Args:
            lines: Lines returned by ``response.iter_lines()``.
            raise_for_status: If set, raised when ``raise_for_status()`` is called.

        Returns:
            MagicMock suitable for use as the return value of ``httpx.stream``.

        """
        mock_response = MagicMock()
        if raise_for_status is not None:
            mock_response.raise_for_status.side_effect = raise_for_status
        else:
            mock_response.raise_for_status.return_value = None
        mock_response.iter_lines.return_value = iter(lines)
        ctx = MagicMock()
        ctx.__enter__.return_value = mock_response
        ctx.__exit__.return_value = False
        return ctx

    def test_returns_response_on_done_event(self) -> None:
        """The done event's response field is returned as the reply."""
        ctx = self._make_stream_ctx([
            'data: {"type": "stage", "label": "Routing…"}',
            'data: {"type": "done", "response": "Hello!"}',
        ])
        with (
            patch("ui.app.httpx.stream", return_value=ctx),
            patch("ui.app.st.status", return_value=MagicMock()),
        ):
            result = _stream_message("thread-1", 7, "hi")
        assert result == ("Hello!", None)

    def test_proposal_event_is_returned_alongside_response(self) -> None:
        """A proposal captured mid-stream is returned with the done response."""
        ctx = self._make_stream_ctx([
            (
                'data: {"type": "proposal", "proposal_id": "p1", '
                '"action": "book", "summary": {"total": "100.00"}}'
            ),
            'data: {"type": "done", "response": "Review below."}',
        ])
        with (
            patch("ui.app.httpx.stream", return_value=ctx),
            patch("ui.app.st.status", return_value=MagicMock()),
        ):
            result = _stream_message("thread-1", 7, "book it")
        assert result == (
            "Review below.",
            {"proposal_id": "p1", "action": "book", "summary": {"total": "100.00"}},
        )

    def test_error_event_returns_message_with_no_proposal(self) -> None:
        """An error event returns its message and drops any pending proposal."""
        ctx = self._make_stream_ctx([
            'data: {"type": "error", "message": "Something went wrong."}',
        ])
        with (
            patch("ui.app.httpx.stream", return_value=ctx),
            patch("ui.app.st.status", return_value=MagicMock()),
        ):
            result = _stream_message("thread-1", 7, "hi")
        assert result == ("Something went wrong.", None)

    def test_request_body_includes_customer_id(self) -> None:
        """The POST body carries thread_id, customer_id, and text."""
        ctx = self._make_stream_ctx(['data: {"type": "done", "response": "ok"}'])
        with (
            patch("ui.app.httpx.stream", return_value=ctx) as mock_stream,
            patch("ui.app.st.status", return_value=MagicMock()),
        ):
            _stream_message("thread-1", 7, "hi")
        _, kwargs = mock_stream.call_args
        assert kwargs["json"] == {
            "thread_id": "thread-1",
            "customer_id": 7,
            "text": "hi",
        }
        assert kwargs["headers"] == {"Accept": "text/event-stream"}

    def test_stage_events_update_status_label(self) -> None:
        """Each stage event calls status.update() with the stage label."""
        ctx = self._make_stream_ctx([
            'data: {"type": "stage", "label": "Step 1"}',
            'data: {"type": "stage", "label": "Step 2"}',
            'data: {"type": "done", "response": "ok"}',
        ])
        mock_status = MagicMock()
        with (
            patch("ui.app.httpx.stream", return_value=ctx),
            patch("ui.app.st.status", return_value=mock_status),
        ):
            _stream_message("thread-1", 7, "hi")
        labels = [
            call.kwargs["label"]
            for call in mock_status.update.call_args_list
            if "label" in call.kwargs
        ]
        assert "Step 1" in labels
        assert "Step 2" in labels

    def test_done_event_marks_status_complete(self) -> None:
        """A done event collapses the status widget to the complete state."""
        ctx = self._make_stream_ctx([
            'data: {"type": "done", "response": "ok"}',
        ])
        mock_status = MagicMock()
        with (
            patch("ui.app.httpx.stream", return_value=ctx),
            patch("ui.app.st.status", return_value=mock_status),
        ):
            _stream_message("thread-1", 7, "hi")
        mock_status.update.assert_called_with(label="Done", state="complete")

    def test_non_data_lines_skipped(self) -> None:
        """Lines without the 'data: ' prefix are ignored."""
        ctx = self._make_stream_ctx([
            "comment line",
            "",
            'data: {"type": "done", "response": "ok"}',
        ])
        with (
            patch("ui.app.httpx.stream", return_value=ctx),
            patch("ui.app.st.status", return_value=MagicMock()),
        ):
            result = _stream_message("thread-1", 7, "hi")
        assert result == ("ok", None)

    def test_malformed_json_lines_skipped(self) -> None:
        """Lines with invalid JSON are skipped without raising."""
        ctx = self._make_stream_ctx([
            "data: not-json",
            'data: {"type": "done", "response": "ok"}',
        ])
        with (
            patch("ui.app.httpx.stream", return_value=ctx),
            patch("ui.app.st.status", return_value=MagicMock()),
        ):
            result = _stream_message("thread-1", 7, "hi")
        assert result == ("ok", None)

    def test_returns_fallback_when_stream_ends_without_done(self) -> None:
        """If the stream ends before a done event the fallback string is returned."""
        ctx = self._make_stream_ctx([
            'data: {"type": "stage", "label": "Step"}',
        ])
        with (
            patch("ui.app.httpx.stream", return_value=ctx),
            patch("ui.app.st.status", return_value=MagicMock()),
        ):
            result = _stream_message("thread-1", 7, "hi")
        assert result == ("No response received.", None)

    def test_503_returns_starting_up_message(self) -> None:
        """HTTP 503 returns the 'still starting up' message."""
        mock_response = MagicMock()
        mock_response.status_code = 503
        exc = httpx.HTTPStatusError(
            "Service Unavailable",
            request=MagicMock(),
            response=mock_response,
        )
        ctx = self._make_stream_ctx([], raise_for_status=exc)
        with (
            patch("ui.app.httpx.stream", return_value=ctx),
            patch("ui.app.st.status", return_value=MagicMock()),
        ):
            result = _stream_message("thread-1", 7, "hi")
        assert "starting up" in result[0].lower()
        assert result[1] is None

    def test_other_http_error_returns_server_error_message(self) -> None:
        """Non-503 HTTP errors return a generic message containing the status code."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        exc = httpx.HTTPStatusError(
            "Internal Server Error",
            request=MagicMock(),
            response=mock_response,
        )
        ctx = self._make_stream_ctx([], raise_for_status=exc)
        with (
            patch("ui.app.httpx.stream", return_value=ctx),
            patch("ui.app.st.status", return_value=MagicMock()),
        ):
            result = _stream_message("thread-1", 7, "hi")
        assert "500" in result[0]

    def test_timeout_returns_timeout_message(self) -> None:
        """A TimeoutException returns the timeout user message."""
        ctx = MagicMock()
        ctx.__enter__.side_effect = httpx.TimeoutException("timed out")
        ctx.__exit__.return_value = False
        with (
            patch("ui.app.httpx.stream", return_value=ctx),
            patch("ui.app.st.status", return_value=MagicMock()),
        ):
            result = _stream_message("thread-1", 7, "hi")
        assert "timed out" in result[0].lower() or "timeout" in result[0].lower()

    def test_connection_error_returns_api_unreachable_message(self) -> None:
        """Network errors return the 'could not reach the API' message."""
        ctx = MagicMock()
        ctx.__enter__.side_effect = httpx.ConnectError("refused")
        ctx.__exit__.return_value = False
        with (
            patch("ui.app.httpx.stream", return_value=ctx),
            patch("ui.app.st.status", return_value=MagicMock()),
        ):
            result = _stream_message("thread-1", 7, "hi")
        assert "api" in result[0].lower() or "reach" in result[0].lower()

    def test_error_branch_marks_status_error(self) -> None:
        """All error paths set the status widget to the error state."""
        ctx = MagicMock()
        ctx.__enter__.side_effect = httpx.ConnectError("refused")
        ctx.__exit__.return_value = False
        mock_status = MagicMock()
        with (
            patch("ui.app.httpx.stream", return_value=ctx),
            patch("ui.app.st.status", return_value=mock_status),
        ):
            _stream_message("thread-1", 7, "hi")
        assert any(
            c.kwargs.get("state") == "error"
            for c in mock_status.update.call_args_list
        )


# ---------------------------------------------------------------------------
# Proposal-dialog summary renderers
# ---------------------------------------------------------------------------


class TestSummaryRenderers:
    """The summary renderers draw strictly from `proposal["summary"]`.

    This is the load-bearing detail behind the confirmation dialog: the
    numbers shown must be the ones the application priced, never LLM text.
    """

    def test_book_summary_renders_each_room_and_total(self) -> None:
        """Book summary lists every room-stay and the headline total."""
        summary = {
            "rooms": [
                {
                    "room_number": 204,
                    "check_in": "2025-03-03",
                    "check_out": "2025-03-05",
                    "nights": 2,
                    "amount": "300.00",
                },
            ],
            "total": "300.00",
        }
        with patch("ui.app.st") as mock_st:
            _render_book_summary(summary)
        mock_st.write.assert_called_once()
        written = mock_st.write.call_args[0][0]
        assert "204" in written
        assert "300.00" in written
        mock_st.markdown.assert_called_once()
        assert "300.00" in mock_st.markdown.call_args[0][0]

    def test_cancel_summary_renders_refund_per_room(self) -> None:
        """Cancel summary lists each room's refund and the total refund."""
        summary = {
            "rooms": [{"room_number": 204, "amount": "150.00"}],
            "total": "150.00",
        }
        with patch("ui.app.st") as mock_st:
            _render_cancel_summary(summary)
        written = mock_st.write.call_args[0][0]
        assert "204" in written
        assert "150.00" in written
        assert "refund" in mock_st.markdown.call_args[0][0].lower()

    def test_modify_summary_renders_before_and_after(self) -> None:
        """Modify summary shows each room-stay's before/after side by side."""
        summary = {
            "changes": [
                {
                    "before": {
                        "room_number": 101,
                        "check_in": "2025-03-01",
                        "check_out": "2025-03-03",
                    },
                    "after": {
                        "room_number": 202,
                        "check_in": "2025-03-05",
                        "check_out": "2025-03-07",
                    },
                },
            ],
            "total": "400.00",
        }
        with patch("ui.app.st") as mock_st:
            _render_modify_summary(summary)
        written = mock_st.write.call_args[0][0]
        assert "101" in written
        assert "202" in written
        assert "400.00" in mock_st.markdown.call_args[0][0]
