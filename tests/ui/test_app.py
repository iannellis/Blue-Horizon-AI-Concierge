"""Tests for the Streamlit UI helper functions.

Covers the pure/semi-pure functions in ui/app.py that do not require a live
Streamlit runtime: health polling, the unified chat stream (including
its `proposal` and `error` events), SSE line parsing, HTTP error translation,
the proposal-summary renderers that back the confirmation dialog, and
`_fetch_bookings`'s None/empty-list distinction, and Axiom log shipping, with
the OTLP exporter replaced by the SDK's in-memory one so nothing is sent.

Not covered here: `st.dialog`-decorated flows, session-state-driven widgets
(`_render_customer_picker`, `_render_reservations`, `_render_chat`,
`_render_guest_assignment` itself), and button click handling -- these need
a full Streamlit AppTest harness to exercise meaningfully, which is out of
scope for this module. The guest claim registry it calls into
(`_try_claim_guest`, `_touch_claim`, `_release_claim`) is plain module
state guarded by a lock, with no widget dependency, so it is covered here
directly.

The entire module is skipped if streamlit is not installed in the current
environment (it lives in the optional ``ui`` dependency group).
"""

# ruff: noqa: S101

import logging
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock, patch

import httpx2
import pytest
from opentelemetry.instrumentation.logging.handler import LoggingHandler
from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter

from blue_horizon.config import AxiomLoggingConfig

if TYPE_CHECKING:
    from opentelemetry.sdk._logs import LoggerProvider

streamlit = pytest.importorskip("streamlit", reason="streamlit not installed")

from ui.app import (  # noqa: E402
    _GUEST_CLAIM_TTL_S,
    ChatTurnResult,
    _axiom_config,
    _axiom_exporter,
    _check_health,
    _configure_logging,
    _confirm_proposal,
    _fetch_bookings,
    _guest_claims,
    _GuestClaim,
    _handle_stream_event,
    _http_error_message,
    _int_or_none,
    _log_api_failure,
    _not_from_export_thread,
    _parse_sse_line,
    _persist_guest_identity,
    _release_claim,
    _render_book_summary,
    _render_cancel_summary,
    _render_modify_summary,
    _shipping_handler,
    _stream_message,
    _touch_claim,
    _try_claim_guest,
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
        with patch("ui.app.httpx2.get", return_value=mock_response):
            assert _check_health() is True

    def test_returns_false_on_503(self) -> None:
        """HTTP 503 response → False (agent still starting up)."""
        mock_response = MagicMock()
        mock_response.status_code = 503
        with patch("ui.app.httpx2.get", return_value=mock_response):
            assert _check_health() is False

    def test_returns_false_on_500(self) -> None:
        """Any non-200 status code → False."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        with patch("ui.app.httpx2.get", return_value=mock_response):
            assert _check_health() is False

    def test_returns_false_on_connection_error(self) -> None:
        """Network errors (ConnectError) → False."""
        with patch(
            "ui.app.httpx2.get",
            side_effect=httpx2.ConnectError("refused"),
        ):
            assert _check_health() is False

    def test_returns_false_on_timeout(self) -> None:
        """Request timeout → False."""
        with patch(
            "ui.app.httpx2.get",
            side_effect=httpx2.TimeoutException("timeout"),
        ):
            assert _check_health() is False


# ---------------------------------------------------------------------------
# API failure logging
# ---------------------------------------------------------------------------


class TestApiFailureLogging:
    """An unreachable API logs one line; an unexpected error keeps its traceback."""

    def test_connect_error_logs_one_line(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A refused connection logs a warning naming the error, with no traceback."""
        with (
            patch("ui.app.httpx2.get", side_effect=httpx2.ConnectError("refused")),
            caplog.at_level(logging.WARNING, logger="ui.app"),
        ):
            assert _fetch_bookings(13) is None
        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert record.exc_info is None
        assert "customer_id=13" in record.getMessage()
        assert "refused" in record.getMessage()

    def test_malformed_body_keeps_traceback(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A response that is not the expected JSON is a defect, logged in full."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.side_effect = ValueError("not JSON")
        with (
            patch("ui.app.httpx2.get", return_value=mock_response),
            caplog.at_level(logging.WARNING, logger="ui.app"),
        ):
            assert _fetch_bookings(13) is None
        assert len(caplog.records) == 1
        assert caplog.records[0].exc_info is not None

    def test_context_is_named_and_attached(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The thread and guest appear in the text and as the API's field names."""
        with caplog.at_level(logging.WARNING, logger="ui.app"):
            _log_api_failure(
                httpx2.ReadTimeout("slow"),
                "Chat request timed out",
                thread_id="t-1",
                customer_id=13,
            )
        (record,) = caplog.records
        assert record.getMessage() == (
            "Chat request timed out thread_id=t-1 customer_id=13: ReadTimeout('slow')"
        )
        assert record.__dict__["thread_id"] == "t-1"
        assert record.__dict__["customer_id"] == 13  # noqa: PLR2004

    def test_unknown_context_is_left_off(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A call made for no guest carries neither field."""
        with caplog.at_level(logging.WARNING, logger="ui.app"):
            _log_api_failure(
                httpx2.ConnectError("refused"), "Could not fetch the guest list",
            )
        (record,) = caplog.records
        assert record.getMessage() == (
            "Could not fetch the guest list: ConnectError('refused')"
        )
        assert "thread_id" not in record.__dict__
        assert "customer_id" not in record.__dict__


# ---------------------------------------------------------------------------
# _fetch_bookings
# ---------------------------------------------------------------------------


class TestFetchBookings:
    """_fetch_bookings distinguishes "could not load" from "genuinely none".

    `None` and `[]` are deliberately different return values -- collapsing
    them would render a backend outage in `_render_reservations` as "No
    reservations yet.", indistinguishable from a guest with no bookings.
    """

    def test_returns_list_on_success(self) -> None:
        """A successful response returns its bookings list, even if empty."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"bookings": [{"booking_id": 1}]}
        with patch("ui.app.httpx2.get", return_value=mock_response):
            assert _fetch_bookings(7) == [{"booking_id": 1}]

    def test_returns_empty_list_when_guest_has_no_bookings(self) -> None:
        """A successful response with no bookings returns `[]`, not `None`."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"bookings": []}
        with patch("ui.app.httpx2.get", return_value=mock_response):
            assert _fetch_bookings(7) == []

    def test_returns_none_on_connection_error(self) -> None:
        """A network failure returns `None`, distinct from an empty list."""
        with patch(
            "ui.app.httpx2.get", side_effect=httpx2.ConnectError("refused"),
        ):
            assert _fetch_bookings(7) is None

    def test_returns_none_on_http_error(self) -> None:
        """A non-2xx response returns `None`, distinct from an empty list."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx2.HTTPStatusError(
            "error", request=MagicMock(), response=mock_response,
        )
        with patch("ui.app.httpx2.get", return_value=mock_response):
            assert _fetch_bookings(7) is None


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

    def test_malformed_proposal_event_is_ignored(self) -> None:
        """A proposal event missing a required key is dropped, not indexed raw.

        A raw `event["proposal_id"]` index would raise `KeyError`, which the
        generic exception handler in `_stream_message` would otherwise report
        as an unhelpful "Could not reach the API".
        """
        status = MagicMock()
        pending, result = _handle_stream_event(
            {"type": "proposal", "action": "book", "summary": {}},
            status,
            None,
        )
        assert pending is None
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
        assert result == ChatTurnResult(ok=True, text="All set.", proposal=proposal)

    def test_done_event_without_proposal(self) -> None:
        """A done event with no prior proposal returns None as the proposal half."""
        status = MagicMock()
        _pending, result = _handle_stream_event(
            {"type": "done", "response": "Hi!"}, status, None,
        )
        assert result == ChatTurnResult(ok=True, text="Hi!", proposal=None)

    def test_error_event_ends_the_turn(self) -> None:
        """An error event marks the status widget as errored and ends the turn."""
        status = MagicMock()
        pending, result = _handle_stream_event(
            {"type": "error", "message": "Something broke."}, status, None,
        )
        status.update.assert_called_with(label="Error", state="error")
        assert pending is None
        assert result == ChatTurnResult(
            ok=False, text="Something broke.", proposal=None,
        )


# ---------------------------------------------------------------------------
# _http_error_message
# ---------------------------------------------------------------------------


class TestHttpErrorMessage:
    """_http_error_message translates HTTP errors into guest-facing copy."""

    def test_503_returns_starting_up_message(self) -> None:
        """HTTP 503 maps to the 'still starting up' message."""
        response = MagicMock()
        response.status_code = 503
        exc = httpx2.HTTPStatusError(
            "unavailable", request=MagicMock(), response=response,
        )
        assert "starting up" in _http_error_message(exc).lower()

    def test_other_status_includes_code(self) -> None:
        """Any other status code is echoed back in the message."""
        response = MagicMock()
        response.status_code = 500
        exc = httpx2.HTTPStatusError("error", request=MagicMock(), response=response)
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
        """Build a mock for ``httpx2.stream`` used as a context manager.

        Args:
            lines: Lines returned by ``response.iter_lines()``.
            raise_for_status: If set, raised when ``raise_for_status()`` is called.

        Returns:
            MagicMock suitable for use as the return value of ``httpx2.stream``.

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
            patch("ui.app.httpx2.stream", return_value=ctx),
            patch("ui.app.st.status", return_value=MagicMock()),
        ):
            result = _stream_message("thread-1", 7, "hi")
        assert result == ChatTurnResult(ok=True, text="Hello!", proposal=None)

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
            patch("ui.app.httpx2.stream", return_value=ctx),
            patch("ui.app.st.status", return_value=MagicMock()),
        ):
            result = _stream_message("thread-1", 7, "book it")
        assert result == ChatTurnResult(
            ok=True,
            text="Review below.",
            proposal={
                "proposal_id": "p1", "action": "book", "summary": {"total": "100.00"},
            },
        )

    def test_error_event_returns_message_with_no_proposal(self) -> None:
        """An error event returns its message and drops any pending proposal."""
        ctx = self._make_stream_ctx([
            'data: {"type": "error", "message": "Something went wrong."}',
        ])
        with (
            patch("ui.app.httpx2.stream", return_value=ctx),
            patch("ui.app.st.status", return_value=MagicMock()),
        ):
            result = _stream_message("thread-1", 7, "hi")
        assert result == ChatTurnResult(
            ok=False, text="Something went wrong.", proposal=None,
        )
        assert result.ok is False

    def test_request_body_includes_customer_id(self) -> None:
        """The POST body carries thread_id, customer_id, and text."""
        ctx = self._make_stream_ctx(['data: {"type": "done", "response": "ok"}'])
        with (
            patch("ui.app.httpx2.stream", return_value=ctx) as mock_stream,
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
            patch("ui.app.httpx2.stream", return_value=ctx),
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
            patch("ui.app.httpx2.stream", return_value=ctx),
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
            patch("ui.app.httpx2.stream", return_value=ctx),
            patch("ui.app.st.status", return_value=MagicMock()),
        ):
            result = _stream_message("thread-1", 7, "hi")
        assert result == ChatTurnResult(ok=True, text="ok", proposal=None)

    def test_malformed_json_lines_skipped(self) -> None:
        """Lines with invalid JSON are skipped without raising."""
        ctx = self._make_stream_ctx([
            "data: not-json",
            'data: {"type": "done", "response": "ok"}',
        ])
        with (
            patch("ui.app.httpx2.stream", return_value=ctx),
            patch("ui.app.st.status", return_value=MagicMock()),
        ):
            result = _stream_message("thread-1", 7, "hi")
        assert result == ChatTurnResult(ok=True, text="ok", proposal=None)

    def test_returns_fallback_when_stream_ends_without_done(self) -> None:
        """If the stream ends before a done event the fallback string is returned."""
        ctx = self._make_stream_ctx([
            'data: {"type": "stage", "label": "Step"}',
        ])
        with (
            patch("ui.app.httpx2.stream", return_value=ctx),
            patch("ui.app.st.status", return_value=MagicMock()),
        ):
            result = _stream_message("thread-1", 7, "hi")
        assert result == ChatTurnResult(
            ok=True, text="No response received.", proposal=None,
        )

    def test_503_returns_starting_up_message(self) -> None:
        """HTTP 503 returns the 'still starting up' message, and is not ok.

        Not `ok` is what tells `_submit_chat_message` to keep the guest's own
        message standing alone in history rather than appending a fabricated
        assistant reply -- see `_render_pending_resend`.
        """
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_response.json.side_effect = ValueError("not JSON")
        exc = httpx2.HTTPStatusError(
            "Service Unavailable",
            request=MagicMock(),
            response=mock_response,
        )
        ctx = self._make_stream_ctx([], raise_for_status=exc)
        with (
            patch("ui.app.httpx2.stream", return_value=ctx),
            patch("ui.app.st.status", return_value=MagicMock()),
        ):
            result = _stream_message("thread-1", 7, "hi")
        assert result.ok is False
        assert "starting up" in result.text.lower()
        assert result.proposal is None

    def test_other_http_error_returns_server_error_message(self) -> None:
        """Non-503 HTTP errors return a generic message containing the status code."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        exc = httpx2.HTTPStatusError(
            "Internal Server Error",
            request=MagicMock(),
            response=mock_response,
        )
        ctx = self._make_stream_ctx([], raise_for_status=exc)
        with (
            patch("ui.app.httpx2.stream", return_value=ctx),
            patch("ui.app.st.status", return_value=MagicMock()),
        ):
            result = _stream_message("thread-1", 7, "hi")
        assert result.ok is False
        assert "500" in result.text

    def test_timeout_returns_timeout_message(self) -> None:
        """A TimeoutException returns the timeout copy, with no retry instruction.

        The copy renders above the "Send again" button, which is the retry
        affordance, so the text itself must not tell the guest to try again.
        """
        ctx = MagicMock()
        ctx.__enter__.side_effect = httpx2.TimeoutException("timed out")
        ctx.__exit__.return_value = False
        with (
            patch("ui.app.httpx2.stream", return_value=ctx),
            patch("ui.app.st.status", return_value=MagicMock()),
        ):
            result = _stream_message("thread-1", 7, "hi")
        assert result.ok is False
        assert "too long" in result.text.lower()
        assert "try again" not in result.text.lower()

    def test_connection_error_returns_api_unreachable_message(self) -> None:
        """Network errors return fixed copy, never the raw exception text."""
        ctx = MagicMock()
        ctx.__enter__.side_effect = httpx2.ConnectError("some raw socket detail")
        ctx.__exit__.return_value = False
        with (
            patch("ui.app.httpx2.stream", return_value=ctx),
            patch("ui.app.st.status", return_value=MagicMock()),
        ):
            result = _stream_message("thread-1", 7, "hi")
        assert result.ok is False
        assert "reach" in result.text.lower()
        assert "some raw socket detail" not in result.text

    def test_error_branch_marks_status_error(self) -> None:
        """All error paths set the status widget to the error state."""
        ctx = MagicMock()
        ctx.__enter__.side_effect = httpx2.ConnectError("refused")
        ctx.__exit__.return_value = False
        mock_status = MagicMock()
        with (
            patch("ui.app.httpx2.stream", return_value=ctx),
            patch("ui.app.st.status", return_value=mock_status),
        ):
            _stream_message("thread-1", 7, "hi")
        assert any(
            c.kwargs.get("state") == "error"
            for c in mock_status.update.call_args_list
        )


# ---------------------------------------------------------------------------
# _confirm_proposal
# ---------------------------------------------------------------------------


def _mock_response(
    status_code: int, body: dict[str, object] | None = None,
) -> MagicMock:
    """Build a mock `httpx2.Response` for `_confirm_proposal` tests.

    Args:
        status_code: HTTP status code the mock should report.
        body: JSON body the mock's `.json()` should return. `None` produces
            a response whose `.json()` raises, matching a body that is not
            JSON at all.

    Returns:
        A `MagicMock` standing in for an `httpx2.Response`.

    """
    response = MagicMock()
    response.status_code = status_code
    if body is None:
        response.json.side_effect = ValueError("not JSON")
    else:
        response.json.return_value = body
    return response


class TestConfirmProposal:
    """`_confirm_proposal` maps each HTTP outcome to a `ConfirmOutcome`.

    Per the plan's Phase 1 table: `confirmed`/`refused`/`expired`/
    `wrong_guest` are terminal (`keep_pending=False`); `unavailable` and
    `unreachable` are not (`keep_pending=True`), since in both cases the
    write was never evaluated and a retry is safe.
    """

    def test_200_is_confirmed_and_not_kept_pending(self) -> None:
        """A 200 response reports `confirmed` with the server's own message."""
        response = _mock_response(200, {"status": "confirmed", "message": "Booked!"})
        with patch("ui.app.httpx2.post", return_value=response):
            outcome = _confirm_proposal("prop-1", 7)
        assert outcome.status == "confirmed"
        assert outcome.message == "Booked!"
        assert outcome.keep_pending is False

    def test_409_is_refused_and_not_kept_pending(self) -> None:
        """A 409 response reports `refused` with the write's own refusal detail."""
        response = _mock_response(
            409, {"detail": "Room 204 is not available for every night requested."},
        )
        with patch("ui.app.httpx2.post", return_value=response):
            outcome = _confirm_proposal("prop-1", 7)
        assert outcome.status == "refused"
        assert outcome.message == "Room 204 is not available for every night requested."
        assert outcome.keep_pending is False

    def test_404_is_expired_and_not_kept_pending(self) -> None:
        """A 404 response reports `expired`."""
        response = _mock_response(404, {"detail": "That request has expired."})
        with patch("ui.app.httpx2.post", return_value=response):
            outcome = _confirm_proposal("prop-1", 7)
        assert outcome.status == "expired"
        assert outcome.keep_pending is False

    def test_403_is_wrong_guest_and_not_kept_pending(self) -> None:
        """A 403 response reports `wrong_guest`."""
        response = _mock_response(403, {"detail": "Not your request."})
        with patch("ui.app.httpx2.post", return_value=response):
            outcome = _confirm_proposal("prop-1", 7)
        assert outcome.status == "wrong_guest"
        assert outcome.keep_pending is False

    def test_503_is_unavailable_and_kept_pending(self) -> None:
        """A 503 response reports `unavailable` and keeps the dialog open."""
        response = _mock_response(
            503, {"detail": "Could not reach the database to complete this request."},
        )
        with patch("ui.app.httpx2.post", return_value=response):
            outcome = _confirm_proposal("prop-1", 7)
        assert outcome.status == "unavailable"
        assert outcome.keep_pending is True

    def test_503_with_no_detail_falls_back_to_fixed_copy(self) -> None:
        """A 503 with a non-JSON body still yields safe, non-raw copy."""
        response = _mock_response(503, body=None)
        with patch("ui.app.httpx2.post", return_value=response):
            outcome = _confirm_proposal("prop-1", 7)
        assert outcome.status == "unavailable"
        assert outcome.keep_pending is True
        assert "still pending" in outcome.message

    def test_network_failure_is_unreachable_and_kept_pending(self) -> None:
        """A connection error reports `unreachable` and keeps the dialog open.

        No raw exception text leaks into the message -- see the fixed copy
        used here, unlike the pre-fix `f"Could not reach the API: {exc}"`.
        """
        with patch(
            "ui.app.httpx2.post", side_effect=httpx2.ConnectError("refused"),
        ):
            outcome = _confirm_proposal("prop-1", 7)
        assert outcome.status == "unreachable"
        assert outcome.keep_pending is True
        assert "refused" not in outcome.message

    def test_unexpected_status_code_falls_back_to_unreachable(self) -> None:
        """An unmapped status code is treated as unknown, not as a refusal.

        The safest default when the outcome is genuinely unknown is that a
        retry is still safe, never that the write already happened.
        """
        response = _mock_response(500, {"detail": "Internal Server Error"})
        with patch("ui.app.httpx2.post", return_value=response):
            outcome = _confirm_proposal("prop-1", 7)
        assert outcome.status == "unreachable"
        assert outcome.keep_pending is True


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


# ---------------------------------------------------------------------------
# _int_or_none
# ---------------------------------------------------------------------------


class TestIntOrNone:
    """_int_or_none parses a query-param string, tolerating garbage."""

    def test_parses_valid_int(self) -> None:
        """A numeric string parses to its int value."""
        assert _int_or_none("7") == 7  # noqa: PLR2004

    def test_none_input_returns_none(self) -> None:
        """A missing query param (None) returns None."""
        assert _int_or_none(None) is None

    def test_non_numeric_string_returns_none(self) -> None:
        """A malformed value (e.g. a hand-edited URL) returns None, not a raise."""
        assert _int_or_none("not-a-number") is None


# ---------------------------------------------------------------------------
# Guest claim registry
# ---------------------------------------------------------------------------


class TestGuestClaimRegistry:
    """_try_claim_guest / _touch_claim / _release_claim keep guests exclusive."""

    def setup_method(self) -> None:
        """Start each test with an empty claim registry."""
        _guest_claims.clear()

    def teardown_method(self) -> None:
        """Leave no claims behind for the next test."""
        _guest_claims.clear()

    def test_claims_an_available_guest(self) -> None:
        """An unclaimed candidate is assigned to the requesting session."""
        customer_id = _try_claim_guest("session-a", [1, 2, 3])
        assert customer_id in {1, 2, 3}
        assert _guest_claims[customer_id].session_id == "session-a"

    def test_returns_none_when_all_claimed(self) -> None:
        """No candidate is left unclaimed → None, not a collision."""
        _try_claim_guest("session-a", [1])
        assert _try_claim_guest("session-b", [1]) is None

    def test_expired_claim_is_swept_and_reclaimable(self) -> None:
        """A claim idle past the TTL is dropped and its guest reassignable."""
        _guest_claims[1] = _GuestClaim(
            session_id="session-a",
            last_seen=datetime.now(UTC) - timedelta(seconds=_GUEST_CLAIM_TTL_S + 1),
        )
        customer_id = _try_claim_guest("session-b", [1])
        assert customer_id == 1
        assert _guest_claims[1].session_id == "session-b"

    def test_touch_refreshes_own_claim(self) -> None:
        """Touching a claim this session holds refreshes it and returns its id."""
        _try_claim_guest("session-a", [1])
        before = _guest_claims[1].last_seen
        result = _touch_claim(1, "session-a")
        assert result == 1
        assert _guest_claims[1].last_seen >= before

    def test_touch_recreates_missing_claim(self) -> None:
        """A claim missing entirely (e.g. after a process restart) is recreated."""
        result = _touch_claim(1, "session-a")
        assert result == 1
        assert _guest_claims[1].session_id == "session-a"

    def test_touch_refuses_to_steal_another_sessions_claim(self) -> None:
        """A claim now held by a different session is left alone, not stolen.

        This is what makes it safe to restore `customer_id` from the URL
        after a refresh: if the guest was reassigned while this session was
        away, touching it must not silently take it back.
        """
        _try_claim_guest("session-a", [1])
        result = _touch_claim(1, "session-b")
        assert result is None
        assert _guest_claims[1].session_id == "session-a"

    def test_release_drops_the_claim(self) -> None:
        """Releasing a held claim removes it from the registry."""
        _try_claim_guest("session-a", [1])
        _release_claim(1)
        assert 1 not in _guest_claims

    def test_release_none_is_a_no_op(self) -> None:
        """Releasing `None` (no claim ever held) does not raise."""
        _release_claim(None)


# ---------------------------------------------------------------------------
# _persist_guest_identity
# ---------------------------------------------------------------------------


class TestPersistGuestIdentity:
    """_persist_guest_identity writes sid/cid into the URL query params."""

    def test_writes_session_and_customer_ids(self) -> None:
        """Both ids land in `st.query_params`, customer_id as a string."""
        with patch("ui.app.st") as mock_st:
            mock_st.session_state.session_id = "session-a"
            _persist_guest_identity(7)
        mock_st.query_params.__setitem__.assert_any_call("sid", "session-a")
        mock_st.query_params.__setitem__.assert_any_call("cid", "7")


# ---------------------------------------------------------------------------
# Log shipping
# ---------------------------------------------------------------------------


def _shipping_handlers() -> list[LoggingHandler]:
    """List the root logger's shipping handlers.

    Returns:
        list[LoggingHandler]: Every OpenTelemetry handler on the root logger.

    """
    return [h for h in logging.getLogger().handlers if isinstance(h, LoggingHandler)]


def _shut_down(handler: LoggingHandler) -> None:
    """Send what `handler` has queued and stop its batch processor thread.

    Args:
        handler: A handler built by `_shipping_handler`.

    """
    cast("LoggerProvider", handler._logger_provider).shutdown()  # noqa: SLF001


@pytest.fixture
def restore_root_logger() -> Iterator[None]:
    """Restore the root logger's handlers and level after a test.

    Any shipping handler the test added is shut down first, so no batch
    processor thread outlives the test.

    Yields:
        None, while the test runs.

    """
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    for handler in _shipping_handlers():
        _shut_down(handler)
    root.handlers[:] = handlers
    root.setLevel(level)


class TestLogShipping:
    """The UI ships its records to Axiom only when both variables are set."""

    def test_record_ships_as_the_ui_service_with_its_context(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A shipped record keeps its message, level, guest, and thread."""
        exporter = InMemoryLogRecordExporter()
        handler = _shipping_handler(_axiom_config(), exporter)
        ui_logger = logging.getLogger("ui.app")
        monkeypatch.setattr(ui_logger, "handlers", [handler])
        monkeypatch.setattr(ui_logger, "propagate", False)
        _log_api_failure(
            httpx2.ConnectError("refused"),
            "Chat request failed",
            thread_id="t-1",
            customer_id=13,
        )
        _shut_down(handler)

        (record,) = exporter.get_finished_logs()
        attributes = record.log_record.attributes or {}
        assert str(record.log_record.body).startswith(
            "Chat request failed thread_id=t-1 customer_id=13:",
        )
        # OpenTelemetry's name for Python's WARNING level.
        assert record.log_record.severity_text == "WARN"
        assert attributes["thread_id"] == "t-1"
        assert attributes["customer_id"] == 13  # noqa: PLR2004
        assert record.resource.attributes["service.name"] == "blue-horizon-ui"

    @pytest.mark.usefixtures("restore_root_logger")
    def test_reruns_add_one_shipping_handler(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Configuring on every Streamlit rerun still leaves one shipping handler."""
        monkeypatch.setenv("AXIOM_API_KEY", "key")
        monkeypatch.setenv("AXIOM_DATASET", "dataset")
        monkeypatch.setattr(
            "ui.app._axiom_exporter", lambda *_args: InMemoryLogRecordExporter(),
        )
        _configure_logging()
        _configure_logging()
        assert len(_shipping_handlers()) == 1

    @pytest.mark.usefixtures("restore_root_logger")
    @pytest.mark.parametrize(("api_key", "dataset"), [("key", ""), ("", "dataset")])
    def test_ships_nothing_without_both_variables(
        self, monkeypatch: pytest.MonkeyPatch, api_key: str, dataset: str,
    ) -> None:
        """A key without a dataset, or a dataset without a key, ships nothing."""
        monkeypatch.setenv("AXIOM_API_KEY", api_key)
        monkeypatch.setenv("AXIOM_DATASET", dataset)
        _configure_logging()
        assert not _shipping_handlers()

    def test_reads_the_same_settings_the_api_types(self) -> None:
        """The file's keys are exactly the API's, so a rename cannot go unnoticed."""
        config = _axiom_config()
        assert set(config) == set(AxiomLoggingConfig.model_fields)
        AxiomLoggingConfig.model_validate(config)

    def test_exporter_sends_the_token_and_dataset(self) -> None:
        """The exporter targets the shared endpoint with Axiom's two headers."""
        config = _axiom_config()
        exporter = _axiom_exporter(config, "xaat-key", "ds")
        assert exporter._endpoint == config["otlp_endpoint"]  # noqa: SLF001
        assert exporter._headers == {  # noqa: SLF001
            "Authorization": "Bearer xaat-key",
            "X-Axiom-Dataset": "ds",
        }
        exporter.shutdown()

    def test_export_thread_records_are_not_shipped(self) -> None:
        """A record logged on the SDK's export thread is kept off the pipeline."""
        record = logging.LogRecord("t", logging.WARNING, __file__, 1, "x", None, None)
        record.threadName = "OtelBatchLogRecordProcessor"
        assert _not_from_export_thread(record) is False
        record.threadName = "MainThread"
        assert _not_from_export_thread(record) is True
