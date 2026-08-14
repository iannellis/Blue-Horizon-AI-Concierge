"""Streamlit chat UI for the Blue Horizon AI Concierge.

Provides a single-thread chat interface that communicates with the Blue Horizon
FastAPI backend via its REST API.

Run with:
    streamlit run ui/app.py

Environment variables:
    BLUE_HORIZON_API_URL: Base URL for the Blue Horizon API.
        Defaults to ``http://localhost:8000``.
    GOOGLE_CLIENT_ID: When set, enables Google OAuth authentication.
        Set alongside ``GOOGLE_CLIENT_SECRET`` as HuggingFace Space secrets.
        Access is controlled via Google Cloud Console's OAuth test users list.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import httpx
import streamlit as st
from dotenv import load_dotenv

if TYPE_CHECKING:
    from streamlit.delta_generator import DeltaGenerator

load_dotenv()

logger = logging.getLogger(__name__)

_API_BASE: str = os.getenv("BLUE_HORIZON_API_URL", "http://localhost:8000").rstrip("/")
_AUTH_ENABLED: bool = bool(os.getenv("GOOGLE_CLIENT_ID"))
_CHAT_TIMEOUT_S: float = 90.0
_HEALTH_TIMEOUT_S: float = 3.0
# Real DB-backed reads (guest list, bookings), as opposed to the bare health
# ping above -- the booking pool opens connections on demand (min_size=0,
# see app_config.toml), so a request after any idle stretch can pay a Neon
# compute cold-start on top of the query itself.
_DATA_TIMEOUT_S: float = 20.0
_HTTP_OK: int = 200
_HTTP_SERVICE_UNAVAILABLE: int = 503
_HEALTH_POLL_ONLINE_INTERVAL_S: int = 30
_HEALTH_POLL_OFFLINE_INTERVAL_S: int = 5


# ============================
# Session state
# ============================


def _init_session_state() -> None:
    """Initialise all required ``st.session_state`` keys on first run.

    Creates a fresh conversation thread ID, an empty message history, and
    records the current time as the last activity timestamp (reserved for
    future inactivity-timeout support).
    """
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "customer_id" not in st.session_state:
        st.session_state.customer_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_activity" not in st.session_state:
        # Reserved for future 10-minute inactivity timeout.
        st.session_state.last_activity = datetime.now(UTC)
    if "pending_toast" not in st.session_state:
        st.session_state.pending_toast = None
    if "pending_proposal" not in st.session_state:
        st.session_state.pending_proposal = None


def _reset_session() -> None:
    """Clear the conversation and start a new thread.

    Generates a fresh ``thread_id``, empties the message history, resets the
    last-activity timestamp, and dismisses any pending proposal -- a dialog
    from a conversation that no longer exists must not still be confirmable.
    """
    _dismiss_pending_proposal()
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.last_activity = datetime.now(UTC)


# ============================
# API calls
# ============================


def _check_health() -> bool:
    """Poll the API health endpoint.

    Returns:
        ``True`` if the API responds with HTTP 200, ``False`` otherwise
        (including network errors and HTTP 503 while the agent is starting up).

    """
    try:
        response = httpx.get(f"{_API_BASE}/v1/health", timeout=_HEALTH_TIMEOUT_S)
        return response.status_code == _HTTP_OK  # noqa: TRY300
    except Exception:  # noqa: BLE001
        return False


def _call_reset() -> str | None:
    """Call the reset endpoint to restore the working database branch.

    Returns:
        ``None`` on success, or a user-facing error string on failure.

    """
    try:
        response = httpx.post(f"{_API_BASE}/v1/reset", timeout=_CHAT_TIMEOUT_S)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == _HTTP_SERVICE_UNAVAILABLE:
            return "Reset is not configured on this deployment."
        return f"Reset failed ({exc.response.status_code}). Please try again."
    except Exception as exc:  # noqa: BLE001
        return f"Could not reach the API: {exc}"
    else:
        return None


@st.cache_data(ttl=300)
def _fetch_customers_uncached() -> list[dict[str, Any]]:
    """Fetch the guest list for the identity dropdown.

    Cached for five minutes: the seeded customer list never changes between
    a demo database reload. Streamlit only caches on a successful return, so
    a failed lookup (e.g. a cold booking-DB connection) is retried on the
    next call instead of being memoized as "no guests."

    Returns:
        list[dict[str, Any]]: `{customer_id, first_name, last_name}` per
        guest.

    Raises:
        httpx.HTTPError: If the API could not be reached or returned an
            error status.

    """
    response = httpx.get(f"{_API_BASE}/v1/customers", timeout=_DATA_TIMEOUT_S)
    response.raise_for_status()
    return response.json()


def _fetch_customers() -> list[dict[str, Any]]:
    """Fetch the guest list, logging and falling back to empty on failure.

    Returns:
        list[dict[str, Any]]: `{customer_id, first_name, last_name}` per
        guest, or an empty list if the API could not be reached.

    """
    try:
        return _fetch_customers_uncached()
    except Exception:
        logger.warning("Could not fetch the guest list.", exc_info=True)
        return []


def _fetch_bookings(customer_id: int) -> list[dict[str, Any]]:
    """Fetch one guest's bookings for the reservations panel.

    Args:
        customer_id: Guest whose bookings to fetch.

    Returns:
        list[dict[str, Any]]: Booking summaries, or an empty list if the API
        could not be reached.

    """
    try:
        response = httpx.get(
            f"{_API_BASE}/v1/bookings",
            params={"customer_id": customer_id},
            timeout=_DATA_TIMEOUT_S,
        )
        response.raise_for_status()
        return response.json().get("bookings", [])
    except Exception:
        logger.warning(
            "Could not fetch bookings for customer_id=%s.",
            customer_id,
            exc_info=True,
        )
        return []


def _confirm_proposal(proposal_id: str, customer_id: int) -> dict[str, Any] | None:
    """Confirm a pending proposal.

    Args:
        proposal_id: Proposal to confirm.
        customer_id: Guest confirming it.

    Returns:
        dict[str, Any] | None: The confirm endpoint's JSON body, or ``None``
        on failure.

    """
    try:
        response = httpx.post(
            f"{_API_BASE}/v1/booking/confirm",
            json={"proposal_id": proposal_id, "customer_id": customer_id},
            timeout=_CHAT_TIMEOUT_S,
        )
        response.raise_for_status()
        return response.json()
    except Exception:  # noqa: BLE001
        return None


def _dismiss_proposal(proposal_id: str, customer_id: int) -> None:
    """Dismiss a pending proposal, best-effort.

    Failures are swallowed: the proposal's own server-side TTL is the
    backstop, and the caller has already dropped it from local state either
    way.

    Args:
        proposal_id: Proposal to dismiss.
        customer_id: Guest dismissing it.

    """
    with contextlib.suppress(Exception):
        httpx.post(
            f"{_API_BASE}/v1/booking/dismiss",
            json={"proposal_id": proposal_id, "customer_id": customer_id},
            timeout=_HEALTH_TIMEOUT_S,
        )


def _dismiss_pending_proposal() -> None:
    """Dismiss and clear any proposal left pending in session state."""
    proposal = st.session_state.get("pending_proposal")
    if proposal is None:
        return
    customer_id = st.session_state.get("customer_id")
    if customer_id is not None:
        _dismiss_proposal(proposal["proposal_id"], customer_id)
    st.session_state.pending_proposal = None


# ============================
# Rendering
# ============================


@st.fragment(run_every=_HEALTH_POLL_ONLINE_INTERVAL_S)
def _render_online_poll() -> None:
    """Silently poll the API and trigger a full app rerun if it goes offline.

    Registered only while the API is online. Streamlit removes the fragment
    from the render tree on the next full rerun (when the API is down), so
    polling stops automatically once the chatbot goes offline.
    """
    if not _check_health():
        st.rerun()


@st.fragment(run_every=_HEALTH_POLL_OFFLINE_INTERVAL_S)
def _render_recovery_poll() -> None:
    """Silently poll the API and trigger a full app rerun when it recovers.

    Registered only while the API is offline. Streamlit removes the fragment
    from the render tree on the next full rerun (when the API is back), so
    polling stops automatically once the chatbot comes online.
    """
    if _check_health():
        st.rerun()


def _render_login_page() -> None:
    """Render the sign-in page shown to unauthenticated visitors."""
    st.info("Please sign in with your Google account to access the concierge.")
    st.button(
        "Sign in with Google",
        on_click=st.login,
        use_container_width=True,
    )


def _render_customer_picker() -> None:
    """Render the guest dropdown and reset the session on a change of guest."""
    customers = _fetch_customers()
    if not customers:
        st.warning("Could not load the guest list.")
        return

    labels = [f"{c['first_name']} {c['last_name']}" for c in customers]
    ids = [c["customer_id"] for c in customers]
    current = st.session_state.customer_id
    index = ids.index(current) if current in ids else 0

    selected_label = st.selectbox("Guest", labels, index=index, key="guest_select")
    selected_id = ids[labels.index(selected_label)]

    if selected_id != current:
        st.session_state.customer_id = selected_id
        _reset_session()
        st.rerun()


def _render_reservations() -> None:
    """Render the "Your reservations" sidebar panel for the current guest.

    Built strictly from `GET /v1/bookings`, never from chat text -- the same
    rule the confirmation dialog follows, and for the same reason: a receipt
    the guest cannot go back and check is barely a receipt.
    """
    st.subheader("Your reservations")
    customer_id = st.session_state.customer_id
    if customer_id is None:
        return

    bookings = _fetch_bookings(customer_id)
    if not bookings:
        st.caption("No reservations yet.")
        return

    for booking in bookings:
        label = booking["confirmation_number"] or f"Booking {booking['booking_id']}"
        with st.container(border=True):
            st.markdown(f"**{label}** ({booking['status']})")
            for room in booking["rooms"]:
                st.caption(
                    f"Room {room['room_number']}: {room['check_in']} → "
                    f"{room['check_out']} — ${room['total_amount']}",
                )
            if booking["rooms"]:
                st.caption(f"Total: ${booking['total_amount']}")


def _render_sidebar() -> None:
    """Render the sidebar: auth info, guest picker, health status, and controls."""
    with st.sidebar:
        st.header("Blue Horizon")

        if _AUTH_ENABLED:
            st.caption(f"Signed in as **{st.user.name or st.user.email}**")
            st.button("Sign out", on_click=st.logout, use_container_width=True)
            st.divider()

        _render_customer_picker()
        st.divider()

        if _check_health():
            st.success("Chatbot: Online")
            _render_online_poll()
        else:
            st.error("Chatbot: Offline")
            _render_recovery_poll()

        st.divider()
        _render_reservations()
        st.divider()

        if st.button("New Conversation", use_container_width=True):
            _reset_session()
            st.rerun()

        if st.button("Reset demo database", use_container_width=True):
            with st.spinner("Resetting…"):
                error = _call_reset()
            if error:
                st.error(error)
            else:
                _reset_session()
                st.session_state.pending_toast = "Demo database reset."
                st.rerun()


def _md(text: str) -> None:
    """Render text as Markdown with dollar signs escaped to prevent LaTeX parsing.

    Args:
        text: Text to render as Markdown.

    """
    st.markdown(text.replace("$", r"\$"))


def _parse_sse_line(line: str) -> dict[str, Any] | None:
    """Parse one line of an SSE stream into its JSON event payload.

    Args:
        line: One line read from the response body.

    Returns:
        dict[str, Any] | None: The decoded event, or ``None`` for lines that
        are not a ``data: `` event or fail to decode (e.g. a keepalive
        comment line).

    """
    if not line.startswith("data: "):
        return None
    try:
        return json.loads(line[6:])
    except json.JSONDecodeError:
        return None


def _handle_stream_event(
    event: dict[str, Any],
    status: DeltaGenerator,
    pending_proposal: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, tuple[str, dict[str, Any] | None] | None]:
    """Apply one SSE event to the status widget.

    Args:
        event: Decoded SSE event.
        status: The turn's `st.status` widget.
        pending_proposal: Proposal captured from an earlier event this turn,
            if any.

    Returns:
        tuple: `(pending_proposal, result)`. `pending_proposal` carries
        forward to the next event; `result` is non-``None`` exactly when the
        caller should return it immediately (an `error` or `done` event ends
        the turn).

    """
    event_type = event.get("type")
    if event_type == "stage":
        status.update(label=event["label"], state="running")
    elif event_type == "error":
        status.update(label="Error", state="error")
        return pending_proposal, (
            str(event.get("message", "Something went wrong.")), None,
        )
    elif event_type == "done":
        status.update(label="Done", state="complete")
        response_text = str(event.get("response", "No response received."))
        return pending_proposal, (response_text, pending_proposal)
    elif event_type == "proposal":
        pending_proposal = {
            "proposal_id": event["proposal_id"],
            "action": event["action"],
            "summary": event["summary"],
        }
    return pending_proposal, None


def _stream_message(
    thread_id: str, customer_id: int, text: str,
) -> tuple[str, dict[str, Any] | None]:
    """Stream a chat turn from the unified endpoint and display live progress.

    Connects to ``POST /v1/chat`` with ``Accept: text/event-stream`` and
    updates a :func:`streamlit.status` widget as each event arrives.

    This function must be called inside a ``st.chat_message`` context so that
    the status widget is rendered inside the assistant bubble.

    Args:
        thread_id: Unique identifier for the current conversation thread.
        customer_id: The current guest, bound to `thread_id` on first use.
        text: The user's message text.

    Returns:
        tuple[str, dict[str, Any] | None]: The assistant's reply text (or a
        user-facing error message), and a pending proposal dict if the turn
        created one.

    """
    status = st.status("Routing your request…", state="running")
    pending_proposal: dict[str, Any] | None = None
    try:
        with httpx.stream(
            "POST",
            f"{_API_BASE}/v1/chat",
            json={"thread_id": thread_id, "customer_id": customer_id, "text": text},
            headers={"Accept": "text/event-stream"},
            timeout=_CHAT_TIMEOUT_S,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                event = _parse_sse_line(line)
                if event is None:
                    continue
                pending_proposal, result = _handle_stream_event(
                    event, status, pending_proposal,
                )
                if result is not None:
                    return result
    except httpx.HTTPStatusError as exc:
        error_message = _http_error_message(exc)
    except httpx.TimeoutException:
        error_message = (
            "The request timed out. The agent may be busy — please try again."
        )
    except Exception as exc:  # noqa: BLE001
        error_message = f"Could not reach the API: {exc}"
    else:
        return "No response received.", pending_proposal

    status.update(label="Error", state="error")
    return error_message, None


def _http_error_message(exc: httpx.HTTPStatusError) -> str:
    """Translate a chat-endpoint HTTP error into user-facing copy.

    Args:
        exc: HTTP status error raised by the chat request.

    Returns:
        str: User-facing error message.

    """
    if exc.response.status_code == _HTTP_SERVICE_UNAVAILABLE:
        return "The system is still starting up. Please try again in a moment."
    return (
        f"The server returned an error ({exc.response.status_code}). "
        "Please try again."
    )


def _render_book_summary(summary: dict[str, Any]) -> None:
    """Render a `book` proposal's line items inside the confirmation dialog.

    Args:
        summary: Proposal summary with a `rooms` list and a `total`.

    """
    for room in summary["rooms"]:
        st.write(
            f"Room {room['room_number']}: {room['check_in']} → {room['check_out']} "
            f"({room['nights']} nights) — ${room['amount']}",
        )
    st.markdown(f"**Total: ${summary['total']}**")


def _render_cancel_summary(summary: dict[str, Any]) -> None:
    """Render a `cancel` proposal's line items inside the confirmation dialog.

    Args:
        summary: Proposal summary with a `rooms` list and a refund `total`.

    """
    for room in summary["rooms"]:
        st.write(f"Room {room['room_number']}: refund ${room['amount']}")
    st.markdown(f"**Total refund: ${summary['total']}**")


def _render_modify_summary(summary: dict[str, Any]) -> None:
    """Render a `modify` proposal's before/after changes inside the dialog.

    Args:
        summary: Proposal summary with a `changes` list and a new `total`.

    """
    for change in summary["changes"]:
        before, after = change["before"], change["after"]
        st.write(
            f"Room {before['room_number']} ({before['check_in']} → "
            f"{before['check_out']}) **becomes** Room {after['room_number']} "
            f"({after['check_in']} → {after['check_out']})",
        )
    st.markdown(f"**New total: ${summary['total']}**")


_SUMMARY_RENDERERS = {
    "book": _render_book_summary,
    "cancel": _render_cancel_summary,
    "modify": _render_modify_summary,
}


@st.dialog("Review your request")
def _render_proposal_dialog(proposal: dict[str, Any]) -> None:
    """Render the confirmation dialog for a pending proposal.

    Built strictly from `proposal["summary"]` -- resolved, priced line items
    computed by the application -- and never from assistant chat text. This
    is the load-bearing detail from the design this UI implements: the
    dialog and the eventual commit must never be able to disagree, so do not
    "simplify" this to reuse the assistant's message text instead.

    Args:
        proposal: Pending proposal dict with `proposal_id`, `action`, and
            `summary`.

    """
    renderer = _SUMMARY_RENDERERS[proposal["action"]]
    renderer(proposal["summary"])

    col_confirm, col_cancel = st.columns(2)
    if col_confirm.button("Confirm", type="primary", use_container_width=True):
        customer_id = st.session_state.customer_id
        result = _confirm_proposal(proposal["proposal_id"], customer_id)
        st.session_state.pending_proposal = None
        message = (
            result["message"] if result else "Could not reach the API to confirm."
        )
        st.session_state.messages.append({"role": "assistant", "content": message})
        st.rerun()

    if col_cancel.button("Cancel", use_container_width=True):
        customer_id = st.session_state.customer_id
        _dismiss_proposal(proposal["proposal_id"], customer_id)
        st.session_state.pending_proposal = None
        st.rerun()


def _render_chat() -> None:
    """Render the chat message history, pending proposal, and new user input."""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            _md(msg["content"])

    if st.session_state.pending_proposal is not None:
        _render_proposal_dialog(st.session_state.pending_proposal)

    placeholder = "Ask about the hotel, or get help searching or booking rooms..."
    if prompt := st.chat_input(
        placeholder, disabled=st.session_state.customer_id is None,
    ):
        st.session_state.last_activity = datetime.now(UTC)
        _dismiss_pending_proposal()

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            _md(prompt)

        with st.chat_message("assistant"):
            reply, proposal = _stream_message(
                st.session_state.thread_id, st.session_state.customer_id, prompt,
            )
            _md(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.session_state.pending_proposal = proposal
        st.rerun()


# ============================
# Entry point
# ============================


def main() -> None:
    """Configure the Streamlit page and render all UI components."""
    st.set_page_config(
        page_title="Blue Horizon Concierge",
        page_icon="🏨",
        layout="centered",
    )
    st.title("Blue Horizon Concierge 🌅")
    st.subheader("AI concierge to help with room bookings and answer questions about "
                 "the hotel and its services/amenities")

    if _AUTH_ENABLED and not st.user.is_logged_in:
        _render_login_page()
        st.stop()

    _init_session_state()
    if msg := st.session_state.pending_toast:
        st.toast(msg)
        st.session_state.pending_toast = None
    _render_sidebar()
    _render_chat()


if __name__ == "__main__":
    main()
