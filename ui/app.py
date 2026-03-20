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

import json
import os
import uuid
from datetime import UTC, datetime

import httpx
import streamlit as st

_API_BASE: str = os.getenv("BLUE_HORIZON_API_URL", "http://localhost:8000").rstrip("/")
_AUTH_ENABLED: bool = bool(os.getenv("GOOGLE_CLIENT_ID"))
_CHAT_TIMEOUT_S: float = 90.0
_HEALTH_TIMEOUT_S: float = 3.0
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
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_activity" not in st.session_state:
        # Reserved for future 10-minute inactivity timeout.
        st.session_state.last_activity = datetime.now(UTC)
    if "pending_toast" not in st.session_state:
        st.session_state.pending_toast = None


def _reset_session() -> None:
    """Clear the conversation and start a new thread.

    Generates a fresh ``thread_id``, empties the message history, and resets
    the last-activity timestamp.
    """
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


def _send_message(thread_id: str, text: str) -> str:
    """Send a user message to the chat API and return the assistant's reply.

    Args:
        thread_id: Unique identifier for the current conversation thread.
        text: The user's message text.

    Returns:
        The assistant's reply as a plain string, or a user-facing error message
        if the request fails.

    """
    try:
        response = httpx.post(
            f"{_API_BASE}/v1/chat",
            json={"thread_id": thread_id, "text": text},
            timeout=_CHAT_TIMEOUT_S,
        )
        response.raise_for_status()
        data = response.json()
        ai_messages = [m for m in data.get("messages", []) if m["type"] == "ai"]
        return ai_messages[-1]["content"] if ai_messages else "No response received."
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == _HTTP_SERVICE_UNAVAILABLE:
            return "The system is still starting up. Please try again in a moment."
        return (
            f"The server returned an error ({exc.response.status_code}). "
            "Please try again."
        )
    except httpx.TimeoutException:
        return "The request timed out. The agent may be busy — please try again."
    except Exception as exc:  # noqa: BLE001
        return f"Could not reach the API: {exc}"


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



def _render_sidebar() -> None:
    """Render the sidebar: auth info, health status, and conversation controls."""
    with st.sidebar:
        st.header("Blue Horizon")

        if _AUTH_ENABLED:
            st.caption(f"Signed in as **{st.user.name or st.user.email}**")
            st.button("Sign out", on_click=st.logout, use_container_width=True)
            st.divider()

        if _check_health():
            st.success("Chatbot: Online")
            _render_online_poll()
        else:
            st.error("Chatbot: Offline")
            _render_recovery_poll()

        st.divider()

        if st.button("New Conversation", use_container_width=True):
            _reset_session()
            st.rerun()

        if st.button("Clear My Bookings", use_container_width=True):
            with st.spinner("Clearing bookings…"):
                error = _call_reset()
            if error:
                st.error(error)
            else:
                _reset_session()
                st.session_state.pending_toast = "Bookings cleared."
                st.rerun()


def _md(text: str) -> None:
    """Render text as Markdown with dollar signs escaped to prevent LaTeX parsing.

    Args:
        text: Text to render as Markdown.

    """
    st.markdown(text.replace("$", r"\$"))


def _stream_message(thread_id: str, text: str) -> str:
    """Stream stage events from the API and display a live progress indicator.

    Connects to the ``/v1/chat/stream`` SSE endpoint and updates a
    :func:`streamlit.status` widget as each stage event arrives.  Returns
    the final response text from the ``done`` event.

    This function must be called inside a ``st.chat_message`` context so that
    the status widget is rendered inside the assistant bubble.

    Args:
        thread_id: Unique identifier for the current conversation thread.
        text: The user's message text.

    Returns:
        The assistant's reply as a plain string, or a user-facing error
        message if the request fails.

    """
    status = st.status("Routing your request\u2026", state="running")
    try:
        with httpx.stream(
            "POST",
            f"{_API_BASE}/v1/chat/stream",
            json={"thread_id": thread_id, "text": text},
            timeout=_CHAT_TIMEOUT_S,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                try:
                    event = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                event_type = event.get("type")
                if event_type == "stage":
                    status.update(label=event["label"], state="running")
                elif event_type == "done":
                    status.update(label="Done", state="complete")
                    return str(event.get("response", "No response received."))
    except httpx.HTTPStatusError as exc:
        status.update(label="Error", state="error")
        if exc.response.status_code == _HTTP_SERVICE_UNAVAILABLE:
            return "The system is still starting up. Please try again in a moment."
        return (
            f"The server returned an error ({exc.response.status_code}). "
            "Please try again."
        )
    except httpx.TimeoutException:
        status.update(label="Error", state="error")
        return "The request timed out. The agent may be busy — please try again."
    except Exception as exc:  # noqa: BLE001
        status.update(label="Error", state="error")
        return f"Could not reach the API: {exc}"
    return "No response received."


def _render_chat() -> None:
    """Render the chat message history and handle new user input."""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            _md(msg["content"])

    placeholder = "Ask about the hotel, or get help searching or booking rooms..."
    if prompt := st.chat_input(placeholder):
        st.session_state.last_activity = datetime.now(UTC)

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            _md(prompt)

        with st.chat_message("assistant"):
            reply = _stream_message(st.session_state.thread_id, prompt)
            _md(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})


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
