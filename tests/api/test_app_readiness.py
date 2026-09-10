"""Unit tests for the not-ready gates in `blue_horizon.api.app`.

Unlike `tests/api/test_app.py` (`db_integration`-marked, drives the real
stack), the gates covered here are pure: `chat()`, `list_customers()`, and
`list_bookings()` each check readiness before doing any I/O, so a mocked
`orchestrator` is sufficient and no real Postgres/Redis/OpenAI is needed.
"""
# ruff: noqa: S101

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

import blue_horizon.api.app as app_module
from blue_horizon.agents.orchestration import Readiness

if TYPE_CHECKING:
    from collections.abc import Iterator

_HTTP_SERVICE_UNAVAILABLE = 503


@pytest.fixture
def mock_orchestrator(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace the module-level `orchestrator` singleton with a mock.

    `app_module`'s route handlers close over the module-global `orchestrator`
    name, so patching that attribute is enough for every handler to see the
    mock without touching how the app is constructed.

    Args:
        monkeypatch: Pytest fixture for reversible attribute patches.

    Returns:
        The `MagicMock` now installed as `blue_horizon.api.app.orchestrator`,
        with `start`/`stop` stubbed so the real `lifespan` can run harmlessly.

    """
    mock = MagicMock()
    mock.start = AsyncMock()
    mock.stop = AsyncMock()
    monkeypatch.setattr(app_module, "orchestrator", mock)
    return mock


@pytest.fixture
def client(mock_orchestrator: MagicMock) -> Iterator[TestClient]:  # noqa: ARG001
    """Build a `TestClient` around the app with its orchestrator mocked.

    Args:
        mock_orchestrator: Only depended on for fixture ordering -- the
            patch must be installed before `lifespan` runs `start()`.

    Yields:
        TestClient: Ready to issue requests against the mocked app.

    """
    with TestClient(app_module.app) as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# /v1/chat readiness gate
# ---------------------------------------------------------------------------


class TestChatNotReady:
    """`/v1/chat` returns 503 on both content-negotiated branches when not ready."""

    def test_json_branch_returns_503_with_retry_after(
        self, client: TestClient, mock_orchestrator: MagicMock,
    ) -> None:
        """The plain-JSON branch is gated before `ainvoke` is ever called."""
        mock_orchestrator.is_ready = False
        mock_orchestrator.readiness = Readiness.STARTING
        mock_orchestrator.get_readiness_message.return_value = "Still starting up."

        response = client.post(
            "/v1/chat",
            json={"thread_id": "t1", "customer_id": 1, "text": "hi"},
        )

        assert response.status_code == _HTTP_SERVICE_UNAVAILABLE
        assert response.headers["retry-after"]
        body = response.json()
        assert body["status"] == "starting"
        assert body["message"] == "Still starting up."
        assert body["retry_after_s"] > 0
        mock_orchestrator.ainvoke.assert_not_called()

    def test_sse_branch_returns_503_json_not_a_stream(
        self, client: TestClient, mock_orchestrator: MagicMock,
    ) -> None:
        """The SSE branch gets the same plain 503 body, never a `done` event."""
        mock_orchestrator.is_ready = False
        mock_orchestrator.readiness = Readiness.STARTING
        mock_orchestrator.get_readiness_message.return_value = "Still starting up."

        response = client.post(
            "/v1/chat",
            json={"thread_id": "t1", "customer_id": 1, "text": "hi"},
            headers={"Accept": "text/event-stream"},
        )

        assert response.status_code == _HTTP_SERVICE_UNAVAILABLE
        assert response.headers["retry-after"]
        assert not response.headers["content-type"].startswith("text/event-stream")
        body = response.json()
        assert body["status"] == "starting"
        assert "done" not in response.text
        mock_orchestrator.ainvoke_stream.assert_not_called()

    def test_failed_readiness_reports_failed_status(
        self, client: TestClient, mock_orchestrator: MagicMock,
    ) -> None:
        """A permanently failed orchestrator is reported as `failed`, not `starting`."""
        mock_orchestrator.is_ready = False
        mock_orchestrator.readiness = Readiness.FAILED
        mock_orchestrator.get_readiness_message.return_value = "Won't recover."

        response = client.post(
            "/v1/chat",
            json={"thread_id": "t1", "customer_id": 1, "text": "hi"},
        )

        assert response.status_code == _HTTP_SERVICE_UNAVAILABLE
        assert response.json()["status"] == "failed"


# ---------------------------------------------------------------------------
# /v1/customers and /v1/bookings readiness gates
# ---------------------------------------------------------------------------


class TestDataEndpointsNotReady:
    """`/v1/customers` and `/v1/bookings` return 503, not an uncaught 500."""

    def test_list_customers_returns_503(
        self, client: TestClient, mock_orchestrator: MagicMock,
    ) -> None:
        """A not-yet-initialized write pool is reported as 503."""
        mock_orchestrator.get_readiness_message.return_value = "Still starting up."
        booking_resources = mock_orchestrator.get_booking_resources.return_value
        booking_resources.get_write_pool.side_effect = RuntimeError(
            "BookingSqlResources is not initialized; call await startup_check() first",
        )

        response = client.get("/v1/customers")

        assert response.status_code == _HTTP_SERVICE_UNAVAILABLE
        assert response.headers["retry-after"]
        assert response.json()["detail"] == "Still starting up."

    def test_list_bookings_returns_503(
        self, client: TestClient, mock_orchestrator: MagicMock,
    ) -> None:
        """A not-yet-initialized write pool is reported as 503."""
        mock_orchestrator.get_readiness_message.return_value = "Still starting up."
        booking_resources = mock_orchestrator.get_booking_resources.return_value
        booking_resources.get_write_pool.side_effect = RuntimeError(
            "BookingSqlResources is not initialized; call await startup_check() first",
        )

        response = client.get("/v1/bookings", params={"customer_id": 1})

        assert response.status_code == _HTTP_SERVICE_UNAVAILABLE
        assert response.headers["retry-after"]
        assert response.json()["detail"] == "Still starting up."
