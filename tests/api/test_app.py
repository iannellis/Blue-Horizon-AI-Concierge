"""`TestClient` coverage of the FastAPI app against the real live stack.

Triggers the real `lifespan` startup, so this needs a reachable Postgres
(`PGSQL_RW_DB_URL`/`PGSQL_RO_DB_URL`), Redis (`REDIS_URL`), and an
`OPENAI_API_KEY` -- every agent in `blue_horizon/app_config.toml` runs on
OpenAI models, so no other provider key is required. A handful of tests
drive one real chat turn each; everything else (proposal creation for the
confirm/dismiss tests) goes through `ProposalStore` directly to avoid
spending an LLM call on scenarios that don't need one.

Marked `db_integration` and excluded from the default `pytest` run -- see
`.github/workflows/ci.yml`'s `db-integration-tests` job, which resets the
Development branch before running these.
"""
# ruff: noqa: S101

from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import platform
import time
import uuid
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import pytest
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from blue_horizon.agents.booking import write_ops

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from fastapi.testclient import TestClient

pytestmark = pytest.mark.db_integration

if platform.system() == "Windows":
    # psycopg's async mode cannot run on Windows's default ProactorEventLoop.
    # Set before `TestClient` (and its internal anyio portal thread) or any
    # direct pool below is created, matching `eval/booking_db_manager.py`.
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

_READY_TIMEOUT_S = 90.0
_READY_POLL_INTERVAL_S = 1.0


def _require_env(name: str) -> str:
    """Return an environment variable, skipping the test if it is unset.

    Args:
        name: Environment variable name.

    Returns:
        The variable's value.

    """
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"{name} not set; skipping db_integration test.")
    return value


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    """Start the real app once for the module and wait for it to be ready.

    Yields:
        A `TestClient` with the app's `lifespan` startup already complete.

    """
    for name in ("REDIS_URL", "PGSQL_RW_DB_URL", "PGSQL_RO_DB_URL", "OPENAI_API_KEY"):
        _require_env(name)

    from fastapi.testclient import TestClient as _TestClient  # noqa: PLC0415

    from blue_horizon.api.app import app  # noqa: PLC0415

    with _TestClient(app) as test_client:
        deadline = time.monotonic() + _READY_TIMEOUT_S
        last_status_code = None
        while time.monotonic() < deadline:
            response = test_client.get("/v1/health")
            last_status_code = response.status_code
            if response.status_code == 200:  # noqa: PLR2004
                break
            time.sleep(_READY_POLL_INTERVAL_S)
        else:
            pytest.fail(
                f"App did not become ready within {_READY_TIMEOUT_S}s "
                f"(last /v1/health status: {last_status_code}).",
            )
        yield test_client


def _unique_thread_id() -> str:
    """Build a thread id unique to one test, so tests never share history.

    Returns:
        A fresh UUID4 hex string.

    """
    return f"test-app-{uuid.uuid4().hex}"


@asynccontextmanager
async def _rw_pool() -> AsyncIterator[AsyncConnectionPool[Any]]:
    """Open a short-lived read-write pool independent of the app's own pool.

    The app's pool lives on `TestClient`'s internal event loop; a helper
    pool used from this module's own `asyncio.run()` calls must be entirely
    separate.

    Yields:
        An open `AsyncConnectionPool`.

    """
    db_url = _require_env("PGSQL_RW_DB_URL")
    pool: AsyncConnectionPool[Any] = AsyncConnectionPool(
        conninfo=db_url, min_size=0, max_size=2, open=False,
    )
    await pool.open()
    try:
        yield pool
    finally:
        await pool.close()


async def _first_customer_id() -> int:
    """Fetch one seeded customer id.

    Returns:
        int: A `customer_id`.

    Raises:
        RuntimeError: If no customers are seeded.

    """
    async with _rw_pool() as pool, pool.connection() as conn, conn.cursor(
        row_factory=dict_row,
    ) as cur:
        await cur.execute(
            "SELECT customer_id FROM customers ORDER BY customer_id LIMIT 1",
        )
        row = await cur.fetchone()
    if row is None:
        msg = "No seeded customers; cannot run API tests."
        raise RuntimeError(msg)
    return row["customer_id"]


async def _find_available_room_request() -> write_ops.RoomRequest:
    """Find one room and a single currently-bookable night.

    Returns:
        write_ops.RoomRequest: A one-night request that `commit_booking`
        will accept.

    Raises:
        RuntimeError: If no bookable night is found among a bounded search.

    """
    async with _rw_pool() as pool, pool.connection() as conn, conn.cursor(
        row_factory=dict_row,
    ) as cur:
        await cur.execute(
            "SELECT room_id, room_number FROM rooms ORDER BY room_number LIMIT 30",
        )
        candidates = await cur.fetchall()
        for candidate in candidates:
            await cur.execute(
                """
                SELECT date FROM room_availability
                WHERE room_id = %s AND status = 'Available' AND price IS NOT NULL
                ORDER BY date LIMIT 1
                """,
                (candidate["room_id"],),
            )
            row = await cur.fetchone()
            if row is not None:
                return write_ops.RoomRequest(
                    room_id=candidate["room_id"],
                    room_number=candidate["room_number"],
                    check_in=row["date"],
                    check_out=row["date"] + dt.timedelta(days=1),
                )
    msg = "No available room-night found among the first 30 rooms."
    raise RuntimeError(msg)


class TestCustomers:
    """`GET /v1/customers` -- backs the UI's automated guest assignment."""

    def test_returns_seeded_guests(self, client: TestClient) -> None:
        """The endpoint returns a non-empty list of guest summaries."""
        response = client.get("/v1/customers")
        assert response.status_code == 200  # noqa: PLR2004
        customers = response.json()
        assert len(customers) > 0
        first = customers[0]
        assert {"customer_id", "first_name", "last_name"} <= first.keys()


class TestBookingsScoping:
    """`GET /v1/bookings` -- scoped strictly to the requested guest."""

    def test_only_returns_the_requested_customers_bookings(
        self, client: TestClient,
    ) -> None:
        """A booking made for one guest never appears under another's id."""

        async def _setup() -> tuple[int, write_ops.CommitResult]:
            async with _rw_pool() as pool:
                customer_id = await _first_customer_id()
                request = await _find_available_room_request()
                result = await write_ops.commit_booking(
                    pool, customer_id=customer_id, rooms=[request],
                )
                return customer_id, result

        async def _cleanup(customer_id: int, booking_id: int) -> None:
            async with _rw_pool() as pool:
                await write_ops.cancel_booking(
                    pool, customer_id=customer_id, booking_id=booking_id,
                )

        customer_id, committed = asyncio.run(_setup())
        try:
            owner_response = client.get(
                "/v1/bookings", params={"customer_id": customer_id},
            )
            assert owner_response.status_code == 200  # noqa: PLR2004
            owner_ids = {b["booking_id"] for b in owner_response.json()["bookings"]}
            assert committed.booking_id in owner_ids

            other_customer_id = customer_id + 1
            other_response = client.get(
                "/v1/bookings", params={"customer_id": other_customer_id},
            )
            assert other_response.status_code == 200  # noqa: PLR2004
            other_ids = {b["booking_id"] for b in other_response.json()["bookings"]}
            assert committed.booking_id not in other_ids
        finally:
            asyncio.run(_cleanup(customer_id, committed.booking_id))


class TestChatContentNegotiation:
    """`POST /v1/chat` -- one endpoint, negotiated by `Accept`."""

    def test_default_accept_returns_json(self, client: TestClient) -> None:
        """Without an SSE `Accept` header, the endpoint returns one JSON body."""
        customer_id = asyncio.run(_first_customer_id())
        response = client.post(
            "/v1/chat",
            json={
                "thread_id": _unique_thread_id(),
                "customer_id": customer_id,
                "text": "Hello",
            },
        )
        assert response.status_code == 200  # noqa: PLR2004
        assert response.headers["content-type"].startswith("application/json")
        body = response.json()
        assert "messages" in body
        ai_messages = [m for m in body["messages"] if m["type"] == "ai"]
        assert len(ai_messages) == 1

    def test_sse_accept_streams_events(self, client: TestClient) -> None:
        """An SSE `Accept` header streams `data:` lines ending in a `done` event."""
        customer_id = asyncio.run(_first_customer_id())
        payload = {
            "thread_id": _unique_thread_id(),
            "customer_id": customer_id,
            "text": "Hello",
        }
        events: list[dict[str, Any]] = []
        with client.stream(
            "POST",
            "/v1/chat",
            json=payload,
            headers={"Accept": "text/event-stream"},
        ) as response:
            assert response.status_code == 200  # noqa: PLR2004
            assert "text/event-stream" in response.headers["content-type"]
            events.extend(
                json.loads(line.removeprefix("data: "))
                for line in response.iter_lines()
                if line.startswith("data: ")
            )

        assert events
        assert events[-1]["type"] == "done"


class TestThreadCustomerBinding:
    """A `thread_id` is bound to whichever guest first uses it."""

    def test_mismatched_customer_on_bound_thread_is_rejected(
        self, client: TestClient,
    ) -> None:
        """A second guest reusing another guest's thread id gets 409."""
        first_customer_id = asyncio.run(_first_customer_id())
        thread_id = _unique_thread_id()

        first = client.post(
            "/v1/chat",
            json={
                "thread_id": thread_id,
                "customer_id": first_customer_id,
                "text": "Hello",
            },
        )
        assert first.status_code == 200  # noqa: PLR2004

        second = client.post(
            "/v1/chat",
            json={
                "thread_id": thread_id,
                "customer_id": first_customer_id + 1,
                "text": "Hello again",
            },
        )
        assert second.status_code == 409  # noqa: PLR2004


class TestBookingConfirmDismiss:
    """`POST /v1/booking/confirm` and `POST /v1/booking/dismiss`."""

    def test_confirm_unknown_proposal_returns_404(self, client: TestClient) -> None:
        """Confirming a proposal id that was never created returns 404."""
        customer_id = asyncio.run(_first_customer_id())
        response = client.post(
            "/v1/booking/confirm",
            json={"proposal_id": "does-not-exist", "customer_id": customer_id},
        )
        assert response.status_code == 404  # noqa: PLR2004

    def test_confirm_wrong_customer_returns_403(self, client: TestClient) -> None:
        """Confirming another guest's pending proposal returns 403.

        The ownership check runs before any write is attempted, so this
        proposal's `details`/`summary` never need to reference a real room.
        """
        from blue_horizon.api.app import orchestrator  # noqa: PLC0415

        owner_id = asyncio.run(_first_customer_id())
        other_id = owner_id + 1
        proposal = orchestrator.get_booking_resources().proposals.create(
            thread_id=_unique_thread_id(),
            customer_id=owner_id,
            action="book",
            summary={"rooms": [], "total": "0.00"},
            details=[],
        )
        response = client.post(
            "/v1/booking/confirm",
            json={"proposal_id": proposal.proposal_id, "customer_id": other_id},
        )
        assert response.status_code == 403  # noqa: PLR2004

    def test_dismiss_then_confirm_returns_dismissed_then_404(
        self, client: TestClient,
    ) -> None:
        """A dismissed proposal reports success, then can never be confirmed."""
        from blue_horizon.api.app import orchestrator  # noqa: PLC0415

        customer_id = asyncio.run(_first_customer_id())
        proposal = orchestrator.get_booking_resources().proposals.create(
            thread_id=_unique_thread_id(),
            customer_id=customer_id,
            action="book",
            summary={"rooms": [], "total": "0.00"},
            details=[],
        )

        dismiss_response = client.post(
            "/v1/booking/dismiss",
            json={"proposal_id": proposal.proposal_id, "customer_id": customer_id},
        )
        assert dismiss_response.status_code == 200  # noqa: PLR2004
        assert dismiss_response.json() == {"status": "dismissed"}

        confirm_response = client.post(
            "/v1/booking/confirm",
            json={"proposal_id": proposal.proposal_id, "customer_id": customer_id},
        )
        assert confirm_response.status_code == 404  # noqa: PLR2004

    def test_confirm_success_commits_and_returns_receipt_fields(
        self, client: TestClient,
    ) -> None:
        """A confirmed `book` proposal returns a real confirmation number."""
        from blue_horizon.api.app import orchestrator  # noqa: PLC0415

        customer_id = asyncio.run(_first_customer_id())
        request = asyncio.run(_find_available_room_request())

        # `proposals._commit_proposal` asserts the commit's real total
        # against `summary["total"]` -- exactly the defense-in-depth check
        # that would fire if a guest were ever shown one price and charged
        # another -- so the proposal here must carry the room's true price,
        # not a placeholder.
        async def _price() -> write_ops.PricedRoomStay:
            async with _rw_pool() as pool:
                return (await write_ops.price_rooms(pool, [request]))[0]

        priced = asyncio.run(_price())
        true_total = write_ops.fmt_money(priced.total_amount)
        summary = {
            "rooms": [write_ops.serialize_priced_stay(priced)],
            "total": true_total,
        }

        proposal = orchestrator.get_booking_resources().proposals.create(
            thread_id=_unique_thread_id(),
            customer_id=customer_id,
            action="book",
            summary=summary,
            details=[request],
        )

        response = client.post(
            "/v1/booking/confirm",
            json={"proposal_id": proposal.proposal_id, "customer_id": customer_id},
        )
        assert response.status_code == 200  # noqa: PLR2004
        body = response.json()
        try:
            assert body["status"] == "confirmed"
            assert body["already_confirmed"] is False
            assert body["confirmation_number"] == f"BH{body['booking_id']:06d}"
            assert body["total_amount"] == true_total
        finally:
            asyncio.run(_cancel_booking(customer_id, body["booking_id"]))


async def _cancel_booking(customer_id: int, booking_id: int) -> None:
    """Cancel a booking through the same real write path, for test cleanup.

    Args:
        customer_id: Owner of the booking.
        booking_id: Booking to cancel.

    """
    async with _rw_pool() as pool:
        await write_ops.cancel_booking(
            pool, customer_id=customer_id, booking_id=booking_id,
        )
