"""FastAPI application exposing chat, booking, and health endpoints.

``POST /v1/chat`` is a single, content-negotiated endpoint: an ``Accept:
text/event-stream`` request streams stage/proposal/done events over SSE,
anything else gets one JSON response. ``POST /v1/booking/confirm`` and
``POST /v1/booking/dismiss`` are the only callers of the booking write
functions -- the model can only *propose*, never commit. ``GET /v1/customers``
and ``GET /v1/bookings`` back the UI's identity picker and reservations
panel.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from blue_horizon.agents.booking import proposals as proposals_module
from blue_horizon.agents.booking import write_ops
from blue_horizon.agents.booking.receipts import receipt_message, serialize_write_result
from blue_horizon.agents.exceptions import ThreadCustomerMismatchError
from blue_horizon.agents.orchestration import OrchestrationManager, format_chat_response
from blue_horizon.config import load_app_config

load_dotenv()

logger = logging.getLogger(__name__)

_SSE_MEDIA_TYPE = "text/event-stream"
_KEEPALIVE_INTERVAL_S = 15.0


class ChatPayload(BaseModel):
    """The payload for the chat endpoint.

    Attributes:
        thread_id: The unique identifier for the conversation thread.
        customer_id: The guest sending this message, bound to `thread_id` on
            first use.
        text: The user's query text.

    """

    thread_id: str
    customer_id: int
    text: str


class ProposalActionPayload(BaseModel):
    """The payload for the confirm and dismiss endpoints.

    Attributes:
        proposal_id: Proposal being confirmed or dismissed.
        customer_id: Guest requesting the action; must own the proposal.

    """

    proposal_id: str
    customer_id: int


orchestrator = OrchestrationManager()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
    """Manage the lifespan of the FastAPI app.

    Starts the agent orchestrator when the app is launched and stops it when the app
    is shutdown.

    Arguments:
        _app: The FastAPI application (unused, required by FastAPI signature).

    Yields:
        None: Control to the application lifespan.

    """
    await orchestrator.start()
    yield
    await orchestrator.stop()


app = FastAPI(lifespan=lifespan)

router = APIRouter(prefix="/v1")


def _proposal_event(proposal: proposals_module.Proposal) -> dict[str, Any]:
    """Build the wire-format proposal payload shared by JSON and SSE responses.

    Args:
        proposal: Pending proposal to serialize.

    Returns:
        dict[str, Any]: `proposal_id`, `action`, and `summary`.

    """
    return {
        "proposal_id": proposal.proposal_id,
        "action": proposal.action,
        "summary": proposal.summary,
    }


def _error_message(exc: Exception) -> str:
    """Translate an exception into user-facing copy for an `error` event.

    Args:
        exc: Exception raised while streaming a chat turn.

    Returns:
        str: User-facing error message.

    """
    if isinstance(exc, ThreadCustomerMismatchError):
        return "This conversation belongs to a different guest."
    return load_app_config().orchestration.messages.error


@router.get("/health")
async def health() -> JSONResponse:
    """Return the readiness status of the orchestrator.

    Returns HTTP 200 when the orchestrator is ready to serve requests, or
    HTTP 503 while it is still initializing.

    Returns:
        JSONResponse with ``{"status": "ok"}`` on 200 or
        ``{"status": "starting"}`` on 503.

    """
    if orchestrator.is_ready:
        return JSONResponse({"status": "ok"})
    return JSONResponse({"status": "starting"}, status_code=503)


@router.get("/customers")
async def list_customers() -> list[dict[str, Any]]:
    """List every guest available for the UI's identity dropdown.

    Returns:
        list[dict[str, Any]]: One `{customer_id, first_name, last_name}` per
        guest.

    """
    resources = orchestrator.get_booking_resources()
    seeded_customer_count = (
        load_app_config().load_data.booking_pgsql.seeded_customer_count
    )
    customers = await write_ops.list_customers(
        resources.get_write_pool(), seeded_customer_count=seeded_customer_count,
    )
    return [
        {
            "customer_id": c.customer_id,
            "first_name": c.first_name,
            "last_name": c.last_name,
        }
        for c in customers
    ]


@router.get("/bookings")
async def list_bookings(customer_id: int) -> dict[str, Any]:
    """List one guest's bookings for the UI's reservations panel.

    Args:
        customer_id: Guest whose bookings to return.

    Returns:
        dict[str, Any]: `{"bookings": [...]}`, most recent first.

    Note:
        Unauthenticated: any `customer_id` 1-25 can be queried. Acceptable
        for a demo where guests are a dropdown rather than real accounts,
        but real accounts would need this gated.

    """
    resources = orchestrator.get_booking_resources()
    bookings = await write_ops.list_bookings(
        resources.get_write_pool(), customer_id=customer_id,
    )
    return {"bookings": [write_ops.serialize_booking(b) for b in bookings]}


@router.post("/chat")
async def chat(payload: ChatPayload, request: Request) -> Response:
    """Send a user message to the agent, as JSON or as an SSE stream.

    Content negotiation replaces the old two-endpoint ``/v1/chat`` +
    ``/v1/chat/stream`` split: ``Accept: text/event-stream`` streams stage,
    proposal, and error events followed by a JSON response containing the
    same fields the non-streaming path returns.

    Args:
        payload: Chat payload containing `thread_id`, `customer_id`, and `text`.
        request: Incoming request, inspected for the `Accept` header.

    Returns:
        StreamingResponse with ``Content-Type: text/event-stream`` if
        requested, otherwise a JSONResponse.

    Raises:
        HTTPException: 409 if `thread_id` is already bound to a different
            `customer_id`.

    """
    if _SSE_MEDIA_TYPE in request.headers.get("accept", ""):
        return StreamingResponse(_event_stream(payload), media_type=_SSE_MEDIA_TYPE)

    try:
        result = await orchestrator.ainvoke(
            thread_id=payload.thread_id,
            user_text=payload.text,
            customer_id=payload.customer_id,
        )
    except ThreadCustomerMismatchError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    response = format_chat_response(result)
    proposal = orchestrator.get_booking_resources().proposals.get_pending_for_thread(
        payload.thread_id,
    )
    if proposal is not None:
        response["proposal"] = _proposal_event(proposal)
    return JSONResponse(response)


async def _event_stream(payload: ChatPayload) -> AsyncGenerator[str]:
    r"""Yield SSE-formatted strings for one chat turn, with keepalives.

    A background task drives `orchestrator.ainvoke_stream` and pushes each
    event onto a queue; this generator only ever waits on the queue, so a
    slow turn produces periodic ``: keepalive`` comment lines instead of
    letting a reverse proxy's idle timeout kill an otherwise healthy
    request. Any exception raised while streaming -- including a
    thread/customer mismatch -- is translated into a single ``error`` event
    instead of silently severing the connection.

    Args:
        payload: Chat payload containing `thread_id`, `customer_id`, and `text`.

    Yields:
        SSE strings of the form ``data: {json}\n\n``, or ``: keepalive\n\n``
        comment lines while waiting.

    """
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    async def _pump() -> None:
        """Drive the orchestrator's stream and forward events onto the queue."""
        try:
            async for event in orchestrator.ainvoke_stream(
                thread_id=payload.thread_id,
                user_text=payload.text,
                customer_id=payload.customer_id,
            ):
                await queue.put(event)
        except Exception as exc:
            logger.warning("chat stream failed: %s", exc, exc_info=True)
            await queue.put({"type": "error", "message": _error_message(exc)})
        finally:
            await queue.put(None)

    pump_task = asyncio.create_task(_pump())
    try:
        while True:
            try:
                item = await asyncio.wait_for(
                    queue.get(), timeout=_KEEPALIVE_INTERVAL_S,
                )
            except TimeoutError:
                yield ": keepalive\n\n"
                continue
            if item is None:
                return
            yield f"data: {json.dumps(item)}\n\n"
    finally:
        if not pump_task.done():
            pump_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pump_task


@router.post("/booking/confirm")
async def confirm_booking(payload: ProposalActionPayload) -> dict[str, Any]:
    """Confirm a pending proposal, committing it through `write_ops`.

    This is the only caller of the booking write functions: the model can
    only propose, and only a human clicking Confirm reaches this endpoint.

    Args:
        payload: Proposal id and the guest confirming it.

    Returns:
        dict[str, Any]: `status`, `already_confirmed`, an app-authored
        `message`, and the type-specific result fields.

    Raises:
        HTTPException: 404 if the proposal is unknown or expired; 403 if it
            belongs to a different guest; 409 if the underlying write fails
            (e.g. the nights were taken in the meantime).

    """
    resources = orchestrator.get_booking_resources()
    try:
        outcome = await resources.proposals.confirm(
            proposal_id=payload.proposal_id,
            customer_id=payload.customer_id,
            write_pool=resources.get_write_pool(),
        )
    except proposals_module.ProposalNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except proposals_module.ProposalOwnershipError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except write_ops.BookingWriteError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    message = receipt_message(outcome)
    await orchestrator.append_assistant_message(
        thread_id=outcome.proposal.thread_id, text=message,
    )
    return {
        "status": "confirmed",
        "already_confirmed": outcome.already_confirmed,
        "message": message,
        **serialize_write_result(outcome.result),
    }


@router.post("/booking/dismiss")
async def dismiss_booking(payload: ProposalActionPayload) -> dict[str, Any]:
    """Dismiss a pending proposal without writing anything.

    Args:
        payload: Proposal id and the guest dismissing it.

    Returns:
        dict[str, Any]: `{"status": "dismissed"}`.

    Raises:
        HTTPException: 404 if the proposal is unknown, expired, or already
            used; 403 if it belongs to a different guest.

    """
    resources = orchestrator.get_booking_resources()
    try:
        proposal = resources.proposals.dismiss(
            proposal_id=payload.proposal_id, customer_id=payload.customer_id,
        )
    except proposals_module.ProposalNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except proposals_module.ProposalOwnershipError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    await orchestrator.append_assistant_message(
        thread_id=proposal.thread_id, text="No changes were made.",
    )
    return {"status": "dismissed"}


app.include_router(router)
