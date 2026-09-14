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
from blue_horizon.agents.orchestration import (
    OrchestrationManager,
    Readiness,
    format_chat_response,
)
from blue_horizon.config import load_app_config

load_dotenv()

logger = logging.getLogger(__name__)

_SSE_MEDIA_TYPE = "text/event-stream"
_KEEPALIVE_INTERVAL_S = 15.0
# How long a client should wait before retrying a confirm that failed with
# BookingUnavailableError. Deliberately short: the dominant cause is a Neon
# compute resume, normally a few hundred milliseconds, not a lengthy outage.
_CONFIRM_RETRY_AFTER_S = 5


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


# Maps an exception type this module can specifically recognise to the
# additive `code` on a mid-turn `error` SSE event raised by the pump itself.
# Anything else falls back to "internal" in `_error_event`. A turn that
# failed inside the graph (a router or sub-agent timeout or exception, or an
# empty turn) never reaches this map: the orchestrator yields its own
# `error` event carrying `"timeout"` or `"internal"`, which the pump forwards
# unchanged. `"unavailable"` and `"failed"` are not reachable mid-stream at
# all, since an unready agent is stopped by the readiness gate before
# _event_stream ever starts.
_MID_TURN_ERROR_CODE_BY_EXCEPTION: dict[type[Exception], str] = {
    ThreadCustomerMismatchError: "thread_mismatch",
}


def _error_event(exc: Exception) -> dict[str, Any]:
    """Build a mid-turn SSE `error` event for a chat-stream exception.

    Unlike the pre-stream readiness gate (`_not_ready_response`), this
    cannot be an HTTP status: the 200 and the SSE headers are already
    committed by the time a turn in progress can fail. The same kind of
    signal is instead carried as an additive `code` field on the event body,
    so a client that only reads `message` still gets identical behavior to
    before this field existed.

    Args:
        exc: Exception raised while streaming a chat turn.

    Returns:
        dict[str, Any]: `{"type": "error", "message": str, "code": str}`.

    """
    return {
        "type": "error",
        "message": _error_message(exc),
        "code": _MID_TURN_ERROR_CODE_BY_EXCEPTION.get(type(exc), "internal"),
    }


def _chat_retry_after_s() -> int:
    """Return the `Retry-After` seconds for a `/v1/chat` 503 while not ready.

    Returns:
        int: The configured
        `[orchestration.orchestration].unavailable_retry_after_s`, shared
        across the STARTING and FAILED readiness states -- the init loop
        keeps retrying in both, so there is no basis yet for a longer
        interval in the FAILED case.

    """
    return int(
        load_app_config().orchestration.orchestration.unavailable_retry_after_s,
    )


def _not_ready_response() -> JSONResponse:
    """Build the 503 response for a `/v1/chat` request while not ready.

    Used for both content-negotiated branches: an SSE client gets this same
    plain JSON body instead of a stream, since there is nothing to stream
    yet and `raise_for_status()` on the client side does not care which
    content type came with the 503.

    Returns:
        JSONResponse: 503, a `Retry-After` header, and a body of
        ``{"status": "starting" | "failed", "message": str, "retry_after_s":
        int}``.

    """
    status = "failed" if orchestrator.readiness is Readiness.FAILED else "starting"
    retry_after_s = _chat_retry_after_s()
    body = {
        "status": status,
        "message": orchestrator.get_readiness_message(),
        "retry_after_s": retry_after_s,
    }
    return JSONResponse(
        body,
        status_code=503,
        headers={"Retry-After": str(retry_after_s)},
    )


@router.get("/health")
async def health() -> JSONResponse:
    """Return the readiness status of the orchestrator.

    Returns HTTP 200 when the orchestrator is ready to serve requests, or
    HTTP 503 while it is still initializing or permanently failed -- either
    way, the init loop keeps retrying in the background (see
    `OrchestrationManager.readiness`), so this can only ever report today's
    snapshot, not a promise about whether it will change.

    Returns:
        JSONResponse with ``{"status": "ok"}`` on 200, or
        ``{"status": "starting" | "failed"}`` on 503.

    """
    if orchestrator.is_ready:
        return JSONResponse({"status": "ok"})
    status = "failed" if orchestrator.readiness is Readiness.FAILED else "starting"
    return JSONResponse({"status": status}, status_code=503)


@router.get("/customers")
async def list_customers() -> list[dict[str, Any]]:
    """List every guest available for the UI's automated guest assignment.

    Returns:
        list[dict[str, Any]]: One `{customer_id, first_name, last_name}` per
        guest.

    Raises:
        HTTPException: 503 if the booking database is not yet initialized
            (the startup window), with a `Retry-After` header.

    """
    resources = orchestrator.get_booking_resources()
    try:
        write_pool = resources.get_write_pool()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=orchestrator.get_readiness_message(),
            headers={"Retry-After": str(_chat_retry_after_s())},
        ) from exc
    seeded_customer_count = (
        load_app_config().load_data.booking_pgsql.seeded_customer_count
    )
    customers = await write_ops.list_customers(
        write_pool, seeded_customer_count=seeded_customer_count,
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

    Raises:
        HTTPException: 503 if the booking database is not yet initialized
            (the startup window), with a `Retry-After` header.

    Note:
        Unauthenticated: any seeded `customer_id` (currently 1-15, see
        `seeded_customer_count`) can be queried. Acceptable for a demo where
        guests are assigned automatically rather than real accounts, but
        real accounts would need this gated.

    """
    resources = orchestrator.get_booking_resources()
    try:
        write_pool = resources.get_write_pool()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=orchestrator.get_readiness_message(),
            headers={"Retry-After": str(_chat_retry_after_s())},
        ) from exc
    bookings = await write_ops.list_bookings(write_pool, customer_id=customer_id)
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
        requested, otherwise a JSONResponse. Either becomes a 503 JSON
        response with a `Retry-After` header instead, if the orchestrator is
        not ready -- checked here, before either branch commits to a
        response, since that is the one point where 503 is still an option
        for the streaming branch too (a `StreamingResponse` commits its 200
        as soon as it starts). On the JSON branch, a turn that failed inside
        the graph returns `_failed_turn_response` instead: 504 for a timeout,
        502 otherwise.

    Raises:
        HTTPException: 409 if `thread_id` is already bound to a different
            `customer_id`.

    """
    if not orchestrator.is_ready:
        return _not_ready_response()

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

    turn_error = result.get("turn_error")
    if turn_error is not None:
        return _failed_turn_response(turn_error)

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
            await queue.put(_error_event(exc))
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


def _failed_turn_response(turn_error: str) -> JSONResponse:
    """Build the JSON-branch response for a chat turn that failed in the graph.

    Unlike the SSE branch, nothing is committed yet when the JSON branch
    learns the outcome, so the failure can be a real HTTP status rather than
    a `200` carrying an apology the client cannot tell apart from a reply.

    Args:
        turn_error: The orchestrator's failure code, `"timeout"` or
            `"internal"`.

    Returns:
        JSONResponse: 504 for `"timeout"`, 502 otherwise, with a body of
        ``{"code": str, "message": str}`` matching the SSE `error` event.

    """
    status_code = 504 if turn_error == "timeout" else 502
    return JSONResponse(
        {
            "code": turn_error,
            "message": load_app_config().orchestration.messages.error,
        },
        status_code=status_code,
    )


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
            (e.g. the nights were taken in the meantime); 503 with a
            `Retry-After` header if the database could not be reached at
            all, in which case the proposal was left pending and confirming
            again is safe.

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
    except write_ops.BookingUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
            headers={"Retry-After": str(_CONFIRM_RETRY_AFTER_S)},
        ) from exc
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
