"""The FastAPI API that provides in interface to the agent.

Has one main function: chat, which sends a query to the agent and its response to the
user.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from blue_horizon.agents.orchestration import OrchestrationManager, format_chat_response
from blue_horizon.config import load_app_config
from blue_horizon.neon import reset_branch

load_dotenv()


class ChatPayload(BaseModel):
    """The payload for the chat endpoint.

    Attributes:
        thread_id: The unique identifier for the conversation thread.
        text: The user's query text.

    """

    thread_id: str
    text: str


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


@router.post("/chat")
async def chat(payload: ChatPayload) -> dict[str, Any]:
    """Pass the payload to the agent.

    Arguments:
        payload: The chat payload containing the thread_id identifying the conversation
            and the text of the user's query.

    Returns:
        Response dict from the orchestrator containing the agent's response.

    """
    result = await orchestrator.ainvoke(
        thread_id=payload.thread_id,
        user_text=payload.text,
    )
    return format_chat_response(result)


@router.post("/reset")
async def reset() -> JSONResponse:
    """Reset the working Neon branch to its parent baseline.

    Restores the configured Neon branch (typically ``"Working"``) to the
    state of its parent branch (typically ``"production"``), effectively
    clearing all user-created bookings.

    Returns:
        JSONResponse with ``{"status": "ok"}`` on success.

    Raises:
        HTTPException: 503 if Neon credentials are not configured; 500 if
            the Neon API call fails.

    """
    cfg = load_app_config()
    if not cfg.neon_api_key or not cfg.neon.project_id:
        raise HTTPException(status_code=503, detail="Reset not configured.")
    try:
        await reset_branch(
            cfg.neon.project_id,
            cfg.neon.branch_name,
            cfg.neon_api_key,
            lock_retry_attempts=cfg.neon.lock_retry_attempts,
            lock_retry_delay_s=cfg.neon.lock_retry_delay_s,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return JSONResponse({"status": "ok"})


app.include_router(router)
