"""The FastAPI API that provides in interface to the agent.

Has one main function: chat, which sends a query to the agent and its response to the
user.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from blue_horizon.agents.orchestration import OrchestrationManager


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
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:  # noqa: ARG001
    """Manage the lifespan of the FastAPI app.

    Starts the agent orchestrator when the app is launched and stops it when the app
    is shutdown.

    Arguments:
        app: The FastAPI application (required by FastAPI lifespan signature)

    Yields:
        None

    """
    await orchestrator.start()
    yield
    await orchestrator.stop()

app = FastAPI(lifespan=lifespan)

@app.post("/chat")
async def chat(payload: ChatPayload) -> dict[str, Any]:
    """Pass the payload to the agent.

    Arguments:
        payload: The chat payload containing the thread_id identifying the conversation
            and the text of the user's query.

    Returns:
        A dictionary containing the agent's response.

    """
    return await orchestrator.ainvoke(
        thread_id=payload.thread_id,
        user_text=payload.text,
    )
