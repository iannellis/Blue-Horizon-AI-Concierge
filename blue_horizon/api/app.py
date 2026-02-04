"""The FastAPI API that provides in interface to the agent.

Has one main function: chat, which sends a query to the agent and its response to the
user.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from blue_horizon.agents.orchestration import OrchestrationManager

orchestrator = OrchestrationManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the lifespan of the FastAPI app.

    Starts the agent orchestrator when the app is launched and stops it when the app
    is shutdown.

    Arguments:
        app: The FastAPI application

    """
    await orchestrator.start()
    yield
    await orchestrator.stop()

app = FastAPI(lifespan=lifespan)

@app.post("/chat")
async def chat(payload: dict):
    """Pass the payload to the agent.

    Arguments:
        payload: A dict containing the thread_id identifying the conversation
            and the text of the user's query.

    """
    return await orchestrator.ainvoke(thread_id=payload["thread_id"],
                                      user_text=payload["text"])
