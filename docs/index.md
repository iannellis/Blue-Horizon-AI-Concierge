# Blue Horizon AI Concierge

An AI-powered hotel concierge that handles natural language queries about hotel
information and room bookings.

The **agent backend** is built with [LangGraph](https://github.com/langchain-ai/langgraph)
and served via [FastAPI](https://fastapi.dev). A router LLM classifies each incoming
message and dispatches it to one of two specialised sub-agents: a RAG-based information
agent (Redis vector store plus result merging) or a natural-language-to-SQL
rooms/bookings agent (PostgreSQL). The booking agent only ever searches and *proposes*
- every booking, cancellation, and modification is committed by server-side code once a
guest confirms, never by the model. Conversation history is maintained across turns via
LangGraph's `MemorySaver` checkpointer, keyed by `thread_id`, so multiple concurrent
sessions are fully isolated from one another. The backend streams stage-progress and
proposal events to the client via SSE so the UI can show live status and a confirmation
dialog while the agent works.

The **chat UI** is built with [Streamlit](https://streamlit.io). It connects to the
FastAPI backend and displays a stage-aware progress indicator inside the assistant
bubble as each request is processed, then renders the final response.

## Where to start

<div class="grid cards" markdown>

- **New to the project?**

    Read the [Architecture overview](architecture/index.md) for how the pieces fit
    together, then [Design Goals and Decisions](design-decisions.md) for why they fit
    together that way.

- **Want to run it?**

    [Running Locally](guides/running-locally.md) covers prerequisites, data loading,
    and starting the API and UI. [Configuration](guides/configuration.md) documents
    every tunable and environment variable.

- **Want to integrate with it?**

    The [API Reference](api.md) documents every endpoint, the streaming event
    protocol, and the propose/confirm booking contract.

- **Interested in how it is measured?**

    [Evaluation](evaluation/index.md) has the metric results; the
    [Harness](evaluation/harness.md) and [Stress Test](evaluation/stress-test.md)
    pages explain how those numbers are produced.

</div>

## Live demo

A deployed instance runs on HuggingFace Spaces. Access is controlled via Google OAuth.
Recruiters and hiring managers can request access to be added to the allowed-users list.

Once access is granted, the agent is available at
**[iellis02-blue-horizon.hf.space](https://iellis02-blue-horizon.hf.space)** (this exact
URL must be used; Google login will not work from any other origin).

If the Space has gone to sleep and returns a 500 error, it can be restarted from the
Space page at [huggingface.co/spaces/iellis02/blue-horizon](https://huggingface.co/spaces/iellis02/blue-horizon).
