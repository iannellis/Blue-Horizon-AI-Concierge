# Architecture overview

The system is composed of an **orchestration layer** that routes between two agents:

- **[Information agent](information-agent.md)** answers questions about hotel services,
  amenities, and policies using RAG over a Redis vector store.
- **[Booking agent](booking-agent.md)** searches availability using LLM-generated,
  read-only SQL against a PostgreSQL database, and proposes bookings, cancellations,
  and modifications for the guest to confirm. It never writes to the database itself.

Both sit behind the [orchestration graph](orchestration.md), which classifies each
incoming message, dispatches it, and maintains conversation history.

## Runtime topology

| Component | Technology | Role |
|---|---|---|
| Agent backend | LangGraph + FastAPI | Routing, RAG, NL-to-SQL, proposal lifecycle |
| Chat UI | Streamlit | Guest session, stage indicator, confirmation dialog |
| Vector store | Redis (via LlamaIndex) | FAQ, amenities, and services indices |
| Relational store | PostgreSQL on [Neon](https://neon.tech) | Rooms, availability, customers, bookings |
| Embeddings and chat | OpenAI | `text-embedding-3-small`, `gpt-5.6-luna` |
| Judge and RAG scoring | Google Gemini | Evaluation only, never in the serving path |

Both processes run in a single container under `supervisord` in the deployed
configuration. See [Deployment and CI](../guides/deployment.md).

## Repository structure

```
blue_horizon/          # Main application package
  agents/
    information/       # RAG-based information agent
    booking/           # Read-only NL-to-SQL search + server-owned booking writes
      factory.py         # Agent/tool construction: run_sql (read-only) + propose_* tools
      write_ops.py       # commit_booking / cancel_booking / modify_booking -- the only writers
      proposals.py       # In-process ProposalStore: propose -> confirm/dismiss lifecycle
      receipts.py        # App-authored confirmation/cancellation/modification receipt text
      guardrails.py      # SQL AST allowlist (redundant with the read-only DB role, belt-and-braces)
    orchestration/     # LangGraph router and manager
  api/
    app.py             # FastAPI application
  load_data/
    information_redis.py  # Loads FAQ/services/amenities into Redis
    booking_pgsql.py      # Loads room/availability/customer/booking data and rebuilds
                          # booking tables in PostgreSQL
  system_prompts/      # System prompt templates (.txt)
  config.py            # Pydantic configuration models
  neon.py              # Neon branch reset utility

ui/
  app.py               # Streamlit chat UI: guest session, proposal dialog, reservations panel

deploy/                # Docker/deployment configuration
  supervisord.conf     # Runs FastAPI + Streamlit under supervisord
  generate_secrets.py  # Writes Streamlit secrets from HF Space env vars
  requirements.txt     # Pinned production dependencies

eval/                  # Evaluation framework

notebooks/             # Development and exploration notebooks
  eda.ipynb            # Exploratory data analysis
  information_agent.ipynb
  booking_agent.ipynb
  full_agent.ipynb     # End-to-end agent walkthrough
  orchestration.ipynb
  neonsql.ipynb        # Basic setup of SQL database and natural language querying

tests/                 # Pytest test suite
data/                  # Source data (CSV and pickled DataFrames)
docs/                  # This documentation site
```

## Request lifecycle

1. The UI posts to `POST /v1/chat` with a `thread_id`, a `customer_id`, and the message
   text, requesting SSE via `Accept: text/event-stream`.
2. The orchestration graph loads the thread's history from the `MemorySaver`
   checkpointer and runs the router LLM.
3. The router emits `info`, `booking`, or `refuse`. Refusals short-circuit with a
   configured message; the other two dispatch to the matching sub-agent.
4. Stage events stream to the UI as the sub-agent works.
5. If the booking agent calls a `propose_*` tool, a `proposal` event carries the stored
   proposal fields to the UI, which renders a confirmation dialog from those fields
   alone, never from the assistant's text.
6. A guest clicking **Confirm** calls `POST /v1/booking/confirm`, the only path that
   ever writes a booking. The application, not the model, authors the receipt.

See the [API Reference](../api.md) for the wire-level detail.
