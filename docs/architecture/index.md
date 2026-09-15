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

## Logging

Both processes log to stderr, which `supervisord` forwards to the container log. The
API's logging is set up in its `lifespan` by `blue_horizon/logging_setup.py`, at the
level in `[logging]` (see [Configuration](../guides/configuration.md)). Every line
carries a timestamp, level, logger name, and the conversation's `thread_id` and
`customer_id`. The API binds those two values when a chat turn starts, and every task
LangGraph runs for the turn inherits them, so a line logged inside a node or tool names
its guest and conversation. A line logged outside any turn shows `-` for both.

The container log is lost whenever the Space restarts, so when `AXIOM_API_KEY` and
`AXIOM_DATASET` are set, the API also ships every record to that Axiom dataset over
OTLP, using the OpenTelemetry SDK. Each record carries its message as the body, its
level as the severity, `thread_id` and `customer_id` as attributes when they are bound,
the source file, function, and line, any traceback, and `service.name` set to
`blue-horizon-api`. The handler on the root logger only converts the record and queues
it; the SDK's batch processor sends from its own thread, so logging never waits on the
network. A batch that still fails after the exporter's retries is dropped and reported
on stderr, and the exporter's own log lines are never shipped. Shutdown sends what is
still queued.

The log is written so that a turn can be audited without a LangSmith trace. At `INFO`
it records:

| Event | Logged by |
|---|---|
| Router decision and dispatch | `orchestration/factory.py` |
| The turn's route, outcome, duration, LLM semaphore wait, and token counts | `orchestration/manager.py` |
| Retrieval counts per source | `information/factory.py` |
| Every `run_sql` statement, with its row count or error and its duration | `booking/resources.py` |
| A `propose_*` tool's refusal, with the `booking_id` the model named | `booking/factory.py` |
| Every proposal created, confirmed, dismissed, refused, or not found | `booking/proposals.py` |

Each proposal line names the proposal, action, thread, guest, and dialog total, plus the
`booking_id` and confirmation number once written, so it stands on its own. At
`WARNING` it records a guest refused another guest's proposal or thread, a write blocked
by the read-only role, and a commit that could not reach the database.

Lines that measure something also carry the numbers as record attributes, so Axiom can
chart and aggregate them without parsing text. The turn line carries `route`, `outcome`,
`duration_ms`, `semaphore_wait_ms`, `input_tokens`, and `output_tokens`. The manager
logs it rather than the graph, since only the manager sees how long the turn waited
for one of the `llm_concurrency` slots before its first node ran. A wait that grows
while model time does not means turns are queueing. Tokens are counted by a
`UsageMetadataCallbackHandler` on the turn's config, which reaches every chat model
call in the turn; embedding calls are not counted. Every `run_sql` line for a statement
that passed the guardrail carries `duration_ms`, spanning its retries, so a cold
database start shows there. A confirm that attempted a write carries `action` and
`duration_ms`; a replayed confirm writes nothing and carries neither.

Guest messages, model replies, and `run_sql` result rows are never logged.

The UI logs at `INFO` under the `ui.app` logger: failed or timed-out chat requests,
failures to reach the API, and unexpected confirm statuses. Each line names the
`thread_id` and `customer_id` it concerns, where it has them.

Given the same two variables, the UI ships its records to the same dataset in the same
way, with `service.name` set to `blue-horizon-ui`, so a query can filter by process.
Because the UI imports no `blue_horizon` code, it does not use `logging_setup.py`: it
builds its own handler and reads the endpoint and batching settings from
`app_config.toml`'s `[logging.axiom]` section as data. A UI record carries `thread_id`
and `customer_id` as attributes under the same names as the API's, passed explicitly on
the logging call rather than bound per turn, so one query on either finds both
processes' lines. The provider's exit hook sends what is still queued when Streamlit exits.
