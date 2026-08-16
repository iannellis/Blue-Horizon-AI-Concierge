---
title: Blue Horizon AI Concierge
emoji: 🏨
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
startup_duration_timeout: 1h
---

# Blue Horizon AI Concierge

An AI-powered hotel concierge that handles natural language queries about hotel information and room bookings.

The **agent backend** is built with [LangGraph](https://github.com/langchain-ai/langgraph) and served via [FastAPI](https://fastapi.dev). A router LLM classifies each incoming message and dispatches it to one of two specialised sub-agents: a RAG-based information agent (Redis vector store + result merging) or a natural-language-to-SQL rooms/bookings agent (PostgreSQL). The booking agent only ever searches and *proposes* — every booking, cancellation, and modification is committed by server-side code once a guest confirms, never by the model. Conversation history is maintained across turns via LangGraph's `MemorySaver` checkpointer, keyed by `thread_id`, so multiple concurrent sessions are fully isolated from one another. The backend streams stage-progress and proposal events to the client via SSE so the UI can show live status and a confirmation dialog while the agent works.

The **chat UI** is built with [Streamlit](https://streamlit.io). It connects to the FastAPI backend and displays a stage-aware progress indicator inside the assistant bubble as each request is processed, then renders the final response.

Documentation is available in `documentation.pdf` (to be written).

---

## Live Demo

A deployed instance runs on HuggingFace Spaces. Access is controlled via Google OAuth. Recruiters and hiring managers can request access to be added to the allowed-users list.

Once access is granted, the agent is available at **[iellis02-blue-horizon.hf.space](https://iellis02-blue-horizon.hf.space)** (this exact URL must be used; Google login will not work from any other origin).

If the Space has gone to sleep and returns a 500 error, it can be restarted from the Space page at [huggingface.co/spaces/iellis02/blue-horizon](https://huggingface.co/spaces/iellis02/blue-horizon).

---

## Architecture

The system is composed of an **orchestration layer** that routes between two agents:

- **Information agent** — answers questions about hotel services, amenities, and policies using RAG over a Redis vector store.
- **Booking agent** — searches availability using LLM-generated, read-only SQL against a PostgreSQL database, and proposes bookings/cancellations/modifications for the guest to confirm. It never writes to the database itself.

### Orchestration

![Orchestration diagram](orchestration.png)

The orchestration layer is a LangGraph state graph. A router LLM classifies each incoming message and dispatches it to the appropriate sub-agent (or refuses it if out of scope). Conversation history is maintained in memory via LangGraph's `MemorySaver` checkpointer, keyed by `thread_id`.

### Information Agent

![Information agent diagram](information_agent.png)

The information agent follows a RAG pipeline:

1. A **parse** node extracts one or more sub-queries from the user message, along with any constraints (price bounds, duration bounds, booking requirements, and notice time).
2. Three **retrieval** nodes run in parallel against separate Redis vector indices (FAQ, amenities, services).
3. A **merge** node merges and deduplicates retrieved context.
4. A **respond** node generates the final answer.

Embeddings are produced with OpenAI and stored in Redis via LlamaIndex.

### Booking SQL Agent

The booking agent searches availability via LLM-generated, read-only SQL against a PostgreSQL database (hosted on [Neon](https://neon.tech)) — but it never writes. Two database roles enforce this at the database level, not just in application code: `bh_agent_ro` (`SELECT` only on `rooms`/`room_availability`, used exclusively by the model's `run_sql` tool) and `bh_agent_rw` (used only by the server-side write functions below). SQL guardrails add a redundant table allowlist and cap query results at 50 rows before passing them to the LLM, limiting context size and cost — but the real guarantee is the grant, not the guardrail.

**The model proposes, the application decides.** To book, cancel, or modify, the agent calls a `propose_*` tool that prices and validates the request and stores it — nothing is written to the database yet, and the tool returns only a proposal id, never success text. The UI renders a confirmation dialog built strictly from the *proposal's own stored fields* (never from the model's chat text), and only a guest clicking **Confirm** calls `POST /v1/booking/confirm`, the sole caller of the `write_ops` commit functions (`commit_booking` / `cancel_booking` / `modify_booking`). A successful commit returns a confirmation number that the *application* renders as a new chat message — the model is instructed to never claim a booking succeeded and never invent a confirmation number, so a success claim with no receipt behind it is something the eval suite explicitly checks for (`booking_no_unbacked_success_claims`).

Guests are identified by picking from a dropdown of the first 25 seeded customers (there is no real login for this demo). A `thread_id` is bound to whichever `customer_id` first used it, and the API rejects a mismatched replay of that `thread_id` under a different guest. `GET /v1/bookings` and `POST /v1/reset` are unauthenticated, like the rest of this demo's guest-selection model — acceptable here since guests are a dropdown rather than real accounts, but worth stating plainly rather than leaving a reader to discover it.

---

## Repository Structure

```
blue_horizon/          # Main application package
  agents/
    information/       # RAG-based information agent
    booking/           # Read-only NL-to-SQL search + server-owned booking writes
      factory.py         # Agent/tool construction: run_sql (read-only) + propose_* tools
      write_ops.py        # commit_booking / cancel_booking / modify_booking -- the only writers
      proposals.py         # In-process ProposalStore: propose -> confirm/dismiss lifecycle
      receipts.py           # App-authored confirmation/cancellation/modification receipt text
      guardrails.py        # SQL AST allowlist (redundant with the read-only DB role, belt-and-braces)
    orchestration/      # LangGraph router and manager
  api/
    app.py             # FastAPI application
  load_data/
    information_redis.py  # Loads FAQ/services/amenities into Redis
    booking_pgsql.py      # Loads room/availability/customers data and rebuilds booking tables in PostgreSQL
  system_prompts/      # System prompt templates (.txt)
  config.py            # Pydantic configuration models
  neon.py              # Neon branch reset utility

ui/
  app.py               # Streamlit chat UI: guest picker, proposal dialog, reservations panel

deploy/                # Docker/deployment configuration
  supervisord.conf     # Runs FastAPI + Streamlit under supervisord
  generate_secrets.py  # Writes Streamlit secrets from HF Space env vars
  requirements.txt     # Pinned production dependencies

eval/                  # Evaluation framework (see eval/README.md)

notebooks/             # Development and exploration notebooks
  eda.ipynb            # Exploratory data analysis
  information_agent.ipynb
  booking_agent.ipynb
  full_agent.ipynb     # End-to-end agent walkthrough
  orchestration.ipynb
  neonsql.ipynb        # Basic setup of SQL database and natural language querying

tests/                 # Pytest test suite
data/                  # Source data (CSV and pickled DataFrames)
```

---

## API

The FastAPI backend runs on port `8000` and exposes the following endpoints under the `/v1` prefix.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/v1/health` | Returns `{"status": "ok"}` (200) when ready, `{"status": "starting"}` (503) during init |
| `GET` | `/v1/customers` | Lists the 25 seeded guests as `{customer_id, first_name, last_name}`, for the identity dropdown |
| `GET` | `/v1/bookings?customer_id=` | Lists one guest's reservations (confirmation number, rooms, dates, total, status) |
| `POST` | `/v1/chat` | Send a message. Content-negotiated: `Accept: text/event-stream` streams SSE events, anything else returns one JSON response |
| `POST` | `/v1/booking/confirm` | Commit a pending proposal — the only path that ever writes a booking, cancellation, or modification |
| `POST` | `/v1/booking/dismiss` | Discard a pending proposal without writing anything |
| `POST` | `/v1/reset` | Reset the working Neon database branch to its parent baseline; also clears the proposal store, the thread↔customer registry, and the conversation checkpointer |

`/v1/chat/stream` from earlier versions has been retired in favor of content negotiation on `/v1/chat`.

### Chat request body

```json
{
  "thread_id": "<uuid>",
  "customer_id": 7,
  "text": "Do you have any rooms available this weekend?"
}
```

`thread_id` is bound to whichever `customer_id` first uses it; a later request reusing that `thread_id` under a different `customer_id` is rejected with `409 Conflict`.

### Streaming events

Requesting `/v1/chat` with `Accept: text/event-stream` emits `text/event-stream` (SSE) events, plus periodic `: keepalive` comment lines so a slow turn doesn't trip a reverse proxy's idle timeout:

```
data: {"type": "stage", "label": "Routing your request…"}
data: {"type": "stage", "label": "Processing your room request…"}
data: {"type": "proposal", "proposal_id": "...", "action": "book", "summary": {...}}
data: {"type": "done", "response": "I've put together a request for you to review."}
```

A `proposal` event appears after a `propose_*` tool call, carrying the same fields the confirmation dialog renders — `summary` is action-shaped (`book`/`cancel`/`modify`), never derived from the assistant's own text. Any exception mid-stream (including a `thread_id`/`customer_id` mismatch) is translated into `{"type": "error", "message": "..."}` rather than silently severing the connection. The non-streaming JSON response carries the same `proposal` field when a proposal is pending after the turn.

---

## Configuration

All tunable parameters live in [`blue_horizon/app_config.toml`](blue_horizon/app_config.toml). The file is structured into sections that map to the Pydantic models in [`blue_horizon/config.py`](blue_horizon/config.py):

| Section | Controls |
|---------|----------|
| `[orchestration]` | Router LLM, timeouts, retry backoff, concurrency limit |
| `[info]` | Information agent LLM, embeddings model, Redis tuning, retrieval `top_k` |
| `[booking]` | Booking agent LLM, SQL guardrails, DB pool settings |
| `[booking.proposals]` | `ttl_s` — how long an unconfirmed proposal stays in the in-process store before it's purged (reserves no inventory, so this only bounds store size) |
| `[load_data]` | Paths to the source data pickles |
| `[neon]` | Neon project ID and branch name for the `/v1/reset` endpoint |

Runtime secrets are read from a `.env` file (or environment variables). Both the API and the UI call `load_dotenv()` at startup, so a single `.env` file in the project root covers both processes when running locally.

**Required:**

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |
| `REDIS_URL` | Redis connection URL |
| `PGSQL_RW_DB_URL` | PostgreSQL connection URL, authenticated as the read-write `bh_agent_rw` role — backs `write_ops` (booking commits), `/v1/customers`, and `/v1/bookings`. Never reachable from the model. |
| `PGSQL_RO_DB_URL` | PostgreSQL connection URL, authenticated as the read-only `bh_agent_ro` role — the *only* database connection the model's `run_sql` tool can use; Postgres itself refuses any write attempted through it |

`bh_agent_rw` and `bh_agent_ro` are not created automatically — they must already exist in the database with passwords set before these URLs will work. See [Load data](#1-load-data) for the one-time grant step.

**Optional:**

| Variable | Description |
|----------|-------------|
| `PGSQL_ROOT_DB_URL` | PostgreSQL connection URL, authenticated with schema-owner privileges — used only by the data-loading tooling ([`booking_pgsql.py`](blue_horizon/load_data/booking_pgsql.py) and `regrant_booking_agent_role.sql`) to create/reload tables and (re)grant the `bh_agent_rw`/`bh_agent_ro` roles; not read by the API or UI at runtime. Required to run [Load data](#1-load-data), not to start the app afterward. |
| `NEON_API_KEY` | Neon management API key — enables the `/v1/reset` endpoint; omit to disable |
| `LANGSMITH_API_KEY` | LangSmith API key — enables LangSmith tracing |
| `LANGSMITH_TRACING` | Set to `true` to activate tracing (requires `LANGSMITH_API_KEY`) |
| `LANGCHAIN_PROJECT` | LangSmith project name to log traces under |
| `GOOGLE_CLIENT_ID` | Google OAuth client ID — enables the login gate in the UI |
| `GOOGLE_CLIENT_SECRET` | Google OAuth client secret (required when `GOOGLE_CLIENT_ID` is set) |
| `COOKIE_SECRET` | Secret used to sign the Streamlit auth session cookie (required when OAuth is enabled) |

---

## Running Locally

### Prerequisites

- Python 3.13
- A running Redis instance
- A PostgreSQL database (Neon or local)
- Dependencies installed via [Poetry](https://python-poetry.org/):

```bash
poetry install --with ui
```

### 1. Load data

Populate Redis with hotel information:

```bash
python -m blue_horizon.load_data.information_redis
```

Populate PostgreSQL with room, availability, and customer data:

```bash
python -m blue_horizon.load_data.booking_pgsql
```

Then grant the two database roles the agent depends on — `bh_agent_ro` (read-only, used by the model's `run_sql` tool) and `bh_agent_rw` (used only by server-side write functions). The roles themselves are not created by this step: they must already exist in the database with passwords set (`PGSQL_RO_DB_URL` / `PGSQL_RW_DB_URL` authenticate as them) before it runs. `regrant_booking_agent_role.sql` only (re)applies the least-privilege grants to those already-present roles:

```bash
psql "$PGSQL_ROOT_DB_URL" -f blue_horizon/load_data/regrant_booking_agent_role.sql
```

`booking_pgsql.py` drops and recreates tables, so this grant step must run after every reload, and both `PGSQL_RW_DB_URL` and `PGSQL_RO_DB_URL` must point at their respective roles before starting the API — pointing both at the same role silently defeats the read-only guarantee, which is why `startup_check` refuses to start if a trial write through the read-only URL succeeds.

### 2. Start the API

```bash
fastapi run blue_horizon/api/app.py --port 8000
```

### 3. Start the UI

```bash
streamlit run ui/app.py
```

The UI connects to `http://localhost:8000` by default. Override with the `BLUE_HORIZON_API_URL` environment variable.

---

## Docker Deployment

The included `Dockerfile` builds a single image that runs both the FastAPI backend and the Streamlit UI under `supervisord`. This is the configuration used for deployment on [HuggingFace Spaces](https://huggingface.co/spaces).

```bash
docker build -t blue-horizon .
docker run -p 7860:7860 \
  -e OPENAI_API_KEY=... \
  -e REDIS_URL=... \
  -e PGSQL_RW_DB_URL=... \
  -e PGSQL_RO_DB_URL=... \
  blue-horizon
```

The UI is served on port `7860`. The FastAPI backend runs internally on `127.0.0.1:8000` and is not exposed outside the container.

Optional Google OAuth can be enabled by setting `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, and `COOKIE_SECRET` as HuggingFace Space secrets.

---

## Evaluation Metrics

Metrics collected from a structured evaluation run over the 206-case dataset at [`eval/datasets/hotel_agent_eval_206.jsonl`](eval/datasets/hotel_agent_eval_206.jsonl). Every metric that requires a model judge or scorer — all three conversation-level scores plus the four RAG metrics below — is graded by Gemini `gemini-3.5-flash-lite`, configured in [`eval/eval_config_206.toml`](eval/eval_config_206.toml) as both `[judge].model` and `[ragas].llm_model`. **RAG answer relevancy** additionally embeds the question with the Gemini `gemini-embedding-2` model (`[ragas].embedding_model`) to compare it against the query Ragas regenerates from the answer — the only metric in this run that consumes an embedding model rather than pure LLM judging. The remaining per-turn metrics (route accuracy, booking outcomes, info filters/reference checks) are deterministic, code-based checks with no model in the loop. **RAG context precision** uses a hotel-tuned Ragas prompt ([`eval/evaluators/_rag_prompts.py`](eval/evaluators/_rag_prompts.py)) that credits a retrieved chunk as useful when it answers any one clause of a multi-part question rather than requiring it to cover the whole question, and skips turns whose reference is only the "nothing matched" sentinel (`[ragas].no_match_reference`) — a contentless refusal gives the judge nothing to score a chunk's usefulness against.

These numbers are a point-in-time snapshot, not a fixed target: they shift whenever the underlying models, prompts, or agent code change, and are expected to be updated regularly as the project evolves. The pinned regression thresholds in [`eval/baselines/`](eval/baselines/) — not this table — are the source of truth CI actually checks against.

### Conversation-level scores

These metrics are assessed once per multi-turn conversation by an LLM judge (`gemini-3.5-flash-lite`).

| Metric | Score |
|--------|-------|
| Consumer quality (1–5) | 4.98 |
| Grounding (1–5) | 4.98 |
| Injection resistance (1–5) | 4.98 |

**Consumer quality** and **Grounding** assess response helpfulness and factual fidelity to retrieved context. **Injection resistance** scores how reliably the system refuses prompt-injection attempts.

### Per-turn scores

These metrics are assessed at each individual exchange and then averaged across all turns in the dataset. The four RAG metrics are scored by Ragas using `gemini-3.5-flash-lite` as the judge LLM; RAG answer relevancy also uses the `gemini-embedding-2` embedding model. Everything else in this table is a deterministic code check, not a model judgment.

| Metric | Score | Scored by |
|--------|-------|-----------|
| Route accuracy | 100% | code |
| Booking — no unexpected failure rate | 100% | code |
| Booking — tool call success rate | 100% | code |
| RAG faithfulness | 95.4% | `gemini-3.5-flash-lite` |
| RAG context recall | 99.5% | `gemini-3.5-flash-lite` |
| RAG context precision | 91.2% | `gemini-3.5-flash-lite` |
| RAG answer relevancy | 83.2% | `gemini-3.5-flash-lite` + `gemini-embedding-2` |
| Info filters pass rate | 96.3% | code |
| Info reference subset pass rate | 100% | code |

**Route accuracy** measures how often the router correctly dispatches to the information or rooms agent (or refuses); this run had zero misroutes across all 206 multi-turn cases. **Booking — no unexpected failure rate** measures whether a booking/cancellation/modification's success-or-failure outcome matched the case's expectation (e.g. a request for an already-taken night is *expected* to fail; that's not counted against the agent). **Booking — tool call success rate** is stricter: it flags any `propose_*`/`run_sql` call that errored on a turn that was expected to succeed. This ran at 96.2% prior to a fix in [`booking.txt`](blue_horizon/system_prompts/booking.txt) requiring every column in a `rooms`/`room_availability` join to be table-qualified (both tables define `max_occupancy`, so an unqualified reference was ambiguous and errored); this run shows zero tool-call errors. **RAG faithfulness** measures whether generated answers stay within the retrieved context; **RAG context recall** measures whether all relevant context was retrieved. **RAG context precision**'s jump from prior runs reflects the prompt and no-match-skip changes described above, not a retrieval change. **Info filters pass rate** measures whether the information agent correctly honours user-specified constraints (price, duration, booking requirements, etc.) when selecting and presenting results. **Info reference subset pass rate** measures whether the expected source documents (FAQ entries, amenity cards, service cards) appear in the set of documents retrieved for a given query.

### Latency (end-to-end, ms)

| Route | p50 | p95 | p99 |
|-------|-----|-----|-----|
| Refuse | 1,055 | 1,336 | 1,819 |
| Info | 4,565 | 7,574 | 9,922 |
| Booking | 4,830 | 6,926 | 8,680 |

Refuse requests resolve fastest (router only). Info requests go through the full RAG pipeline (parse → parallel retrieval → merge → respond). Booking requests run NL-to-SQL search plus, on a propose call, an in-process pricing pass against `room_availability` — the write itself only happens later, on confirm, and is not included in this per-turn latency.

---

## Stress Test

A concurrent stress test simulated 50 simultaneous sessions with 5 booking operations each (250 operations total, `book`/`modify`/`cancel` weighted 50/25/25) against the booking agent, biased 80% of the time toward a hot subset of 10 room/date targets to force contention. Every operation went through the full propose → auto-confirm path — the harness's stand-in for a human clicking Confirm — so this exercises real commits under load, not just proposals.

| Result | Value |
|--------|-------|
| Operations completed (of 250) | 250 (0 errored) |
| Successful bookings/modifies/cancels | 98 (39.2%) |
| Correctly-refused conflicts | 152 (60.8%) |
| Double-booking violations | 0 |
| Reservations with null status | 0 |
| Expected bookings confirmed in DB | 26 / 26 |

The high conflict rate is expected and by design — 80% of operations target the same 10 hot slots, so most of them are supposed to lose the race. What matters is that they lose it *cleanly*.

A **double-booking violation** is two `booking_rooms` rows for the same room with overlapping date ranges — i.e., `commit_booking`/`modify_booking` failed to enforce mutual exclusion under load. This is now a schema-level impossibility, not just something checked for after the fact: `booking_rooms` carries a GiST exclusion constraint, `booking_rooms_no_overlap` (see `setup_booking_rooms_schema` in [`blue_horizon/load_data/booking_pgsql.py`](blue_horizon/load_data/booking_pgsql.py)), that Postgres enforces on every insert or update, on top of `commit_booking`/`modify_booking`'s own `FOR UPDATE` locking on `room_availability`. If that constraint is ever hit — only possible if `room_availability` and `booking_rooms` have drifted out of sync — `write_ops` catches it and turns it into the same clean `BookingWriteError` refusal a guest sees for any other unavailable night, instead of leaking a raw database error.

The stress test's own violation check queries `booking_rooms` directly, via the shared [`eval/db_invariants.py`](eval/db_invariants.py) helper the eval harness also uses so the two can't drift apart. An earlier version instead grouped `room_availability` by `(room_id, date)` — a tautology, since that table's own `UNIQUE` constraint makes such a group impossible to violate regardless of what the agent did. With the exclusion constraint above in place, this query is now belt-and-suspenders rather than the primary guarantee: a genuine violation is unreachable through normal writes, so it mainly serves as an audit signal if the constraint is ever dropped or bypassed. It was first trusted only after being confirmed to go red against a hand-inserted overlapping pair, once, by hand; that verification is now a permanent regression test in [`tests/booking/test_db_invariants.py`](tests/booking/test_db_invariants.py), which today tests the constraint's own refusal directly, since a hand-inserted overlapping pair can no longer even be inserted — see [its pre-constraint history](https://github.com/iannellis/Blue-Horizon-AI-Concierge/blob/e67fb9b33279219a0d4b6b8fc8a85bbb92733dfa/tests/booking/test_db_invariants.py) for how it worked before.

A **null-status reservation** is a `room_availability` row without a valid status field, indicating a partially-written or corrupted booking. Separately, the harness reconciles every thread's expected final booking against the database — all 26 threads that ended with a successful book/modify had that exact room and date range sitting `Booked` in the database, confirming the agent never reported a success that didn't actually commit.

---

## Evaluation

The `eval/` directory contains a standalone evaluation framework with its own documentation. See [`eval/README.md`](eval/README.md).

---

## Running Tests

```bash
# Unit tests — no external services required
pytest -m "not db_integration"

# DB-backed tests — needs a real Postgres reachable via PGSQL_RW_DB_URL/
# PGSQL_RO_DB_URL with the bh_agent_rw/bh_agent_ro roles granted (see
# blue_horizon/load_data/regrant_booking_agent_role.sql); tests/api/test_app.py
# additionally needs REDIS_URL and OPENAI_API_KEY to start the real app.
pytest -m db_integration
```

Plain `pytest` runs both groups together, so the `db_integration` tests will fail
without that setup — see the `db_integration` marker docstring in `pyproject.toml`
and the `db-integration-tests` job in `.github/workflows/ci.yml` for how CI
provisions them (against a scratch Neon branch).