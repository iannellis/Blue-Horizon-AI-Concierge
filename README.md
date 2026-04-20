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

The **agent backend** is built with [LangGraph](https://github.com/langchain-ai/langgraph) and served via [FastAPI](https://fastapi.dev). A router LLM classifies each incoming message and dispatches it to one of two specialised sub-agents: a RAG-based information agent (Redis vector store + reranking) or a natural-language-to-SQL rooms/bookings agent (PostgreSQL). Conversation history is maintained across turns via LangGraph's `MemorySaver` checkpointer, keyed by `thread_id`, so multiple concurrent sessions are fully isolated from one another. The backend streams stage-progress events to the client via SSE so the UI can show live status while the agent works.

The **chat UI** is built with [Streamlit](https://streamlit.io). It connects to the FastAPI backend and displays a stage-aware progress indicator inside the assistant bubble as each request is processed, then renders the final response.

Documentation is available in `documentation.pdf`.

---

## Live Demo

A deployed instance runs on HuggingFace Spaces. Access is controlled via Google OAuth. Recruiters and hiring managers can request access to be added to the allowed-users list.

Once access is granted, the agent is available at **[iellis02-blue-horizon.hf.space](https://iellis02-blue-horizon.hf.space)** (this exact URL must be used; Google login will not work from any other origin).

If the Space has gone to sleep and returns a 500 error, it can be restarted from the Space page at [huggingface.co/spaces/iellis02/blue-horizon](https://huggingface.co/spaces/iellis02/blue-horizon).

---

## Architecture

The system is composed of an **orchestration layer** that routes between two agents:

- **Information agent** — answers questions about hotel services, amenities, and policies using RAG over a Redis vector store.
- **Rooms agent** — searches availability and creates bookings using LLM-generated SQL against a PostgreSQL database.

### Orchestration

![Orchestration diagram](orchestration.png)

The orchestration layer is a LangGraph state graph. A router LLM classifies each incoming message and dispatches it to the appropriate sub-agent (or refuses it if out of scope). Conversation history is maintained in memory via LangGraph's `MemorySaver` checkpointer, keyed by `thread_id`.

### Information Agent

![Information agent diagram](information_agent.png)

The information agent follows a RAG pipeline:

1. A **parse** node extracts one or more sub-queries from the user message, along with any constraints (price bounds, duration bounds, booking requirements, and notice time).
2. Three **retrieval** nodes run in parallel against separate Redis vector indices (FAQ, amenities, services).
3. A **rerank** node merges and deduplicates retrieved context.
4. A **respond** node generates the final answer.

Embeddings are produced with OpenAI and stored in Redis via LlamaIndex.

### Rooms Agent

The rooms agent translates user requests into parameterised SQL queries against a PostgreSQL database (hosted on [Neon](https://neon.tech)). SQL guardrails enforce a table allowlist and cap query results at 50 rows before passing them to the LLM, limiting context size and cost.

> **Note:** The rooms database currently has no concept of user identity. There is a single shared pool of reservations with no per-guest ownership, so the agent cannot distinguish one user's bookings from another's.

---

## Repository Structure

```
blue_horizon/          # Main application package
  agents/
    information/       # RAG-based information agent
    rooms/             # SQL-based rooms/booking agent
    orchestration/     # LangGraph router and manager
  api/
    app.py             # FastAPI application
  load_data/
    information_redis.py  # Loads FAQ/services/amenities into Redis
    rooms_pgsql.py        # Loads room/availability data into PostgreSQL
  system_prompts/      # System prompt templates (.txt)
  config.py            # Pydantic configuration models
  neon.py              # Neon branch reset utility

ui/
  app.py               # Streamlit chat UI

deploy/                # Docker/deployment configuration
  supervisord.conf     # Runs FastAPI + Streamlit under supervisord
  generate_secrets.py  # Writes Streamlit secrets from HF Space env vars
  requirements.txt     # Pinned production dependencies

eval/                  # Evaluation framework (see eval/README.md)

notebooks/             # Development and exploration notebooks
  eda.ipynb            # Exploratory data analysis
  information_agent.ipynb
  rooms_agent.ipynb
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
| `POST` | `/v1/chat` | Send a message; returns the complete agent response |
| `POST` | `/v1/chat/stream` | Send a message; streams stage-progress events then the final response as SSE |
| `POST` | `/v1/reset` | Reset the working Neon database branch to its parent baseline (clears bookings) |

### Chat request body

```json
{
  "thread_id": "<uuid>",
  "text": "Do you have any rooms available this weekend?"
}
```

### Streaming events

The `/v1/chat/stream` endpoint emits `text/event-stream` (SSE) events:

```
data: {"type": "stage", "label": "Routing your request…"}
data: {"type": "stage", "label": "Processing your room request…"}
data: {"type": "done", "response": "Here are the available rooms…"}
```

---

## Configuration

All tunable parameters live in [`blue_horizon/app_config.toml`](blue_horizon/app_config.toml). The file is structured into sections that map to the Pydantic models in [`blue_horizon/config.py`](blue_horizon/config.py):

| Section | Controls |
|---------|----------|
| `[orchestration]` | Router LLM, timeouts, retry backoff, concurrency limit |
| `[info]` | Information agent LLM, embeddings model, Redis tuning, retrieval `top_k` |
| `[rooms]` | Rooms agent LLM, SQL guardrails, DB pool settings |
| `[load_data]` | Paths to the source data pickles |
| `[neon]` | Neon project ID and branch name for the `/v1/reset` endpoint |

Runtime secrets are read from a `.env` file (or environment variables). Both the API and the UI call `load_dotenv()` at startup, so a single `.env` file in the project root covers both processes when running locally.

**Required:**

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |
| `REDIS_URL` | Redis connection URL |
| `PGSQL_DB_URL` | PostgreSQL connection URL |

**Optional:**

| Variable | Description |
|----------|-------------|
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

Populate PostgreSQL with room and availability data:

```bash
python -m blue_horizon.load_data.rooms_pgsql
```

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
  -e PGSQL_DB_URL=... \
  blue-horizon
```

The UI is served on port `7860`. The FastAPI backend runs internally on `127.0.0.1:8000` and is not exposed outside the container.

Optional Google OAuth can be enabled by setting `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, and `COOKIE_SECRET` as HuggingFace Space secrets.

---

## Evaluation Metrics

Metrics collected from a structured evaluation run over the 200-case dataset at [`eval/datasets/hotel_agent_eval_200.jsonl`](eval/datasets/hotel_agent_eval_200.jsonl).

### Conversation-level scores

These metrics are assessed once per multi-turn conversation by an LLM judge.

| Metric | Score |
|--------|-------|
| Consumer quality (1–5) | 4.97 |
| Grounding (1–5) | 4.98 |
| Injection resistance (1–5) | 5.00 |

**Consumer quality** and **Grounding** assess response helpfulness and factual fidelity to retrieved context. **Injection resistance** scores how reliably the system refuses prompt-injection attempts.

### Per-turn scores

These metrics are assessed at each individual exchange and then averaged across all turns in the dataset.

| Metric | Score |
|--------|-------|
| Route accuracy | 99.4% |
| Rooms — no unexpected failure rate | 99.3% |
| RAG faithfulness | 97.5% |
| RAG context recall | 100% |
| RAG context precision | 82.8% |
| RAG answer relevancy | 71.5% |
| Info filters pass rate | 100% |
| Info reference subset pass rate | 99.2% |

**Route accuracy** measures how often the router correctly dispatches to the information or rooms agent (or refuses). **RAG faithfulness** measures whether generated answers stay within the retrieved context; **RAG context recall** measures whether all relevant context was retrieved. **Info filters pass rate** measures whether the information agent correctly honours user-specified constraints (price, duration, booking requirements, etc.) when selecting and presenting results. **Info reference subset pass rate** measures whether the expected source documents (FAQ entries, amenity cards, service cards) appear in the set of documents retrieved for a given query.

### Latency (end-to-end, ms)

| Route | p50 | p95 | p99 |
|-------|-----|-----|-----|
| Refuse | 2,602 | 6,920 | 19,894 |
| Info | 8,591 | 29,268 | 46,123 |
| Rooms | 11,090 | 33,329 | 41,692 |

Refuse requests resolve fastest (router only). Info requests go through the full RAG pipeline (parse → parallel retrieval → rerank → respond). Rooms requests run NL-to-SQL generation plus a live PostgreSQL query.

### Stress test — database invariants

A concurrent stress test simulated 50 simultaneous sessions with 5 booking operations each (250 operations total) against the rooms agent. All 250 operations completed without error. Afterwards, two database invariants were verified:

| Invariant | Result |
|-----------|--------|
| Double-booking violations | 0 |
| Reservations with null status | 0 |

A **double-booking violation** occurs when the same room is booked for overlapping dates by two concurrent sessions — i.e., the agent failed to enforce mutual exclusion under load. A **null-status reservation** is a row inserted without a valid status field, indicating a partially-written or corrupted booking.

---

## Evaluation

The `eval/` directory contains a standalone evaluation framework with its own documentation. See [`eval/README.md`](eval/README.md).

---

## Running Tests

```bash
pytest
```