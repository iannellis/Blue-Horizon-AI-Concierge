# Configuration

All tunable parameters live in
[`blue_horizon/app_config.toml`](https://github.com/iannellis/Blue-Horizon-AI-Concierge/blob/main/blue_horizon/app_config.toml).
The file is structured into sections that map to the Pydantic models in
[`blue_horizon/config.py`](https://github.com/iannellis/Blue-Horizon-AI-Concierge/blob/main/blue_horizon/config.py).

| Section | Controls |
|---------|----------|
| `[orchestration]` | Router LLM, timeouts, retry backoff, concurrency limit |
| `[info]` | Information agent LLM, embeddings model, Redis tuning, retrieval `top_k` |
| `[booking]` | Booking agent LLM, SQL guardrails, DB pool settings |
| `[booking.proposals]` | `ttl_s`, how long an unconfirmed proposal stays in the in-process store before it is purged |
| `[load_data]` | Paths to the source data pickles, and `seeded_customer_count` |

Neon branch management (project ID, branch name, reset tuning) lives entirely in
`eval/`, not here - the serving app has no Neon management awareness. See
[Evaluation](../evaluation/harness.md#prerequisites).

## Notable settings

**`[orchestration.orchestration].llm_concurrency`** (default 15) caps concurrent LLM
pipeline executions. It prevents tokens-per-minute exhaustion under high concurrency,
which otherwise fails every in-flight request at once rather than making a few wait.

**`[booking.db.pool].max_idle_s`** (default 240) closes idle connections before Neon's
serverless compute suspends at roughly 300 seconds, so the pool does not hand out
connections the server has already killed.

**`[booking.db.pool].min_size`** is 0, so no connections are held proactively. This
means the first query after an idle stretch pays a cold-start cost, which is why UI data
fetches use a 20-second timeout rather than the health check's 3 seconds.

**`[booking.proposals].ttl_s`** (default 1800) bounds the proposal store's size only. A
proposal reserves no inventory, so expiry costs a guest nothing beyond having to ask
again.

**`[load_data.booking_pgsql].seeded_customer_count`** (default 15) sets how many source
customers get the dense low `customer_id` block that the UI's guest assignment and the
eval and stress harnesses assume. It is the single source of truth that both
`booking_pgsql.py` and `write_ops.list_customers` read, so the two cannot drift apart.

**`[orchestration.orchestration].unavailable_retry_after_s`** (default 5.0) is the
`Retry-After` seconds returned with a `/v1/chat` (and `/v1/customers`, `/v1/bookings`)
`503` while the orchestrator is not ready, in both the `STARTING` and `FAILED` readiness
states. Kept short by design: the init loop keeps retrying in both states, so a client
polling at this interval sees a transient outage recover on its own. See
[Orchestration](../architecture/orchestration.md#readiness).

**`[orchestration.messages].failed`** is shown while readiness is `FAILED`: the last
startup attempt was classified permanent. Unlike `.unavailable` (shown for `STARTING`,
carrying no retry instruction because recovery is expected on its own), this string must
not offer a retry affordance the guest cannot act on - the process itself keeps
retrying regardless, but nothing the guest does changes that.

**`statement_timeout` is not in this file.** It is set at the database role level, with
`ALTER ROLE`, so that it applies under PgBouncer transaction pooling, where a
per-connection `SET` would not reliably survive. It bounds every statement the booking
agent issues, including the model-authored SQL that `run_sql` runs as `bh_agent_ro`. It is
applied by `blue_horizon/load_data/regrant_booking_agent_role.sql`, which the data loader
runs on Parent, and a branch reset carries it to every child branch along with the
grants. See [Running Locally](running-locally.md#2-create-the-database-roles).

**`search_path` is not set at the role level.** The agent roles use the PostgreSQL
default of `"$user", public`, which resolves to `public` because no schema is named after
either role. Code paths that depend on it, such as `db_utils` and the loader, issue
`SET search_path TO public` on their own connections.

## Environment variables

Runtime secrets are read from a `.env` file or from the environment. Both the API and
the UI call `load_dotenv()` at startup, so a single `.env` file in the project root
covers both processes when running locally.

### Required

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |
| `REDIS_URL` | Redis connection URL |
| `PGSQL_RW_DB_URL` | PostgreSQL URL authenticated as the read-write `bh_agent_rw` role. Backs `write_ops` (booking commits), `/v1/customers`, and `/v1/bookings`. Never reachable from the model. |
| `PGSQL_RO_DB_URL` | PostgreSQL URL authenticated as the read-only `bh_agent_ro` role. The *only* database connection the model's `run_sql` tool can use; Postgres itself refuses any write attempted through it. |

!!! warning "The roles are not created automatically"
    `bh_agent_rw` and `bh_agent_ro` must already exist in the database with passwords
    set before these URLs will work, and on Neon they must exist on **every branch**
    these URLs point at, since a role belongs to a branch rather than to the project.
    See [Running Locally](running-locally.md#2-create-the-database-roles) for the
    creation step and the per-branch trap.

    Pointing both URLs at the same role silently defeats the read-only guarantee, which
    is why `startup_check` refuses to start if a trial write through the read-only URL
    succeeds.

### Optional

| Variable | Description |
|----------|-------------|
| `PGSQL_ROOT_PARENT_DB_URL` | Schema-owner URL for the **Parent** branch specifically. Used only by the data-loading tooling; not read by the API or UI at runtime. Required to load data, not to start the app afterward. |
| `LANGSMITH_API_KEY` | LangSmith API key, enabling tracing |
| `LANGSMITH_TRACING` | Set to `true` to activate tracing (requires `LANGSMITH_API_KEY`) |
| `LANGCHAIN_PROJECT` | LangSmith project name to log traces under |
| `GOOGLE_CLIENT_ID` | Google OAuth client ID, enabling the login gate in the UI |
| `GOOGLE_CLIENT_SECRET` | Google OAuth client secret (required when `GOOGLE_CLIENT_ID` is set) |
| `COOKIE_SECRET` | Secret used to sign the Streamlit auth session cookie (required when OAuth is enabled) |

### The three root URLs

Three separate schema-owner variables exist, and confusing them is destructive. They are
deliberately named so that they cannot be mistaken for one another:

| Variable | Branch | Read by |
|---|---|---|
| `PGSQL_ROOT_PARENT_DB_URL` | **Parent only** | `booking_pgsql.reload_sql_tables` and the regrant script |
| `PGSQL_ROOT_DB_URL` | Whatever branch a `db_integration` run targets | One test in `test_db_invariants.py`; skips gracefully if unset |
| `PGSQL_ROOT_EVAL_DB_URL` | Development | Copied onto `PGSQL_ROOT_DB_URL` by `tests/conftest.py` and by CI |

`reload_sql_tables` drops and recreates tables, so it must never run against a branch
that gets reset in place. That is the entire reason `PGSQL_ROOT_PARENT_DB_URL` has its
own name rather than sharing `PGSQL_ROOT_DB_URL`. See
[Design Goals and Decisions](../design-decisions.md#test-databases-are-separated-from-production-by-construction).

### Eval-only variables

The evaluation harness uses `_EVAL`-suffixed variants, documented in
[Evaluation](../evaluation/harness.md#prerequisites). `tests/conftest.py` copies them
over their plain-named counterparts so a local test run cannot touch Production.
