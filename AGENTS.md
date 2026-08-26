**Before running any command that needs the Python environment, read `AGENTS.local.md`
at the repo root if it is present.** It holds this machine's interpreter path, which is
deliberately not recorded in a tracked file. It is untracked, so it is absent in a fresh
clone, and its absence is not an error. Claude Code loads it automatically through
`CLAUDE.md`; other assistants should open it explicitly, since neither Codex nor most
other tools support imports.

## Code conventions

* All Python code should be well-structured, modular, follow PEP8, and have meaningful names.
* For library modules, code should be written following the Clean Code guidelines. That is, if function A calls B then C, but B calls D, the they should be layed out in the order A,B,D,C. Same goes for methods inside classes. Pure dataclasses, classes inheriting from Protocol, and classes inheriting from the Pydantic BaseModel, can go near the top of the file. Do as you wish for scripts.
* After completing each of the user's requests that involves a Python code change, using the project's Python environment (its path is machine-specific; see `AGENTS.local.md`), run `ruff --select ALL --no-cache` on each Python file edited. Fix all listed problems. Then run pyright on each Python file edited. Fix all listed problems.
* Each Python function, method, class, and the module should have a complete Google docstring including descriptions of inputs, output, and any exceptions raised.
* When writing free-form text (readme, documentation, or comments), do NOT use em-dashes.

## What this project is

An AI hotel concierge. A router LLM classifies each guest message and dispatches it to
one of two sub-agents: an **information agent** (RAG over three Redis vector indices)
and a **booking agent** (read-only NL-to-SQL search against PostgreSQL on Neon, plus
proposals the guest confirms). Anything outside those two jobs is refused.

The organising principle of the whole system: **the model proposes, the application
decides.** The model can search and suggest. It cannot write to the database, cannot
author a confirmation number, and cannot claim a booking succeeded.

## Invariants

These are load-bearing. Breaking one is a defect even if the tests pass and the feature
works. Each says why, so you can tell when a change is genuinely adjacent versus when it
is dismantling a guarantee.

1. **The model never writes to the database.** Its `run_sql` tool connects as
   `bh_agent_ro`, which holds `SELECT` on `rooms` and `room_availability` and nothing
   else. `customers`, `bookings`, and `booking_rooms` are off that role entirely, so
   guest data cannot reach a third-party inference provider through free-form SQL.
2. **`blue_horizon/agents/booking/write_ops.py` is the only module that writes bookings,
   and `POST /v1/booking/confirm` is its only caller.** Do not add a second write path.
3. **`propose_*` tools return a proposal id and nothing else.** Never success text. The
   UI builds its confirmation dialog strictly from the proposal's stored fields, never
   from the assistant's prose, so a model that describes a booking wrongly cannot make
   the guest agree to the wrong thing.
4. **The application authors confirmation numbers** (`write_ops._confirmation_number`).
   The model is prompted never to invent one, and the eval suite checks for it
   (`booking_no_unbacked_success_claims`).
5. **`list_my_bookings` takes `customer_id` from `RunnableConfig`, injected server-side.**
   The parameter is invisible to the model by design. Never promote it to a
   model-supplied argument; that is how one guest reads another's reservations.
6. **Double-booking is structurally impossible, not detected afterward.** Three layers:
   `SELECT ... FOR UPDATE` on `room_availability` inside each write transaction (the
   primary mechanism), the `booking_rooms_no_overlap` GiST exclusion constraint, and the
   `prevent_maintenance_booking` trigger. The constraint is deliberately **not**
   `DEFERRABLE`, which would let overlapping rows exist transiently.
7. **`startup_check` must keep refusing to boot when a trial write through
   `PGSQL_RO_DB_URL` succeeds.** It catches the configuration mistake of pointing both
   URLs at the same role, which would silently defeat invariant 1.
8. **The SQL guardrail (`booking/guardrails.py`) stays even though the grant makes it
   redundant.** Two independent mechanisms is the point. It also produces better error
   messages than a raw privilege refusal.
9. **The information agent is a fixed DAG, not a tool-calling agent.** Do not reintroduce
   retrieval tools or a decision loop. An empty retrieval is a correct answer, and the
   previous tool-calling version treated it as a malfunction: retrying, widening filters
   on its own initiative, or fabricating a service.
10. **Nothing may hold a `booking_id` across a Neon branch reset.** Resets are a
    first-class operation here; anyone can put the demo back to a known state.
11. **`reload_sql_tables()` regrants in the same transaction as the reload.** Dropping and
    recreating tables drops their grants. When the regrant was a separate manual step it
    was forgotten, leaving the Parent branch with zero grants and silently breaking every
    branch reset downstream.
12. **`cancel_booking` never touches `price`,** so a night's rate survives a
    book/cancel round trip.
13. **The loader inserts every source customer, not just the seeded ones.** Loading a
    subset silently drops every pre-existing booking belonging to anyone else on a foreign
    key, which corrupts `room_availability`. The rich block is remapped to a dense
    `customer_id` range of `[1, seeded_customer_count]` and loaded first so the UI and
    eval harnesses keep working.
14. **The judge model stays in a different family from the agent model.** Agents run on
    OpenAI, the LLM judge and Ragas metrics run on Gemini. Scoring a model with itself
    invites correlated blind spots. Gemini must never appear in the serving path.

## Module boundaries

- `ui/app.py` talks to the API over HTTP only. It imports no `blue_horizon` code, and
  must not start.
- `eval/` imports `blue_horizon`. The reverse never happens.
- Tunables live in `blue_horizon/app_config.toml`, typed by Pydantic models in
  `blue_horizon/config.py`, reached as `AppConfig`. Secrets come from environment
  variables only. Do not hardcode a tunable, and do not add a second copy of one that is
  already in config: `seeded_customer_count` was previously duplicated as a literal in
  four files with nothing keeping them in sync.
- System prompts are `.txt` files in `blue_horizon/system_prompts/`, referenced by
  filename through config, never inlined in Python.
- Retry policy is `tenacity`. SQL parsing is `sqlglot`. DataFrame validation is `pandera`.
  Four bespoke backoff implementations and a regex SQL guardrail preceded these; do not
  hand-roll replacements.

## Where things are

| Path | Contents |
|---|---|
| `blue_horizon/agents/orchestration/` | LangGraph router and manager, `MemorySaver` checkpointing per `thread_id` |
| `blue_horizon/agents/information/` | Parse, three parallel retrievers, merge, respond |
| `blue_horizon/agents/booking/factory.py` | Tool construction: `run_sql` (read-only) and the `propose_*` tools |
| `blue_horizon/agents/booking/write_ops.py` | `commit_booking` / `cancel_booking` / `modify_booking`, the only writers |
| `blue_horizon/agents/booking/proposals.py` | In-process `ProposalStore`, propose to confirm/dismiss lifecycle |
| `blue_horizon/agents/booking/guardrails.py` | `sqlglot` AST allowlist |
| `blue_horizon/api/app.py` | FastAPI app, SSE streaming, the confirm and dismiss endpoints |
| `blue_horizon/load_data/` | Redis and PostgreSQL loaders, plus `regrant_booking_agent_role.sql` |
| `blue_horizon/neon.py` | Neon branch reset utility |
| `eval/` | LangSmith harness, evaluators, 206-case dataset, concurrency stress test |
| `tests/` | Mirrors the package layout; `db_integration` marker for DB-backed tests |

## Read before changing

The documentation site under `docs/` is the authority. It is written for humans, it is
current, and it explains reasoning this file only summarises. Consult it rather than
inferring intent from the code.

- `docs/architecture/` for how the pieces fit: `orchestration.md`,
  `information-agent.md`, `booking-agent.md`. Read the relevant page before changing an
  agent's control flow or tool surface.
- `docs/design-decisions.md` for why something is the way it is, including the failure
  that prompted each change. Read this before proposing to revert or simplify anything
  in the invariants list above.
- `docs/api.md` for the wire protocol: endpoints, SSE events, the propose/confirm
  contract.
- `docs/guides/configuration.md` for every tunable and environment variable.
- `docs/evaluation/` for the harness, datasets, baselines, and stress test.

Keep this split intact when you write documentation. `docs/architecture/` describes the
system as it is now and must be updated in the same commit as the code it describes.
`docs/design-decisions.md` is a log: add a new entry, do not rewrite an old one. If a
fact would need editing in both files after a code change, it is duplicated rather than
split.

## Environment and tooling

The project targets Python 3.13 in a uv-managed environment. That environment's path is
machine-specific and is therefore not recorded here: it lives in `AGENTS.local.md`,
which is untracked. In a fresh clone, create one before running the commands below.
Dependency groups are `ui`, `eval`, `notebook`, `lint`, and `docs`.

```bash
ruff check --select ALL --no-cache <file>
python -m pyright --pythonpath <project interpreter> <file>

pytest -m "not db_integration"   # unit tests, no external services
pytest -m db_integration         # needs a real Postgres
```

Gotchas that have each cost real debugging time:

- **`pyright` needs `--pythonpath` explicitly.** Without it, `python -m pyright` resolves
  imports against whatever interpreter is first on `PATH` and reports a misleading clean
  run.
- **`uv sync` needs `UV_PROJECT_ENVIRONMENT` set,** or it creates a stray `.venv/` in the
  repo root instead of using the environment above. The `VIRTUAL_ENV` mismatch warning it
  prints is expected.
- **`run_id` is a reserved key in LangGraph config metadata.** Passing the same value
  across several `ainvoke` calls on one `thread_id` makes LangGraph deduplicate: only the
  first call executes, the rest silently return the cached checkpoint. Use any other key
  name.
- **A local `.env` points `PGSQL_RW_DB_URL` and `PGSQL_RO_DB_URL` at Production,** because
  that is what running the app locally needs. `tests/conftest.py` copies the `_EVAL`
  suffixed variables over them before any test runs, but only when all three are set. Set
  them locally before running `db_integration` by hand, or the suite books and cancels
  real rows in Production.
- **`PGSQL_ROOT_PARENT_DB_URL` is not `PGSQL_ROOT_DB_URL`.** The first addresses the
  Parent branch and is read only by the data loader. The second addresses whatever branch
  a test run targets. `reload_sql_tables` must never run against a branch that gets reset
  in place.
- **`.env` cannot be loaded with plain `source .env`.** The `PGSQL_*_DB_URL` values
  contain an unescaped `&` that backgrounds the assignment. Use a line-by-line `read` and
  `export` loop.
- **Editing `eval/datasets/*.jsonl` does not update the hosted LangSmith dataset** that
  `run_experiment.py` actually reads. Patch the hosted example with
  `client.update_example` too.
- **An occasional "SSL connection closed" warning right after a Neon branch reset in
  `eval.stress` is self-healing** and already retried by design. Do not fix it.
