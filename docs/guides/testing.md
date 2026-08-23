# Testing

```bash
# Unit tests -- no external services required
pytest -m "not db_integration"

# DB-backed tests -- needs a real Postgres
pytest -m db_integration
```

Plain `pytest` runs both groups together, so the `db_integration` tests will fail
without the setup below.

## What the DB-backed tests need

- A real Postgres reachable via `PGSQL_RW_DB_URL` and `PGSQL_RO_DB_URL`, with the
  `bh_agent_rw` and `bh_agent_ro` roles granted (see
  [`regrant_booking_agent_role.sql`](https://github.com/iannellis/Blue-Horizon-AI-Concierge/blob/main/blue_horizon/load_data/regrant_booking_agent_role.sql)).
- `tests/api/test_app.py` additionally needs `REDIS_URL` and `OPENAI_API_KEY` to start
  the real application.
- One test in `test_db_invariants.py` wants `PGSQL_ROOT_DB_URL`, with schema-owner
  privilege against whichever branch the rest of the run targets, so it can temporarily
  drop `booking_rooms_no_overlap` inside a rolled-back transaction. It skips gracefully
  if unset.

!!! danger "`PGSQL_ROOT_DB_URL` is not `PGSQL_ROOT_PARENT_DB_URL`"
    The Parent-branch variable must never point at a branch these tests run against.
    See [Configuration](configuration.md#the-three-root-urls).

## The Production-branch guard

A normal local `.env` points `PGSQL_RW_DB_URL` and `PGSQL_RO_DB_URL` at the Production
branch, because that is what running the app locally needs. Running `db_integration`
tests in that environment would book and cancel real rows there.

[`tests/conftest.py`](https://github.com/iannellis/Blue-Horizon-AI-Concierge/blob/main/tests/conftest.py)
guards against this. Whenever `PGSQL_RW_EVAL_DB_URL`, `PGSQL_RO_EVAL_DB_URL`, and
`PGSQL_ROOT_EVAL_DB_URL` are also set, its `pytest_configure` hook copies each onto its
plain-named counterpart before any test runs, so a local `db_integration` run lands on
the Development branch instead. CI's `db-integration-tests` job gets the same
substitution from its own secrets.

**Without those `_EVAL` variables set locally**, `db_integration` tests run against
whatever the plain-named variables already point at, Production included. Setting the
`_EVAL` variables locally is worth doing before running this suite by hand.

## Test layout

```
tests/
  api/            # FastAPI endpoints, content negotiation, SSE events
  booking/        # write_ops, proposals, guardrails, DB invariants, role privileges
  information/    # Parser, retrieval, graph nodes
  orchestration/  # Router and manager
  load_data/      # Pure pandas helpers: overlap resolution, clamping, customer ranking
  deploy/         # generate_secrets.py
  eval/           # Evaluator and harness unit tests
  ui/             # Streamlit app behavior
  conftest.py     # The Production-branch guard
```

## Linting and type checking

Per the project conventions, after any Python change:

```bash
ruff check --select ALL --no-cache <file>
python -m pyright --pythonpath D:/uv_envs/blue_horizon_3.13/Scripts/python.exe <file>
```

`--pythonpath` must be passed explicitly. Without it, `python -m pyright` resolves
imports against whatever interpreter is first on `PATH`, which may not be the project's
environment, producing a misleading clean run.

## CI

See [Deployment and CI](deployment.md#continuous-integration) for how the test jobs are
sequenced and which of them gate the deploy.
