# Evaluation harness

End-to-end evaluation suite for the Blue Horizon AI Concierge agent. Covers multi-turn
conversation quality, routing accuracy, injection resistance, database integrity, RAG
retrieval quality, and concurrency stress testing.

## Directory structure

```
eval/
├── run_experiment.py           # Main evaluation runner (LangSmith aevaluate)
├── ci_check.py                 # Baseline comparison tool for CI
├── create_langsmith_dataset.py # Upload JSONL cases to a LangSmith dataset
├── config.py                   # Pydantic config models and TOML loader
├── booking_db_manager.py       # Neon branch reset helper
├── db_invariants.py            # Shared DB invariant queries
├── _result_utils.py            # Result aggregation and latency utilities
├── _utils.py                   # Shared helpers (json_safe, truncate, …)
├── analyze_results.py          # Ad-hoc results analysis script
│
├── evaluators/                 # One module per evaluator
│   ├── _routing.py             # Route accuracy
│   ├── _injection.py           # Injection tripwire detection
│   ├── _booking.py             # Rooms tool outcomes + DB invariants
│   ├── _judge.py               # LLM-as-judge rubric scoring (Gemini)
│   ├── _rag.py                 # Ragas faithfulness / relevancy / precision / recall
│   ├── _rag_prompts.py         # Custom context-precision prompt
│   ├── _info_references.py     # Info agent reference subset pass rate
│   ├── _info_filters.py        # Info agent expected filter extraction
│   └── _latency.py             # Per-turn wall-clock latency
│
├── langsmith_target/           # Target function called per example by aevaluate
│
├── stress/                     # Concurrent booking stress test
│   ├── __main__.py             # Entry point: python -m eval.stress
│   ├── runner.py               # Async workload driver
│   ├── workload.py             # Operation mix (BOOK / MODIFY / CANCEL)
│   ├── db.py                   # DB invariant checks
│   ├── reconciliation.py       # Post-run reconciliation
│   ├── artifacts.py            # Log + summary writers
│   └── models.py               # Stress run data models
│
├── datasets/
│   ├── hotel_agent_eval_23.jsonl    # 23-case smoke dataset
│   ├── hotel_agent_eval_206.jsonl   # 206-case full dataset
│   └── hotel_agent_eval_206_manifest.json
│
├── baselines/
│   ├── hotel_agent_eval_23_baseline.json   # CI smoke eval thresholds
│   └── hotel_agent_eval_206_baseline.json  # Full eval thresholds
│
├── eval_config_23.toml         # Config for the 23-case smoke eval
├── eval_config_206.toml        # Config for the 206-case full eval
├── stress_config.toml          # Config for the stress test
└── requirements.txt            # Pinned dependencies for CI
```

## Prerequisites

!!! warning "Evaluation assumes a Neon-hosted PostgreSQL database"
    This is a harder requirement than the application itself has. The app runs against
    any PostgreSQL instance, but the eval and stress harnesses reset a database branch
    to its parent baseline **before every run**, so that each run starts from identical
    inventory and results are comparable across runs. That reset goes through the Neon
    management API, driven by `eval/booking_db_manager.py` and the `[neon]` section of
    the eval config, which names the project id and the branch to restore.

    Consequences of pointing the harness at a plain PostgreSQL server:

    - `NEON_API_KEY` has nothing to authenticate against and the reset step fails.
    - Even with the reset stubbed out, results drift between runs, because bookings
      committed by one run remain and change what the next run can book.

    Running elsewhere means replacing `booking_db_manager.reset_branch` with an
    equivalent restore-to-baseline step, not merely skipping it. Note also that the
    branch named in the config is reset *in place*, which is why
    `PGSQL_ROOT_PARENT_DB_URL` is a separate variable that must never point at it. See
    [Configuration](../guides/configuration.md#the-three-root-urls).

| Variable | Required | Description |
|---|---|---|
| `LANGSMITH_API_KEY` | Yes | LangSmith API key |
| `LANGSMITH_TRACING` | Yes | Set to `true` to enable tracing |
| `LANGCHAIN_PROJECT` | Yes | LangSmith project name for traces |
| `GEMINI_API_KEY` | Yes | Gemini API key (judge, Ragas) |
| `OPENAI_API_KEY` | Yes | OpenAI API key (agents, embeddings) |
| `PGSQL_RW_EVAL_DB_URL` | Yes | Read-write (`bh_agent_rw`) connection string for the eval DB |
| `PGSQL_RO_EVAL_DB_URL` | Yes | Read-only (`bh_agent_ro`) connection string for the eval DB, used exclusively by `run_sql` |
| `NEON_API_KEY` | Yes | Neon management API key (branch reset) |
| `REDIS_URL` | Yes | Redis connection URL (agent cache) |

```bash
pip install -r eval/requirements.txt
```

The `_EVAL`-suffixed database URLs are copied onto their plain-named counterparts early
in `main()`, before the first `load_app_config()` call. That ordering matters: CI only
ever sets the `_EVAL` names, so a `load_app_config()` that runs before the override has
nothing to validate against and fails with a pydantic `ValidationError`.

## Running evaluations

### Smoke eval (23 cases)

Used as a CI gate on every push to `main`:

```bash
python -m eval.run_experiment --config eval/eval_config_23.toml
```

### Full eval (206 cases)

Run manually for comprehensive quality checks:

```bash
python -m eval.run_experiment --config eval/eval_config_206.toml
```

Both commands write results to `eval/outputs/<experiment_name>/`:

- `results.jsonl` - per-case evaluator feedback
- `summary.json` - run metadata plus separate `case_based_summary` and
  `turn_based_summary` metric blocks, along with per-route latency quantiles

Logs are written to `eval/logs/<experiment_name>.log` and mirrored to stdout.

### CLI options

| Flag | Description |
|---|---|
| `--config PATH` | TOML config to use. Defaults to the packaged `eval_config.toml` when omitted. |
| `--case-id CASE_ID` | Restrict the run to one dataset `case_id` (e.g. `case_0117`). May be passed multiple times. |
| `--router-only` | Run only the `eval_routing_accuracy` evaluator, skipping every other evaluator including the LLM judge and Ragas metrics. Useful for a fast, judge-free routing check. |
| `--no-upload` | Run entirely locally, bypassing `aevaluate` so zero requests reach LangSmith's trace-ingestion endpoint. |

A quick, fully local routing check over the full dataset:

```bash
python -m eval.run_experiment --config eval/eval_config_206.toml --router-only --no-upload
```

!!! note "Why `--no-upload` bypasses `aevaluate` entirely"
    `aevaluate(upload_results=False)` still traces every target run, because the
    installed LangSmith SDK's `_aforward` helper hardcodes
    `tracing_context(enabled=True)` regardless of that parameter. `--no-upload` therefore
    calls `run_example` directly through a separate `_run_locally` path with no tracing
    wrapper.

    Dataset examples are still *read* from LangSmith, which is a lightweight read rather
    than a trace. Local `results.jsonl` and `summary.json` are written as normal.

## How evaluation works

### Pipeline overview

```
run_experiment.py
│
├── booking_db_manager.py  ← reset Neon branch to clean snapshot
│
└── aevaluate(run_example, evaluators=[...])
    │
    ├── run_example(example)          ← called once per dataset example (concurrent)
    │   ├── ensure_orchestration_ready()   ← lazy-init shared OrchestrationManager singleton
    │   └── for each turn:
    │       ├── EvalCaptureCallback()      ← fresh capture handler per turn
    │       ├── orchestration_mgr.ainvoke(user_text, callbacks=[callback])
    │       └── collect turn_output dict
    │
    └── evaluator(run_output, example)  ← called once per case after run_example
        └── scores each turn against expected_* fields from the dataset
```

### OrchestrationManager singleton

A single `OrchestrationManager` is started once at the beginning of the run and shared
across all concurrent `run_example` calls via an async lock. This matches how the agent
runs in production and avoids re-initializing the LangGraph graph, database pool, and
Redis cache for every example.

Each case gets a fresh UUID `thread_id`, which isolates its LangGraph conversation state
from all other concurrent cases.

!!! danger "`run_id` is a reserved key in LangGraph config metadata"
    Passing `config['metadata'] = {'run_id': ...}` with the same value across multiple
    `ainvoke` calls on one `thread_id` causes LangGraph (1.0.x and later) to deduplicate
    runs. Only the first call executes the graph; subsequent ones silently return the
    cached checkpoint. Use any other key name.

### Per-turn execution

For every turn in a case, `run_example`:

1. **Creates a fresh `EvalCaptureCallback`**, a LangChain `AsyncCallbackHandler` that
   intercepts LangGraph's internal events during `ainvoke`.

2. **Calls `orchestration_mgr.ainvoke(user_text, callbacks=[callback])`**, running the
   full agent graph for the turn.

3. **Captures**, via LangChain callback hooks:

   | Field | Captured by | Source |
   |---|---|---|
   | `route_pred` | `on_chain_end` | Orchestrator node output containing a `"route"` key |
   | `tool_summary` | `on_tool_start` / `on_tool_end` | One entry per tool call: `run_sql`, `parser`, `query_faq`, `query_amenities`, `query_services`, `merge` |
   | `contexts_used` | `on_chain_end` (merge node) + `on_tool_end` (`run_sql`) | Merge node `top_results` text plus SQL row strings, used by RAG and judge evaluators |

4. **Measures wall-clock latency** with `asyncio.get_running_loop().time()` around the
   `ainvoke` call.

5. **Appends a `turn_output` dict** to the case's `turn_outputs` list:
   `{assistant_text, route_pred, tool_summary, contexts_used, latency_ms}`.

### How evaluators consume turn outputs

Each evaluator receives:

- **`run_output`** - `{"turn_outputs": [...], "case_tags": [...]}` from `run_example`
- **`example`** - the raw dataset record with `turns[].expected_*` fields

Evaluators zip `turn_outputs` with `turns` and score each turn independently. For
example, `eval_routing_accuracy` compares `turn_output["route_pred"]` against
`turn["expected_route"]`; `eval_rag_metrics_info_turns` passes
`turn_output["contexts_used"]` and `turn["reference"]` to Ragas.

Results are written to `results.jsonl` as LangSmith feedback items, then aggregated into
`summary.json`, which reports:

- `case_based_summary` - arithmetic means over case-level evaluator scores
- `turn_based_summary` - turn-weighted or direct per-turn aggregates for metrics that
  expose turn-level scoring data
- `latency_quantiles_ms` - per-route wall-clock latency quantiles

## Experiment metadata

Each run's metadata is derived automatically and attached to the LangSmith experiment,
and written into `summary.json` so it survives when `upload_results` is off:

- `git_sha` and `git_dirty`, via `langsmith.env.get_git_info()`
- `dataset_name`, `dataset_limit`, `max_concurrency`, `upload_results`
- Router, info, and booking model plus `reasoning_effort`; `judge_model`;
  `ragas_llm_model` and `ragas_embedding_model`
- `app_config_sha256`, `eval_config_sha256`, and one content hash **per prompt file**
  (router, info-system, info-parser, booking)

The per-file prompt hashes exist so a diff between two runs points at *which* prompt
changed rather than reporting that something in the prompts changed. Content
fingerprints matter separately from the commit SHA because a dirty tree or a `--config`
override changes behavior without changing the commit.

## Checking results against a baseline

```bash
python -m eval.ci_check eval/outputs/<experiment_name>/results.jsonl
```

Uses `eval/baselines/hotel_agent_eval_23_baseline.json` by default. For the 206-case
baseline:

```bash
python -m eval.ci_check eval/outputs/<experiment_name>/results.jsonl \
  --baseline eval/baselines/hotel_agent_eval_206_baseline.json
```

Output is a table comparing each metric against its baseline and minimum threshold,
followed by per-route p50/p95/p99 latency stats. Exits `0` on pass, `1` on any failure.
