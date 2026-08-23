# Evaluation results

Metrics collected over the 206-case dataset at
[`eval/datasets/hotel_agent_eval_206.jsonl`](https://github.com/iannellis/Blue-Horizon-AI-Concierge/blob/main/eval/datasets/hotel_agent_eval_206.jsonl).

**Every figure below is the mean of three consecutive full runs** on 2026-08-18
(commit `147aec6`, after the context-precision prompt generalization), with the observed
range across those runs given alongside. Single-run variance on the judge-scored metrics
is large enough that a one-run number is misleading, which is why the repository's
206-case baseline is also set from a three-run average rather than a single run.

Raw outputs:
[`eval/outputs/26-08-18-after_precision_prompt_update/`](https://github.com/iannellis/Blue-Horizon-AI-Concierge/tree/main/eval/outputs/26-08-18-after_precision_prompt_update).

!!! info "These numbers are a snapshot, not a target"
    They shift whenever the underlying models, prompts, or agent code change, and are
    expected to be updated regularly as the project evolves. The pinned regression
    thresholds in [`eval/baselines/`](datasets.md#baseline-files), not this table, are
    the source of truth CI checks against.

## What grades what

Every metric that requires a model judge or scorer - all three conversation-level scores
plus the four RAG metrics - is graded by Gemini `gemini-3.5-flash-lite`, configured in
[`eval/eval_config_206.toml`](https://github.com/iannellis/Blue-Horizon-AI-Concierge/blob/main/eval/eval_config_206.toml)
as both `[judge].model` and `[ragas].llm_model`.

**RAG answer relevancy** additionally embeds the question with the Gemini
`gemini-embedding-2` model (`[ragas].embedding_model`) to compare it against the query
Ragas regenerates from the answer. It is the only metric in this run that consumes an
embedding model rather than pure LLM judging.

The remaining per-turn metrics - route accuracy, booking outcomes, info filters and
reference checks - are deterministic, code-based checks with no model in the loop.

The judge is deliberately from a different model family than the agents, which run on
OpenAI. See
[Design Goals and Decisions](../design-decisions.md#the-judge-is-a-different-model-family-from-the-agent).

## Conversation-level scores

Assessed once per multi-turn conversation by the LLM judge.

| Metric | Score | Range across runs |
|--------|-------|-------------------|
| Consumer quality (1-5) | 4.96 | 4.94 - 4.98 |
| Grounding (1-5) | 5.00 | 4.99 - 5.00 |
| Injection resistance (1-5) | 5.00 | 5.00 - 5.00 |

**Consumer quality** and **Grounding** assess response helpfulness and factual fidelity
to retrieved context. **Injection resistance** scores how reliably the system refuses
prompt-injection attempts, and was a clean 5.00 in all three runs.

!!! warning "Near-ceiling scores measure the dataset as much as the agent"
    Grounding and injection resistance sitting at the top of the scale means these runs
    found no failures, not that none exist. A metric with no headroom cannot show
    improvement and detects only severe regressions. The cases that *do* discriminate
    here are the RAG metrics and the filter pass rate below; the judge scores are best
    read as a tripwire rather than a quality gradient.

## Per-turn scores

Assessed at each individual exchange and averaged across all turns in the dataset.

| Metric | Score | Range across runs | Scored by |
|--------|-------|-------------------|-----------|
| Route accuracy | 100% | 100% - 100% | code |
| Booking - no unexpected failure rate | 100% | 100% - 100% | code |
| Booking - tool call success rate | 100% | 100% - 100% | code |
| RAG faithfulness | 95.7% | 95.0% - 96.0% | `gemini-3.5-flash-lite` |
| RAG context recall | 99.3% | 99.0% - 99.5% | `gemini-3.5-flash-lite` |
| RAG context precision | 90.1% | 89.6% - 90.4% | `gemini-3.5-flash-lite` |
| RAG answer relevancy | 83.9% | 83.6% - 84.2% | `gemini-3.5-flash-lite` + `gemini-embedding-2` |
| Info filters pass rate | 98.3% | 96.7% - 99.2% | code |
| Info reference subset pass rate | 100% | 100% - 100% | code |

Database invariants (`db_no_double_booking`, `db_no_null_status`,
`db_confirmation_numbers_unique`, `db_confirmed_bookings_have_receipt`,
`db_no_available_with_null_price`) and `booking_no_unbacked_success_claims` passed on
every case in all three runs.

!!! note "Turn-weighted here, case-weighted in the baseline"
    `summary.json` reports each metric two ways. `turn_based_summary` averages over every
    turn in the dataset, so a long conversation contributes more than a short one.
    `case_based_summary` averages the per-case scores, giving every conversation equal
    weight. The table above is turn-based, matching what this section claims to measure.

    [`eval/baselines/`](datasets.md#baseline-files) and `ci_check.py` use the
    **case-based** figures, so the pinned baseline values differ slightly from this
    table. For these runs: faithfulness 0.9601 case vs 0.9565 turn, precision 0.8917 vs
    0.9007, recall 0.9915 vs 0.9927, answer relevancy 0.8427 vs 0.8389, filters 0.9790
    vs 0.9833. The conversation-level judge scores above are case-based already, and
    match the baseline exactly.

**Route accuracy** measures how often the router correctly dispatches to the information
or rooms agent, or refuses. There were zero misroutes across all 206 multi-turn cases in
all three runs.

**Booking - no unexpected failure rate** measures whether a booking, cancellation, or
modification's success-or-failure outcome matched the case's expectation. A request for
an already-taken night is *expected* to fail, and that is not counted against the agent.

**Booking - tool call success rate** is stricter: it flags any `propose_*` or `run_sql`
call that errored on a turn expected to succeed. This ran at 96.2% before a fix in
[`booking.txt`](https://github.com/iannellis/Blue-Horizon-AI-Concierge/blob/main/blue_horizon/system_prompts/booking.txt)
requiring every column in a `rooms`/`room_availability` join to be table-qualified. Both
tables define `max_occupancy`, so an unqualified reference was ambiguous and errored.
All three runs show zero tool-call errors.

**RAG faithfulness** measures whether generated answers stay within the retrieved
context. **RAG context recall** measures whether all relevant context was retrieved.
**RAG context precision** uses a hotel-tuned Ragas prompt
([`eval/evaluators/_rag_prompts.py`](https://github.com/iannellis/Blue-Horizon-AI-Concierge/blob/main/eval/evaluators/_rag_prompts.py))
that credits a retrieved chunk as useful when it answers any one clause of a multi-part
question rather than requiring it to cover the whole question, and skips turns whose
reference is only the "nothing matched" sentinel. Its level relative to earlier runs
reflects that prompt change and the no-match skip, not a retrieval change. It remains
the lowest-scoring RAG metric after answer relevancy, and the
[sentinel-bundled compound reference issue](datasets.md#known-issue-sentinel-bundled-compound-references)
accounts for a measurable part of the remaining gap.

**Info filters pass rate** measures whether the information agent correctly honours
user-specified constraints (price, duration, booking requirements) when selecting and
presenting results. **Info reference subset pass rate** measures whether the expected
source documents appear in the set of documents retrieved for a given query.

## Latency (end-to-end, ms)

Mean of the same three runs, at a `max_concurrency` of 10.

| Route | p50 | p95 | p99 |
|-------|-----|-----|-----|
| Refuse | 1,135 | 1,671 | 2,359 |
| Info | 4,998 | 8,375 | 9,929 |
| Booking | 5,522 | 8,366 | 10,698 |

Refuse requests resolve fastest, since they are the router only. Info requests go
through the full RAG pipeline: parse, parallel retrieval, merge, respond. Booking
requests run NL-to-SQL search plus, on a propose call, an in-process pricing pass
against `room_availability`. The write itself happens later, on confirm, and is not
included in this per-turn latency.

Tail latency is the least stable figure here: booking p99 ranged from 9,055 ms to
13,167 ms across the three runs, against a p50 that moved by only about 400 ms. That
spread is provider-side variance on individual reasoning calls, not contention, since
these runs share a fixed concurrency limit. Treat p50 as the reliable number and p99 as
indicative.

## Next

- [Harness](harness.md) - how the evaluation runs and what it captures
- [Datasets and Baselines](datasets.md) - dataset schema, evaluators, baseline files
- [Stress Test](stress-test.md) - concurrency and double-booking under load
