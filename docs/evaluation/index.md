# Evaluation results

Metrics collected over the 241-case dataset,
[`eval/datasets/hotel_agent_eval_full.jsonl`](https://github.com/iannellis/Blue-Horizon-AI-Concierge/blob/main/eval/datasets/hotel_agent_eval_full.jsonl).

**Every figure below is the mean of three full runs** on 2026-11-04 (US time; the run
folders carry the UTC date, 2026-11-05), with the observed range across those runs
given alongside. All three ran the same agent models, prompts,
and configuration: the prompt and config hashes recorded in each run's `summary.json`
match. Single-run variance on the judge-scored metrics is large enough that a one-run
number is misleading, which is why the repository's full-eval baseline is also set from
a three-run average rather than a single run.

Raw outputs:
[`eval/outputs/Full-dataset-26-11-04/`](https://github.com/iannellis/Blue-Horizon-AI-Concierge/tree/main/eval/outputs/Full-dataset-26-11-04),
one subfolder per run.

!!! info "These numbers are a snapshot, not a target"
    They shift whenever the underlying models, prompts, or agent code change, and are
    expected to be updated regularly as the project evolves. The pinned regression
    thresholds in [`eval/baselines/`](datasets.md#baseline-files), not this table, are
    the source of truth CI checks against.

## What grades what

Every metric that requires a model judge or scorer - all three conversation-level scores
plus the four RAG metrics - is graded by Gemini `gemini-3.5-flash-lite`, configured in
[`eval/eval_config_full.toml`](https://github.com/iannellis/Blue-Horizon-AI-Concierge/blob/main/eval/eval_config_full.toml)
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
| Consumer quality (0-5) | 4.986 | 4.971 - 5.000 |
| Grounding (0-5) | 4.99 | 4.98 - 5.00 |
| Injection resistance (0-5) | 5.00 | 5.00 - 5.00 |

**Consumer quality** and **Grounding** assess response helpfulness and factual fidelity
to retrieved context. **Injection resistance** scores how reliably the system refuses
prompt-injection attempts, and was 5 on every conversation in all three runs. The third
run scored 5 everywhere on all three. Grounding dropped to 3 on two conversations in the
first run, and consumer quality dropped below 5 on six conversations in the first run and
two in the second:

- `case_0220` (first run, 3 on both), where the judge said the second turn listed the
  same rooms as the first although the guest had added a price cap.
- `case_0131` (first run, 4 on consumer quality, 3 on grounding), where the judge called
  a confirmed stay the third turn mentioned unsupported by anything earlier in the
  conversation.
- `case_0163` (second run, 3), where the judge said the third turn called a booking
  already confirmed when the guest had only just given dates.
- `case_0187` (first and second runs, 4), where the judge called the third turn confused
  because the booking was already confirmed.
- `case_0127`, `case_0189`, and `case_0004` (first run, 4), where the judge called a
  turn slower or wordier than it needed to be.

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
| Booking - tool call success rate | 99.5% | 99.3% - 99.6% | code |
| Booking - expected search pass rate | 100% | 100% - 100% | code |
| RAG faithfulness | 95.9% | 95.3% - 96.4% | `gemini-3.5-flash-lite` |
| RAG context recall | 99.0% | 98.3% - 99.5% | `gemini-3.5-flash-lite` |
| RAG context precision | 90.1% | 89.7% - 90.6% | `gemini-3.5-flash-lite` |
| RAG answer relevancy | 83.8% | 83.3% - 84.2% | `gemini-3.5-flash-lite` + `gemini-embedding-2` |
| Info filters pass rate | 100% | 100% - 100% | code |
| Info reference subset pass rate | 100% | 100% - 100% | code |

Database invariants (`db_no_double_booking`, `db_no_null_status`,
`db_confirmation_numbers_unique`, `db_confirmed_bookings_have_receipt`,
`db_no_available_with_null_price`), the injection tripwire, `moderation_flag_pass`
(moderation flagged exactly the turns labeled `expect_flagged`), and
`booking_no_unbacked_success_claims` passed on every case in all three runs. That check
accepts a confirm on the same turn or any earlier one as backing for a success claim,
since the agent is told to say a request the application already confirmed is done, and
it matches "already confirmed" and "already done" as well as "is booked".

!!! note "Turn-weighted here, case-weighted in the baseline"
    `summary.json` reports each metric two ways. `turn_based_summary` averages over every
    turn in the dataset, so a long conversation contributes more than a short one.
    `case_based_summary` averages the per-case scores, giving every conversation equal
    weight. The table above is turn-based, matching what this section claims to measure.

    [`eval/baselines/`](datasets.md#baseline-files) and `ci_check.py` use the
    **case-based** figures, so the pinned baseline values differ slightly from this
    table. For these runs: faithfulness 0.9645 case vs 0.9592 turn, precision 0.8910 vs
    0.9014, recall 0.9884 vs 0.9895, answer relevancy 0.8414 vs 0.8378, tool call success
    0.9919 vs 0.9953. Route accuracy, filters, expected search, and reference subset were
    1.0 both ways. The
    conversation-level judge scores above are case-based already, and match the baseline
    exactly.

**Route accuracy** measures how often the router correctly dispatches to the information
or booking agent, both, or refuses. Of 723 conversations (241 cases, three runs), no
turn was misrouted. The previous three runs misrouted one turn each, two of them
injection attempts that reached a sub-agent instead of being refused. Those runs predate
the injection check that now reads every guest message beside the router.

**Booking - expected search pass rate** checks the arguments of `search_rooms` calls on
the turns labeled with an `expected_search`, which are the 32 turns of the
attribute-search cases `case_0207` through `case_0225` and the split-routing cases that
search. Every labeled turn passed in all three runs. The labels accept only the
inclusive reading of a broad request, which `case_0216`'s "15th floor or higher with an
ocean view" missed once in the previous runs (see
[Design Goals and Decisions](../design-decisions.md#a-search-for-an-ocean-view-left-out-the-panoramic-ones)).

**Booking - no unexpected failure rate** measures whether a booking, cancellation, or
modification's success-or-failure outcome matched the case's expectation. A request for
an already-taken night is *expected* to fail, and that is not counted against the agent.
The check is two-sided: a write that succeeds on a turn labeled to fail is also a
mismatch, and so is any `propose_*` call on a turn labeled `expect_no_write`, which the
19 attribute-search cases, the search-only turns of the split-routing cases, and one
search-only turn of `case_0169` carry (see
[Datasets and Baselines](datasets.md#dataset-schema)). Every scored turn matched in all
three runs, between 256 and 263 turns per run, since a turn with no proposal and no `expect_no_write` label is not scored.

**Booking - tool call success rate** is stricter: it flags any `search_rooms`, `propose_*`,
or `confirm_booking` call that errored on a turn not expected to fail, even when the agent
recovered and the turn's outcome still matched. Four calls errored across the three
runs, all the same way: `case_0121`'s first proposal for "Feb 3 to Feb 5" in every run,
and `case_0164`'s for the same dates in the third, took a year outside the bookable
range. `booking.txt` states how to resolve a date given without a year, so the model
skipped a rule it had rather than lacking one. The tool refused each proposal with a
message naming the window, and the agent retried with the right year and booked the
room (see
[Design Goals and Decisions](../design-decisions.md#a-wrong-year-read-as-a-taken-room)).
The previous three runs showed this once.
The
ambiguous-column SQL errors that once held this metric below 100% came from
the model writing its own queries, and cannot recur now that room search is a fixed
parameterized query (see
[Design Goals and Decisions](../design-decisions.md#free-form-sql-had-no-bound-on-what-one-search-could-cost)).

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

The filters pass rate passed every turn in all three runs. `case_0005`'s "I only have
20-minute", read as an exact duration rather than a cap in two of the previous three
runs, is an intermittent miss the smoke run still showed. The rate was 96.0% before two changes to
`information_parser.txt`, both from cases that missed in most runs. `case_0037` states
one duration range for two items, and the
parser dropped it, since its duration rules handled a duration attached to each item
but not one stated once for the whole request, as its price and notice rules already
did. The prompt now says such a duration covers every item. `case_0023` asks for a
no-booking activity and adds that "breakfast info can be separate", and the parser
applied `booking_required: false` to the whole request, treating the set-aside topic as
outside it. The prompt now counts a topic the guest sets aside as an item, and says
every topic is an item for every filter rule, not only duration.

## Latency (end-to-end, ms)

Mean of the same three runs, at a `max_concurrency` of 10.

| Route | p50 | p95 | p99 |
|-------|-----|-----|-----|
| Refuse | 1,637 | 2,297 | 2,507 |
| Info | 5,497 | 9,128 | 11,233 |
| Booking | 5,842 | 8,025 | 9,208 |
| Booking + info | 5,824 | 8,443 | 8,795 |
| Info + booking | 5,846 | 6,166 | 6,176 |

Refuse requests resolve fastest, since they are the router only. Info requests go
through the full RAG pipeline: parse, parallel retrieval, merge, respond. Booking
requests run a room search plus, on a propose call, an in-process pricing pass
against `room_availability`. The write itself happens later, on confirm, and is not
included in this per-turn latency. A split turn runs both agents at once, so it takes
about as long as the slower of the two. Its rows rest on the few turns of twelve cases,
so their tails are the least reliable figures here.

Tail latency is the least stable figure here: booking p99 ranged from 8,779 ms to
9,714 ms and info p99 from 10,499 ms to 12,279 ms across the three runs, against p50s
that moved by under 140 ms. That
spread is provider-side variance on individual reasoning calls, not contention, since
these runs share a fixed concurrency limit. Treat p50 as the reliable number and p99 as
indicative.

## Next

- [Harness](harness.md) - how the evaluation runs and what it captures
- [Datasets and Baselines](datasets.md) - dataset schema, evaluators, baseline files
- [Stress Test](stress-test.md) - concurrency and double-booking under load
