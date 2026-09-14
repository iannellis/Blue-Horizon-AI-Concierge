# Design goals and decisions

This page records what the project set out to do, the constraints that shaped it, and
the decisions that changed direction along the way. Each decision below was made in
response to something that actually went wrong or actually did not work, and most of
them are traceable to a specific commit.

## Goals

**Build a hotel concierge that a guest can trust with a reservation.** Two capabilities
are in scope: answering questions about the hotel, and searching for and changing room
bookings. Everything else is refused.

**Make correctness structural rather than instructed.** A system prompt asking a model
not to double-book is not a safeguard. Every guarantee in this project should hold even
if the model behaves as badly as it possibly could within its tool surface.

**Measure it continuously.** Agent behavior regresses silently when a prompt, a model,
or a library version changes. Quality claims should be backed by a metric with a pinned
baseline that CI can fail on.

## Explicit non-goals

| Out of scope | Why |
|---|---|
| Real guest accounts and authentication | Identity is a demo affordance; guests are assigned from a seeded pool. Adding real auth would add surface without demonstrating anything new. |
| Payment processing | A confirmation number is the terminal state. Nothing here needs PCI scope. |
| Multi-property support | One hotel keeps the schema and the retrieval corpus small enough to reason about. |
| Model fine-tuning | The interesting problems here are orchestration, grounding, and transactional safety, none of which fine-tuning addresses. |

## Constraints that shaped the build

- **The deployment target is a HuggingFace Space that sleeps.** Cold starts are normal,
  so every external dependency needs a bounded retry policy rather than an assumption
  that it is already warm.
- **Neon suspends compute after roughly 300 seconds of inactivity.** The connection
  pool's `max_idle_s` is set to 240 to drop stale connections before the server kills
  them, and real DB-backed reads use a 20-second timeout rather than the health check's
  3 seconds.
- **The database must be resettable.** Anyone can click a button and put the demo back
  to a known state, which means Neon branch resets are a first-class operation and
  nothing may hold a `booking_id` across one.
- **One OpenAI account serves everything, including a 50-session stress test.** Hence a
  concurrency semaphore rather than unbounded parallelism.

---

## Decisions

### The information agent is a workflow, not an agent

**Status:** adopted, replacing the original implementation.

The information agent was first built with LangChain's `create_agent`, exposing the
retrieval tools and letting the LLM decide which to call.

The RAG tools apply the user's parsed constraints as filters on the vector search. When
nothing matches, the tool returns an empty result, which is the *correct* answer. But
the model read an empty tool result as a malfunction and would retry multiple times with
the same result.

The observation that resolved it: there was never a real decision to make. Every info
turn searches all three indices, in parallel, with the parsed filters applied. Once that
became a fixed LangGraph DAG instead of a tool-calling loop, an empty retrieval became
an empty slice of the merged context rather than an error signal, because no component
was left in the loop whose job was to interpret it.

Secondary benefits: exactly two LLM round trips per turn instead of an unpredictable
number, and correspondingly tighter latency variance.

*Generalisation:* use an agent when the model genuinely must choose. When the sequence
is knowable in advance, a workflow is cheaper, faster, and removes an entire category of
misinterpretation.

### The model may propose a booking, never commit one

**Status:** adopted, replacing the original implementation. Commit `5bc6d5b`.

The booking agent's only tool used to be `run_sql` with `UPDATE` privileges. Every
correctness property - atomicity, ownership checks, partial-booking prevention - lived
as prose in a 700-line system prompt, with no reservation record to check the model's
success claims against. The model could report a successful booking that never happened,
and nothing in the system could tell.

The rework split this in two. The model searches and proposes; the application prices,
validates, commits, and authors the receipt. The
[Booking agent](architecture/booking-agent.md#the-model-proposes-the-application-decides)
page documents all four enforcement layers.

The load-bearing detail is that the confirmation dialog is built from the proposal's
stored fields, never from the assistant's text. If the model describes a booking
differently from what it proposed, the guest sees the proposal, not the description.

### Least privilege is a database grant, not a code path

**Status:** adopted. Commit `a86c845`.

The SQL guardrail existed first: an AST allowlist restricting the model to `SELECT` on
two tables. It is good, and it is not sufficient, because it is code in the same process
as the thing it constrains.

Splitting the single agent role into `bh_agent_ro` and `bh_agent_rw` moved the same
allowlist into Postgres grants. The guardrail stayed, now redundant by design. If the
guardrail has a parsing bug, Postgres still refuses. `startup_check` fails the process
if a trial write through the read-only URL succeeds, so the two-role split cannot be
silently undone by a misconfigured environment variable.

An unexpected consequence worth recording: `reload_sql_tables` drops and recreates
tables, which drops their grants. The regrant was a separate manual script, and it was
duly forgotten - leaving the Parent branch with zero grants and every branch reset since
silently inheriting `InsufficientPrivilege` failures across the entire `db_integration`
suite. The fix was to make the regrant the last step of the reload, in the same
transaction, so the two cannot come apart again (commits `a8e0f44`, `1c1ba1d`).

*Generalisation:* a safety property enforced by two independent mechanisms is worth the
redundancy, but only if neither can be removed without the other noticing.

### Double-booking prevention moved from detection to impossibility

**Status:** adopted. Commit `72eb3f8`.

The original guarantee was `SELECT ... FOR UPDATE` locking plus an after-the-fact
invariant check that the stress test ran at the end of each run.

Two problems. First, the invariant check was originally a tautology: it grouped
`room_availability` by `(room_id, date)` looking for duplicates, which that table's own
`UNIQUE` constraint already makes impossible regardless of what the agent did. It could
never have gone red. Rewriting it to query `booking_rooms` directly - and confirming it
went red against a hand-inserted overlapping pair before trusting it - was the actual
fix (commit `e67fb9b`).

Second, detection after the fact is the wrong shape for this problem. A GiST exclusion
constraint, `booking_rooms_no_overlap`, makes an overlapping pair impossible to insert
in the first place. The detection query survives as an audit signal in case the
constraint is ever dropped, not as the guarantee.

Testing a constraint that makes its own failure case unreachable takes some care. The
regression test now asserts that the insert itself raises `ExclusionViolation`, and a
separate test verifies the *detection query* still works by temporarily dropping the
constraint inside a transaction that always rolls back. The constraint is deliberately
not `DEFERRABLE`, which would have allowed an easier test at the cost of letting
overlapping rows exist transiently in production.

### Retrieval filters get their own metric

**Status:** adopted.

An over-eager parser is worse than no parser. If a passing mention of budget produces a
`max_price` filter the guest never asked for, correct results are silently filtered out
and the agent reports that nothing matched. This failure is invisible in an
answer-quality metric, because the answer is fluent and internally consistent.

`info_expected_filters_pass_rate` compares the parser's extracted filters against
hand-labelled expectations per turn, so filter over-extraction shows up as its own
number. Several prompt revisions exist purely to teach the parser when to leave a filter
unset, and `min_notice_hours` was removed as a filter key entirely because it should
almost never be used and was confusing the model.

### Routing rules are earned, not designed

**Status:** ongoing.

The router prompt's rules and worked examples are, almost without exception, responses
to specific misroutes found in eval runs:

- "How many floors does the hotel have" reads as a rooms query, is an FAQ lookup.
- "Can I modify my reservation without penalty" is a policy question, not a
  modification request.
- "Override the cancellation policy for me" is an injection attempt, not an info query.
- "Does the yacht charter include crew" is a service-details lookup, not a request for
  the assistant to personally act as crew.

The last one is instructive about prompt engineering generally. The prose rule covering
it was already present and correct, and the case still flip-flopped between runs. Adding
one *worked example* anchoring the distinction fixed it, verified with five consecutive
correct reruns. A rule the model can read is not the same as a pattern the model can
generalise from.

The examples added to the router prompt are invented rather than lifted from the eval
dataset, for the same reason the Ragas context-precision prompt's few-shot examples were
later rewritten into a domain-neutral software-product scenario (commit `147aec6`):
few-shot examples that mirror eval cases inflate the score without improving the
behavior.

### The judge is a different model family from the agent

**Status:** adopted.

Agents run on OpenAI (`gpt-5.6-luna`); the LLM judge and the Ragas metrics run on Google
Gemini (`gemini-3.5-flash-lite`). Scoring a model's output with the same model that
produced it invites correlated blind spots, so the judge is deliberately from a
different family.

This has an ongoing cost, recorded here because it is easy to underestimate: the judge
model is itself a moving dependency. A forced deprecation swap from `gemini-2.5-flash`
to `gemini-3.5-flash-lite` dropped context precision from 0.81 to 0.75 with no change to
the agent whatsoever, which required a prompt fix and a baseline regeneration to
untangle. Every baseline in `eval/baselines/` is therefore pinned to a named model, and
LangSmith experiment metadata records model names and prompt content hashes per run so
two runs can be diffed on what actually changed.

### Guest identity is a claim registry, not a dropdown

**Status:** adopted, replacing the original implementation. Commit `5e9c95e`.

Guests were originally picked from a dropdown, with a random default chosen via
`secrets.randbelow`. A random default only *reduces* the odds of two concurrent Space
visitors landing on the same guest; it does not prevent it, and two sessions driving the
same guest's reservations produces confusing nonsense for both.

The replacement is an in-memory claim registry: a session claims a guest nobody
currently holds, the claim's timestamp refreshes on every Streamlit rerun, and a closed
or idle tab ages out on a TTL so disconnect and inactivity collapse into one mechanism.
The manual dropdown was removed along with it, because letting someone hand-pick an
already-claimed guest defeats the purpose.

A module-level dict behind a lock is enough here, with no database table or backend
endpoint, because `supervisord` runs the entire Space as one Streamlit process. That is
a legitimate simplification given the deployment topology, and it is written down
because it would be wrong under any multi-process deployment.

### Loading only the seeded guests corrupted the availability data

**Status:** fixed. Commit `180648e`.

The loader originally inserted only the 25 richest customers, because those are the ones
the UI offers. Every pre-existing booking belonging to any other customer then failed
its foreign key and was silently dropped: 14,535 of 14,617 accepted bookings never made
it into the database. `room_availability` was left marked `Booked` across large ranges
with no matching reservation anywhere.

The load now inserts every source customer. The richest `seeded_customer_count`
still get a dense `customer_id` block of `[1, seeded_customer_count]` and are loaded
first, so the UI picker and the eval harnesses' guest selection keep working unchanged;
everyone else follows with arbitrary higher ids.

That count had been duplicated as a hardcoded constant in four separate files with no
mechanism keeping them in sync. It is now
`AppConfig.load_data.booking_pgsql.seeded_customer_count`, threaded as an explicit
parameter to every function that needs it.

*Generalisation:* a silent foreign-key drop during a bulk load is close to invisible.
The reconciliation pass that now runs after the load - checking `room_availability`
against `booking_rooms` in both directions - exists because the corruption was only
found by noticing the availability data looked implausible.

### Test databases are separated from production by construction

**Status:** adopted. Commits `ec21ce9`, `c4a7e20`.

A local `.env` points `PGSQL_RW_DB_URL` and `PGSQL_RO_DB_URL` at the Production Neon
branch, because that is what running the app locally needs. Running the
`db_integration` suite with that environment would book and cancel real rows there.

`tests/conftest.py` copies the `_EVAL`-suffixed variables over their plain-named
counterparts before any test runs, so a local run lands on the Development branch
instead - the same substitution CI performs from its own secrets.

Separately, `PGSQL_ROOT_DB_URL` was split into two variables with deliberately different
names. `PGSQL_ROOT_PARENT_DB_URL` addresses the Parent branch and is read only by the
data loader; `PGSQL_ROOT_DB_URL` addresses whatever branch a test run targets.
`reload_sql_tables` must never run against a branch that gets reset in place, and two
similar names are much harder to confuse than one name with two meanings.

### The `json_safe` defect and its downstream cleanup code

**Status:** fixed. Commit `f4b7568`.

`eval/_utils.json_safe` checked `isinstance(value, Iterable)` before it checked for
`model_dump`. Pydantic v2's `BaseModel` is itself iterable - it yields `(field_name,
value)` pairs - so every model written to `results.jsonl` (every LangSmith
`EvaluationResult`) was serialized as a list of `[name, value]` pairs, one pair per
field including every unset field as `[name, None]`, instead of as a plain object.

Roughly 600 lines of code existed solely to read that shape back: duck-typed `Protocol`
shims, a dedicated pair-list adapter, and a dict-comprehension unwrap in
`analyze_results.py`. Reordering `json_safe`'s branches - `model_dump` before
`Iterable` - fixed the shape at the source, and most of that compensating code was
deleted rather than kept as a second, now-redundant reader.

`results.jsonl`'s on-disk shape for `evaluators.results.value` changed: a list of plain
`{"key": ..., "score": ..., ...}` dicts instead of a list of `[[name, value], ...]`
pair-lists. **The pinned baselines in `eval/baselines/` were deliberately left
unregenerated.** The bug only ever affected the JSON envelope around each result - the
`score` value for every metric was correctly present in both the buggy and fixed shapes,
and `parse_metrics`/`ci_check.py` only ever read `key` and `score` off each entry, never
the spurious `None` fields the bug added. A fresh run's baseline numbers would be
statistically indistinguishable from the existing ones, so regenerating them would
spend real API and Neon time to reproduce the same thresholds already pinned.

*Generalisation:* `isinstance` branch order matters when a type checked earlier in the
chain (`Iterable`) is a superset of a type checked later (has `model_dump`). Pydantic
models being iterable is not obvious from reading `BaseModel` usage elsewhere in the
codebase, which is exactly why this shipped unnoticed until the on-disk shape was
inspected directly.

### The Production guard was a no-op on a plain local run

**Status:** fixed. Commit `162445d`.

The override described in
[Test databases are separated from production by construction](#test-databases-are-separated-from-production-by-construction)
assumed the `_EVAL`-suffixed environment variables would already be present by the time
`pytest_configure` ran. On a plain local `pytest` invocation nothing loads `.env` that
early - `pytest_configure` fires before test collection, before any test module's own
`load_dotenv()` call has had a chance to run - so it always found every `_EVAL` variable
unset and overrode nothing.

`PGSQL_RW_DB_URL`/`PGSQL_RO_DB_URL` still ended up populated, just later and by
accident: whichever `db_integration` test happened to import a module with its own
module-level `load_dotenv()` call first (`tests/api/test_app.py` importing
`blue_horizon.api.app`) filled them in directly from `.env`'s literal values -
Production, by design elsewhere in this same file - since `load_dotenv()` only fills in
variables that are still unset and the override's one chance had already passed.
`PGSQL_ROOT_DB_URL` has no literal entry in `.env` at all, so it just stayed unset and
its one gated test skipped, every time, on every machine - the same visible symptom that
led to noticing the invisible one.

Root-caused by running `pytest -m db_integration` locally: the root-gated test skipped
as expected, but the tests that "passed" turned out to have written real rows to
Production. Fixed by having `pytest_configure` call `load_dotenv()` itself, first,
closing the window entirely rather than depending on import order.

*Generalisation:* a safety mechanism that depends on execution order it does not control
is not a mechanism, it is a coincidence that held until something reordered it. The fix
makes the guard's precondition (`.env` loaded) something it establishes itself rather
than something it assumes.

### Asking the guest to try again was hiding three different failures

**Status:** adopted.

The UI used to show one string, "please try again," for nearly every failure: a router
timeout, a sub-agent exception, a startup window, a lost race for a room, a database
outage. Investigating where that copy actually came from turned up three separate
reasons it was wrong, not one:

1. **It asked the guest to do work the system could already do itself.** The two
   dominant triggers - a HuggingFace Space cold start and a Neon compute resume - are
   both self-healing and time-bounded. Nothing needed the guest's help; the system just
   was not telling them that.
2. **On the confirm path, it invited an action whose safety had never been
   established.** The proposal used to be retired *before* the commit ran, so a retry
   after a failed confirm returned a flat `404 "That request has expired"` - the guest
   who tried to comply with "try again" was punished for it.
3. **It was a catch-all, so it destroyed information the system already had.** A `409`
   carrying the precise reason ("Room 204 is not available for every night requested")
   was collapsed into the same sentence as a network error, discarding the one detail
   that would have actually told the guest something.

**The fix classifies failures by whether a repeat is known to be safe, not by how the
error happened to surface:**

- **Idempotent reads** (`run_sql`, `/v1/bookings`, `/v1/customers`, `/v1/health`): the
  system absorbs the failure itself. A guest is never asked to retry a read; a failed
  one is a system state, reported and retried automatically, not a user error.
- **The chat turn**: not auto-retried server-side, because a turn may already have
  mutated the `ProposalStore`, appended a message to the checkpoint, and consumed a
  concurrency slot. The UI offers a resend action instead of a sentence asking the
  guest to retype, and only enables it once a health poll confirms recovery.
- **The confirm write**: never blind-retried. The fix here is not copy at all, but
  making a client-initiated retry actually safe - see below.

**The false `404` on a confirm retry is fixed by changing what "retire the proposal"
means, not by rewording the failure.** A proposal is now retired only once the write's
outcome is actually known: a determinate refusal (`BookingWriteError`, e.g. the nights
were taken) retires it, but an unreachable database (`BookingUnavailableError`) leaves
it pending, so a second confirm is a genuine retry. The case this alone does not solve
- the `COMMIT` landing but the acknowledgment being lost, so a naive retry re-prices,
finds the guest's own booking, and reports the room as unavailable to the guest who
actually got it - is closed by a reconciliation read
(`write_ops.find_booking_for_rooms`) that is exact, not heuristic, because of the same
exclusion constraint that makes double-booking structurally impossible elsewhere in
this system. See
[Booking agent](architecture/booking-agent.md#3-the-proposeconfirm-flow) for the
mechanism.

**Startup keeps its unbounded retry loop; only the guest-facing claim about it is
bounded.** A `STARTING`/`FAILED` readiness state machine replaces a boolean, so the
guest sees "this will resolve on its own" for an ordinary cold start and "this will not
resolve on its own" for a genuine misconfiguration - but the initialization loop itself
never gives up in either case, because an externally-fixed dependency should still
recover without an operator restart. Giving up on a Space that sleeps would be worse
than continuing to retry it. See
[Orchestration](architecture/orchestration.md#readiness) for the state machine and
the transient/permanent classification.

Mechanism lives in the architecture pages linked above; this entry exists to record the
reasoning and the tiering, per this file's own rule against duplicating the same fact in
two places.

### A failed turn was reported as a successful one

**Status:** adopted.

The tiering above promised that a failed chat turn offers a resend action. Manual
verification with the network cable unplugged showed it did not: the guest saw "Sorry -
that request could not be completed. You can send it again." with no Send again button,
and a sidebar still reading "Chatbot: Online".

The copy was right and the delivery was wrong. Every failure the graph caught itself - a
router or sub-agent timeout or exception, an empty turn - was written into history as an
ordinary assistant message and streamed as a normal `done` event. Only an exception that
escaped the graph became an `error` event, and only an `error` event triggers the
resend. So the most common failures, the ones a dropped network actually produces, were
indistinguishable from a reply. The apology also stayed in the checkpoint, where the
router read it on the next turn as something the concierge had said.

**The failure is now structured state, not prose.** Each failure site records a
`turn_error` code instead of an apology, the failed turn is dropped from history, and
the manager emits an `error` event in place of `done`. Matching the reply text against
`[orchestration.messages].error` in the UI was rejected: it couples the client to
wording, the same fragility `eval/stress/workload.py`'s prose fallback already carries.
The eval and stress harnesses read the code too, since with the failed turn gone from
history, "the last AI message" would otherwise have been an earlier turn's reply.

**The sidebar badge was renamed rather than made to probe dependencies.** `/v1/health`
reports readiness, meaning startup finished, and deliberately touches no dependency: the
UI polls it every few seconds, and a database probe on that cadence would keep the Neon
compute from ever scaling to zero and would wake an archived branch. So the badge now
says "Service: Running", which readiness can honestly claim, and an outage during a
conversation surfaces through the failed turn itself. See
[Orchestration](architecture/orchestration.md#failed-turns) for the mechanism.

### The model's own account of a database outage reached the guest

**Status:** adopted.

Manual verification with the booking database blocked produced "The booking system is
temporarily unavailable, so I couldn't check which Presidential Suites are available for
February 21, 2025 (1 night). Please try again shortly." The `run_sql` retry and its
`DATABASE_UNAVAILABLE` tool result worked as designed. The reply did not: the model added
the retry request itself, and the turn completed normally, so the guest saw an ordinary
answer with no Send again button.

**The reply is discarded, not reworded.** A prompt rule forbidding retry language was
rejected: it depends on the model complying on every run, and it would still leave an
outage looking like an answer. Instead the booking dispatch node reads the `error_kind` of
the turn's `run_sql` results and, on `unavailable`, records a `turn_error` of
`unavailable`. The outage then takes the same failed-turn path as a timeout, with
application-authored copy and a JSON `503`. The cost is that a turn which found
something useful before the outage loses that answer, which is acceptable for a turn the
guest has to resend anyway. See
[Orchestration](architecture/orchestration.md#failed-turns) for the mechanism.

### A booking tool's database outage was reported as an internal failure

**Status:** adopted.

The entry above covered `run_sql`, which returns its outage as a result. The other
booking tools raise instead: `list_my_bookings` raised `BookingUnavailableError`, and the
`propose_*` tools, whose pricing queries were not wrapped at all, raised a raw
`PoolTimeout`. `create_agent` does not catch tool exceptions, so each escaped the booking
agent and was recorded as `internal`. The guest got generic copy, a JSON client got `502`
instead of `503`, and the log got a full traceback, since `PoolTimeout` is a
`psycopg.OperationalError` rather than one of the network errors the log shortens.

**Every booking-database outage now takes one path.** The remaining queries are wrapped so
they raise `BookingUnavailableError` like every other `write_ops` read, and a failed node
whose cause chain holds that error records `unavailable` and logs one line. Catching the
error inside each tool and returning an error result was rejected: it would hand the
model an outage to narrate, which is the problem the entry above removed.

### Tooling choices

| Choice | Replaced | Reason |
|---|---|---|
| `sqlglot` | `sqlparse`, and regex before that | Real AST parsing. Three tests that pass under `sqlglot` fail under the regex guardrails, including quoted enum values such as `{"Ocean, View",Suite}`. |
| `tenacity` | Hand-written retry loops | Four separate bespoke backoff implementations existed across the codebase. |
| `pandera` | Manual DataFrame schema management | Declarative validation on the load path, plus vectorised transforms in place of row loops. |
| `uv` | Poetry | Faster resolution and installs; `uv_build` as the PEP 517 backend once nothing else depended on Poetry's tooling. |
| Workflow DAG | `create_agent` | See [above](#the-information-agent-is-a-workflow-not-an-agent). |

### Known open issue: sentinel-bundled compound references

Context-precision scoring skips turns whose reference is *entirely* the "nothing
matched" sentinel, because a bare refusal gives the judge no content against which to
assess a retrieved chunk's usefulness.

A compound reference that *mixes* the sentinel with a real answer - a multi-part
question where only one clause had no match - is currently left eligible, on the grounds
that the other clause is real content. In practice those turns take the largest
individual precision hits in the dataset.

The proposed fix is to strip sentinel segments from the reference before scoring rather
than skipping the whole turn. The segment-splitting logic already exists; filtering to
non-sentinel segments is the same parsing with a different reducer. This is tracked in
[Datasets and Baselines](evaluation/datasets.md#known-issue-sentinel-bundled-compound-references).
