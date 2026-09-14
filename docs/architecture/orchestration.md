# Orchestration

![Orchestration diagram](../images/orchestration.png)

The orchestration layer is a LangGraph state graph. A router LLM classifies each
incoming message and dispatches it to the appropriate sub-agent, or refuses it if it is
out of scope. Conversation history is maintained in memory via LangGraph's `MemorySaver`
checkpointer, keyed by `thread_id`.

## Routing

The router is a single LLM call driven by
[`system_prompts/orchestration.txt`](https://github.com/iannellis/Blue-Horizon-AI-Concierge/blob/main/blue_horizon/system_prompts/orchestration.txt).
It emits one of three routes:

| Route | Meaning |
|---|---|
| `info` | Hotel services, amenities, policies, FAQ content |
| `booking` | Room availability, pricing, and reservation changes |
| `refuse` | Out of scope, or a prompt-injection attempt |

Refusals never reach a sub-agent. They return `[orchestration.messages].refusal`
verbatim, so a refusal costs one LLM call rather than a full pipeline: this is why the
refuse route's p50 latency is roughly a quarter of the other two.

### The info/booking boundary is the hard part

Most of the router prompt's length exists to handle cases that look like one route and
belong to the other. A question about how many floors the hotel has reads like a rooms
query but is an FAQ lookup. A question about modifying a reservation *without penalty*
is a policy question, not a modification request. A question about whether a chartered
yacht includes crew is a service-details lookup, not a request for the assistant to act
as crew.

These were found empirically: each one is a case in the eval dataset that misrouted,
was diagnosed, and produced either a new rule or a worked example in the prompt. Route
accuracy is measured on every eval run precisely because this boundary is where
regressions show up first. See
[Design Goals and Decisions](../design-decisions.md#routing-rules-are-earned-not-designed).

## Concurrency and timeouts

`[orchestration.orchestration]` in
[`app_config.toml`](https://github.com/iannellis/Blue-Horizon-AI-Concierge/blob/main/blue_horizon/app_config.toml)
bounds every stage:

| Setting | Default | Purpose |
|---|---|---|
| `router_timeout_s` | 30 | Wall-clock cap on the router node |
| `info_timeout_s` | 60 | Wall-clock cap on the information node |
| `booking_timeout_s` | 60 | Wall-clock cap on the booking node |
| `llm_concurrency` | 15 | Maximum concurrent pipeline executions |

`llm_concurrency` is a semaphore, not a queue limit. It exists because the stress test
drives 50 concurrent sessions against a single OpenAI account: without it, a traffic
spike exhausts the tokens-per-minute quota and every in-flight request fails together
rather than a few waiting their turn.

Initialization uses `tenacity` with exponential backoff between `init_retry_base_s` and
`init_retry_max_s`, so a cold Redis or a suspended Neon compute delays startup instead
of failing it. A pending wait is not cut short when a dependency comes back, so
`init_retry_max_s` (10 seconds) is also the worst-case delay between a dependency
recovering and the API becoming ready. A dependency that stays down logs its traceback
on the first failed attempt only, then one line per retry.

## Failed turns

A turn fails when the router or a sub-agent exceeds its timeout above, raises, or the
turn ends without a reply, or when the booking agent could not reach its database. Every
failure site records a `turn_error` of `"timeout"`, `"unavailable"`, or `"internal"` in
the graph state instead of writing a reply:

| Site | Records |
|---|---|
| Router timeout | `timeout` |
| Router exception, or the router choosing the `error` step | `internal` |
| Sub-agent timeout (`info_timeout_s`, `booking_timeout_s`) | `timeout` |
| Sub-agent exception | `internal` |
| Booking agent exception caused by `BookingUnavailableError`, raised by a booking tool that could not reach the database | `unavailable` |
| Booking agent returned, but a `run_sql` result this turn carried `error_kind` `unavailable` | `unavailable` |
| No final assistant message for the turn | `internal`, set by `finalize` |

`finalize` then removes the failed turn from history entirely. The manager reads
`turn_error` from `finalize`'s own output, emits an `error` event instead of `done`, and
invalidates any proposal the turn created. The router clears `turn_error` at the start
of every turn, so a failure still held in the checkpoint from an earlier turn is never
reported again. The event's message is `[orchestration.messages].database_unavailable`
for `unavailable` and `[orchestration.messages].error` for the other codes.

The two `unavailable` rows cover the two ways a booking tool reports an outage.
`list_my_bookings` and the `propose_*` tools raise `BookingUnavailableError`, which
`create_agent` does not catch, so it escapes the booking agent. `run_sql` instead returns
its outage as a result, so the second row discards a reply the agent did produce. After a database
outage the model writes its own account of it, which would reach the guest as an
ordinary answer, free to invite a retry in its own words, and without the Send again
button. The booking dispatch node reads the `error_kind` of the turn's `run_sql` results,
never the reply text, and only results since the turn's own guest message count.

A router or sub-agent exception whose cause chain contains a `BookingUnavailableError` or
a network failure (a LangChain `ModelConnectionError` or `ModelTimeoutError`, or any
`OSError` such as a failed DNS lookup) is logged as one warning line naming the root
cause. Any other exception is logged at ERROR with its full traceback.

Two things follow. A failure can no longer reach a client looking like an ordinary
reply, which is what previously kept the UI from offering its Send again button. And the
router never reads an apology in history as something the concierge said. See
[API Reference](../api.md#streaming-events) for the event shape, and
[Design Goals and Decisions](../design-decisions.md#a-failed-turn-was-reported-as-a-successful-one)
for the reasoning.

## Readiness

`OrchestrationManager` tracks a three-state `Readiness` enum, not a boolean, because
"not ready yet" and "will not become ready by itself" call for different guest-facing
copy and, for the latter, no retry affordance at all:

| State | Entered when | `/v1/chat`, `/v1/customers`, `/v1/bookings` |
|---|---|---|
| `READY` | The compiled agent is built | Served normally |
| `STARTING` | No agent yet, and the last attempt (if any) failed for a reason classified transient | `503` + `Retry-After`, `[orchestration.messages].unavailable` - no retry instruction, since recovery is expected on its own |
| `FAILED` | No agent yet, and the last attempt failed for a reason classified **permanent** | `503` + `Retry-After`, `[orchestration.messages].failed` - explicitly says this will not resolve on its own |

`is_ready` stays available as a derived `READY`-or-not property for callers (like
`/v1/health`) that only need a yes/no answer; `readiness` exposes the full state for
callers that must distinguish `STARTING` from `FAILED`.

**Classification, not the retry loop, is what changes.** The init loop keeps
`stop_never` in both non-ready states: an externally-fixed dependency (a corrected
database role, a Neon compute that finishes waking up) should recover without an
operator restart, even after being classified `FAILED` once. What differs is only the
message shown while waiting. Classification comes from the exception type the last
init attempt raised:

- `ConfigurationError` (a sibling of `OperationalError`, not a subclass, so existing
  `except OperationalError` handlers do not swallow it) marks the failure permanent and
  sets `FAILED`. Raised only from genuinely unrecoverable sites: `startup_check`'s
  read-only-role guard (see [Booking agent](booking-agent.md#1-two-database-roles)),
  a missing packaged prompt file, a missing or blank required database URL, Redis
  rejecting the credentials in `REDIS_URL`, and Postgres rejecting the password in
  either database URL. The Postgres check opens one direct connection per URL before
  the pools, because a pool retries a rejected password internally and a checkout
  only ever reports `PoolTimeout`, which looks the same as an outage. libpq gives that
  rejection no SQLSTATE, so it is matched by its message.
- Everything else - `OperationalError`, or an exception type the classifier does not
  recognise - sets `STARTING`. Defaulting an unrecognised exception to `STARTING`
  rather than `FAILED` is deliberate: a misclassification then degrades to bad copy
  (a permanent failure described as transient) rather than to a guest being told a
  transient outage will never resolve, and the process never gives up regardless.

A `/v1/chat` request is gated on `is_ready` before either content-negotiated branch
commits to a response, since a `StreamingResponse` commits its `200` as soon as it
starts and cannot become a `503` afterward. See
[API Reference](../api.md#readiness-503) for the wire-level response shape, and
[Design Goals and Decisions](../design-decisions.md#asking-the-guest-to-try-again-was-hiding-three-different-failures)
for why "please try again" was replaced by this instead of by better copy alone.

## Session isolation

Each conversation is keyed by a UUID `thread_id`. The checkpointer stores that thread's
message history and booking context, so concurrent sessions cannot see each other's
state. A `thread_id` is additionally bound to whichever `customer_id` first used it, and
the API rejects a later request that replays the same `thread_id` under a different
guest with `409 Conflict`.

## Streaming

The manager translates LangGraph's internal event stream into the coarse, user-facing
stage labels the UI displays inside the assistant bubble. The full SSE protocol is
documented in the [API Reference](../api.md#streaming-events).
