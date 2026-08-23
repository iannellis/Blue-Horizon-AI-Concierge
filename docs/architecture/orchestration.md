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
of failing it.

## Session isolation

Each conversation is keyed by a UUID `thread_id`. The checkpointer stores that thread's
message history and booking context, so concurrent sessions cannot see each other's
state. A `thread_id` is additionally bound to whichever `customer_id` first used it, and
the API rejects a later request that replays the same `thread_id` under a different
guest with `409 Conflict`.

`POST /v1/reset` clears the checkpointer along with the proposal store and the
thread-to-customer registry, because resetting the Neon branch invalidates every
`booking_id` and confirmation number the process is still holding in memory.

## Streaming

The manager translates LangGraph's internal event stream into the coarse, user-facing
stage labels the UI displays inside the assistant bubble. The full SSE protocol is
documented in the [API Reference](../api.md#streaming-events).
