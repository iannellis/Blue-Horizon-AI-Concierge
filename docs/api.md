# API reference

The FastAPI backend runs on port `8000` and exposes the following endpoints under the
`/v1` prefix.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/v1/health` | Returns `{"status": "ok"}` (200) when ready, `{"status": "starting"}` or `{"status": "failed"}` (503) otherwise |
| `GET` | `/v1/customers` | Lists the seeded guests as `{customer_id, first_name, last_name}`, for guest assignment. 503 during the startup window or while the database is unreachable |
| `GET` | `/v1/bookings?customer_id=` | Lists one guest's reservations (confirmation number, rooms, dates, total, status). 503 during the startup window or while the database is unreachable |
| `POST` | `/v1/chat` | Send a message. Content-negotiated: `Accept: text/event-stream` streams SSE events, anything else returns one JSON response. 503 while not ready |
| `POST` | `/v1/booking/confirm` | Commit a pending proposal. The only path that ever writes a booking, cancellation, or modification |
| `POST` | `/v1/booking/dismiss` | Discard a pending proposal without writing anything |

`/v1/chat/stream` from earlier versions has been retired in favor of content negotiation
on `/v1/chat`.

## Chat request body

```json
{
  "thread_id": "<uuid>",
  "customer_id": 7,
  "text": "Do you have any rooms available this weekend?"
}
```

`thread_id` is bound to whichever `customer_id` first uses it. A later request reusing
that `thread_id` under a different `customer_id` is rejected with `409 Conflict`.

## Readiness (503)

Both content-negotiated branches of `/v1/chat` are gated on the orchestrator's
readiness, checked before either branch commits to a response. While not ready, the
endpoint returns:

```json
{
  "status": "starting",
  "message": "The system is starting up and should be ready again shortly. No action is needed on your part.",
  "retry_after_s": 5
}
```

with HTTP `503` and a matching `Retry-After` header, for **both** an SSE request and a
plain JSON request - an SSE client gets this same JSON body instead of a stream, since
there is nothing yet to stream. `status` is `"starting"` while the system is expected to
recover on its own, or `"failed"` if the last startup attempt was classified as a
permanent misconfiguration; either way the client should treat the response as
retryable and wait `retry_after_s` seconds, since the background init loop keeps
retrying even in the `"failed"` case. See
[Orchestration](architecture/orchestration.md#readiness) for what drives the
classification.

`GET /v1/customers` and `GET /v1/bookings` return the same shape (503 with
`Retry-After`, `detail` in place of `message`) during the same startup window, since
both read from the booking database pool. They also return it after startup whenever a
database connection cannot be obtained, for example during a network outage or a Neon
compute resume, with `detail` saying the database could not be reached.

## Streaming events

Requesting `/v1/chat` with `Accept: text/event-stream` emits SSE events, plus periodic
`: keepalive` comment lines so a slow turn does not trip a reverse proxy's idle timeout:

```
data: {"type": "stage", "label": "Routing your request…"}
data: {"type": "stage", "label": "Processing your room request…"}
data: {"type": "proposal", "proposal_id": "...", "action": "book", "summary": {...}}
data: {"type": "done", "response": "I've put together a request for you to review."}
```

| Event type | Meaning |
|---|---|
| `stage` | Progress label for the UI's in-bubble indicator |
| `proposal` | A `propose_*` tool ran; carries the fields the confirmation dialog renders |
| `done` | Final assistant response for a turn that completed |
| `error` | The turn did not complete; ends the stream in place of `done` |

A `proposal` event appears after a `propose_*` tool call, carrying the same fields the
confirmation dialog renders. `summary` is action-shaped (`book`, `cancel`, or `modify`)
and is **never** derived from the assistant's own text.

A turn that does not complete ends with an `error` event instead of `done`, never both.
That covers a router or sub-agent timeout or exception, a turn that produced no reply,
and any exception mid-stream, including a `thread_id`/`customer_id` mismatch:

```json
{"type": "error", "message": "...", "code": "timeout"}
```

`message` is `[orchestration.messages].database_unavailable` for `unavailable`,
`[orchestration.messages].error` for the other failed-turn codes, or a mismatch-specific
string. It is never
written into the conversation history: the failed turn is dropped from the thread
entirely, and any proposal it created is invalidated, so resending the same text starts
clean. `code` cannot be an HTTP status here, unlike the readiness 503 above - the 200 and
the SSE headers are already committed by the time a turn in progress can fail. The
reachable codes are:

| `code` | Meaning |
|---|---|
| `timeout` | The router or a sub-agent exceeded its wall-clock cap |
| `internal` | Any other failure, including an unreachable model provider or a turn with no reply |
| `unavailable` | The booking agent's `run_sql` could not reach the database; the model's reply is discarded |
| `thread_mismatch` | The `thread_id` is bound to a different `customer_id` |

`unavailable` here means the database, not readiness. A readiness `"failed"` is not
reachable mid-stream: an unready orchestrator is stopped by the readiness gate above
before a stream ever starts.

The non-streaming JSON response carries the same `proposal` field when a proposal is
pending after the turn. A turn that did not complete returns `504` (`timeout`), `503` with
`Retry-After` (`unavailable`), or `502` (anything else) instead of `200`, with a body of
`{"code": ..., "message": ...}`, plus `retry_after_s` on a `503`. Nothing
is committed before the JSON branch knows the outcome, so it can use a real status.

## The propose/confirm contract

A client integrating with this API must respect one rule: **render the confirmation
dialog from the `proposal` event's `summary` fields, never from the assistant's text.**

The model is capable of describing a booking differently from the one it proposed. The
proposal is the authoritative record; the prose is not. The same principle applies to
outcomes - a successful `POST /v1/booking/confirm` returns an application-authored
receipt containing the confirmation number, and that receipt is what the guest should
see. The model never generates a confirmation number.

The flow:

1. `POST /v1/chat` returns or streams a `proposal` with a `proposal_id`.
2. The client renders a dialog from `summary`.
3. The guest clicks Confirm, and the client calls `POST /v1/booking/confirm` with the
   `proposal_id`.
4. The response carries the confirmation number and the receipt text. The application
   also writes that receipt into the LangGraph thread, so the agent's next turn knows
   what happened.

Or the guest declines, and the client calls `POST /v1/booking/dismiss`.

Proposals are single-use and expire after `[booking.proposals].ttl_s`. Sending another
chat message on the same thread supersedes any pending proposal.

### Confirm status codes

| Status | Meaning | Proposal afterward |
|---|---|---|
| `200` | Committed. `already_confirmed` distinguishes a fresh write from a replayed cached result (a duplicate confirm) | Retired |
| `404` | Unknown or expired (`ProposalNotFoundError`) | Already gone |
| `403` | Belongs to a different guest (`ProposalOwnershipError`) | Unchanged |
| `409` | The write was evaluated and refused - most commonly, another guest took one of the nights in the meantime. `detail` carries the app-authored reason verbatim | Retired |
| `503` | The database could not be reached at all; nothing was decided either way. `Retry-After` header included | **Kept pending** - confirming again is safe and is a real retry, not a duplicate |

The `503` case is why a client should keep its confirm dialog open (Confirm still
enabled) rather than treating every non-200 the same way: the proposal survives it on
the server, specifically so a guest can press Confirm again after a transient database
blip. See
[Booking agent](architecture/booking-agent.md#3-the-proposeconfirm-flow) for the
retire-on-`409`-but-not-on-`503` distinction this depends on.

## Failure semantics

Not every non-2xx response means the same thing to a client, and treating them
identically is what this API is deliberately designed to avoid:

| Response | Retryable? | Why |
|---|---|---|
| `/v1/chat` `503` | Yes, after `retry_after_s` | Nothing ran; the init loop keeps retrying regardless of `status` |
| `/v1/customers`, `/v1/bookings` `503` | Yes, after `Retry-After` | The startup window, or an unreachable database. Both are idempotent reads |
| Mid-stream `error` event, or JSON `502`/`503`/`504` | Yes for `timeout`, `internal`, and `unavailable`; no for `thread_mismatch` | The failed turn is dropped from history and its proposal invalidated, so resending the same text is safe. The system never resends on its own: the guest decides |
| Confirm `503` | Yes - the proposal is still pending | Nothing was decided; see the table above |
| Confirm `409` | No | The proposal is retired; a client should let the guest start a new request, not retry the same one |
| Confirm `404` / `403` | No | Terminal for this proposal id |
| Dismiss `404` / `403` | No | Terminal for this proposal id |

This mirrors the retry-safety tiering described in
[Design Goals and Decisions](design-decisions.md#asking-the-guest-to-try-again-was-hiding-three-different-failures):
an idempotent read is retried by the system itself and never surfaces a user-facing
failure at all; the chat turn offers a resend rather than auto-retrying, since it may
already have side effects; the confirm write never blind-retries, and instead the `503`
case is engineered to make a client-initiated retry safe.
