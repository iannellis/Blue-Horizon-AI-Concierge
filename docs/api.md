# API reference

The FastAPI backend runs on port `8000` and exposes the following endpoints under the
`/v1` prefix.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/v1/health` | Returns `{"status": "ok"}` (200) when ready, `{"status": "starting"}` (503) during init |
| `GET` | `/v1/customers` | Lists the seeded guests as `{customer_id, first_name, last_name}`, for guest assignment |
| `GET` | `/v1/bookings?customer_id=` | Lists one guest's reservations (confirmation number, rooms, dates, total, status) |
| `POST` | `/v1/chat` | Send a message. Content-negotiated: `Accept: text/event-stream` streams SSE events, anything else returns one JSON response |
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
| `done` | Final assistant response for the turn |
| `error` | An exception occurred mid-stream |

A `proposal` event appears after a `propose_*` tool call, carrying the same fields the
confirmation dialog renders. `summary` is action-shaped (`book`, `cancel`, or `modify`)
and is **never** derived from the assistant's own text.

Any exception mid-stream, including a `thread_id`/`customer_id` mismatch, is translated
into `{"type": "error", "message": "..."}` rather than silently severing the connection.

The non-streaming JSON response carries the same `proposal` field when a proposal is
pending after the turn.

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
