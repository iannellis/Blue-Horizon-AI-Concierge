# Booking agent

The booking agent searches availability via LLM-generated, read-only SQL against a
PostgreSQL database hosted on [Neon](https://neon.tech). It never writes.

## The model proposes, the application decides

This is the central design constraint of the whole system, and it is enforced in four
independent places rather than by instructing the model to behave.

### 1. Two database roles

Two roles enforce read-only access at the database level, not just in application code:

| Role | Grants | Used by |
|---|---|---|
| `bh_agent_ro` | `SELECT` only, on `rooms` and `room_availability` | The model's `run_sql` tool, exclusively |
| `bh_agent_rw` | Read-write | Server-side write functions, `/v1/customers`, `/v1/bookings` |

`customers`, `bookings`, and `booking_rooms` are off the read-only role's grants
entirely, so guest profile data cannot reach a third-party inference provider through
free-form SQL even in principle.

`startup_check` refuses to start the application if a trial write through
`PGSQL_RO_DB_URL` succeeds, which catches the configuration mistake of pointing both
URLs at the same role and silently defeating the guarantee.

### 2. SQL guardrails

A `sqlglot`-parsed AST allowlist restricts queries to `SELECT` against the two permitted
tables and caps results at 50 rows before they reach the LLM, limiting context size and
cost. This is deliberately redundant with the grant. The grant is the guarantee; the
guardrail is a belt-and-braces layer that also produces better error messages than a
raw Postgres privilege refusal.

A privilege refusal (`ReadOnlySqlTransaction`, `InsufficientPrivilege`) is caught
separately from an ordinary SQL error and returned as a flat, non-retryable refusal, so
the model does not waste turns trying to rephrase its way past a permission boundary.

### 3. The propose/confirm flow

To book, cancel, or modify, the agent calls a `propose_*` tool. That tool prices and
validates the request and stores it in an in-process `ProposalStore`. Nothing is written
to the database. The tool returns **only a proposal id, never success text**.

The UI renders a confirmation dialog built strictly from the *proposal's own stored
fields*, never from the model's chat text. Only a guest clicking **Confirm** calls
`POST /v1/booking/confirm`, which is the sole caller of the `write_ops` commit functions
(`commit_booking`, `cancel_booking`, `modify_booking`).

A successful commit returns a confirmation number that the *application* renders as a
new chat message. The model is instructed never to claim a booking succeeded and never
to invent a confirmation number.

Proposals are TTL-bounded (`[booking.proposals].ttl_s`, 30 minutes by default),
single-use, and superseded by any new proposal or user message on the same thread. A
proposal reserves no inventory, so the TTL only bounds the store's size.

### 4. The eval suite checks for the failure mode anyway

`booking_no_unbacked_success_claims` explicitly looks for a success claim with no
receipt behind it. Defense in depth is only credible if something tests that the
defenses hold.

## Tool surface

| Tool | Access | Returns |
|---|---|---|
| `run_sql` | `bh_agent_ro`, `SELECT` only | Up to 50 rows |
| `list_my_bookings` | Server-injected `customer_id` via `RunnableConfig` | This guest's reservations |
| `propose_booking` | None (in-process) | A proposal id |
| `propose_cancellation` | None (in-process) | A proposal id |
| `propose_modification` | None (in-process) | A proposal id |

`list_my_bookings` takes its `customer_id` from `RunnableConfig`, injected by the
server. The parameter is invisible to the model, so the model cannot ask for another
guest's reservations by passing a different id.

## Concurrency and double-booking

Two `booking_rooms` rows for the same room with overlapping date ranges is a
**double-booking violation**. Three mechanisms prevent it:

1. **`SELECT ... FOR UPDATE` locking** on `room_availability` inside each write
   transaction. This is the primary mechanism and is expected to catch every real
   contention case.
2. **A GiST exclusion constraint**, `booking_rooms_no_overlap`, which makes overlapping
   rows impossible to insert or update into existence. It requires the `btree_gist`
   extension so GiST can support `=` on the integer `room_id` alongside the date-range
   overlap operator. `write_ops` catches the resulting `ExclusionViolation` and
   translates it into the same clean `BookingWriteError` refusal a guest sees for any
   other unavailable night, rather than leaking a raw database error.
3. **A `prevent_maintenance_booking` trigger** on `booking_rooms`, refusing any insert
   or update covering a night that `room_availability` marks `Maintenance`. This
   backstops `write_ops._price_one_room`, which already refuses to price such a night
   through the application.

Layers 2 and 3 should only ever fire if `room_availability` and `booking_rooms` have
drifted out of sync. They are audit signals, not the load-bearing guarantee.

Each write function is one explicit transaction: `SELECT ... FOR UPDATE`, verify, write,
then set `confirmation_number` via `RETURNING`. `cancel_booking` never touches `price`,
so a night's rate survives a book/cancel round trip.

See the [Stress Test](../evaluation/stress-test.md) for how this holds up under 50
concurrent sessions deliberately fighting over the same 10 rooms.

## Guest identity

Guests are identified without a real login, since this is a demo. A session with no
guest yet claims a random one that nobody else currently holds, tracked in an in-memory
claim registry guarded by a lock. A single module-level dict is sufficient because the
whole Space runs as one Streamlit process, so every browser session already lives on a
thread of that same process.

A claim's activity timestamp refreshes on every rerun of its session, so a closed or
ten-minutes-idle tab ages out and returns its guest to the pool. Signing out releases
the claim immediately. If every seeded guest is claimed, the session sees a capacity
message.

There is deliberately no manual guest dropdown: letting someone hand-pick an
already-claimed guest would defeat the point of exclusive assignment. See
[Design Goals and Decisions](../design-decisions.md#guest-identity-is-a-claim-registry-not-a-dropdown).

!!! warning "Demo-grade authorization"
    `GET /v1/bookings` is unauthenticated, consistent with the rest of this demo's
    guest model. That is acceptable here because guests are assigned rather than
    real accounts, but it is worth stating plainly rather than leaving a reader to
    discover it.
