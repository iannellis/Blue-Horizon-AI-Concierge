# Booking agent

The booking agent searches availability with a typed search tool over a PostgreSQL
database hosted on [Neon](https://neon.tech). The model chooses filter values; the
application runs a fixed, read-only query. It never writes.

## The model proposes, the application decides

This is the central design constraint of the whole system, and it is enforced in four
independent places rather than by instructing the model to behave.

### 1. Two database roles

Two roles enforce read-only access at the database level, not just in application code:

| Role | Grants | Used by |
|---|---|---|
| `bh_agent_ro` | `SELECT` only, on `rooms` and `room_availability` | The model's `search_rooms` tool, exclusively |
| `bh_agent_rw` | Read-write | Server-side write functions, `/v1/customers`, `/v1/bookings` |

`customers`, `bookings`, and `booking_rooms` are off the read-only role's grants
entirely, so guest profile data cannot reach a third-party inference provider through
a search, even if a later change widened what the search queries.

`startup_check` refuses to start the application if a trial write through
`PGSQL_RO_DB_URL` succeeds, which catches the configuration mistake of pointing both
URLs at the same role and silently defeating the guarantee.

### 2. Fixed parameterized search

The model never writes SQL. Its one read tool, `search_rooms`, takes typed filters:

| Filter | Meaning |
|---|---|
| `check_in`, `check_out` | A stay. Only rooms free and priced for every night come back, with the stay's total. Omit both to search rooms in general. |
| `room_numbers`, `room_types`, `bed_types` | Any of these |
| `amenities` | The room has all of these |
| `view_types` | The room has any of these |
| `min_floor`, `max_floor` | A floor range; "top floor" is both set to the highest floor |
| `min_occupancy`, `max_occupancy` | How many guests the room sleeps, bounded from either side |
| `min_square_feet`, `max_square_feet` | Room size, bounded from either side |
| `accessible` | Require or exclude accessible rooms |
| `min_nightly_price`, `max_nightly_price` | Every night at or above, or at or below, this; needs dates |
| `sort_by` | Stay price, floor, room size, or room number, lowest first; a `_desc` form (`price_desc`, `floor_desc`, ...) runs highest first. Price ordering needs dates; without them results are in room number order |

The allowed values for room type, bed type, amenity, and view type are read from the
database at startup (`db_utils.fetch_rooms_metadata`) and become `Literal` types in the
tool's argument model, so the schema the model sees lists every legal value, and an
unknown one is rejected before any SQL runs. The same startup read supplies the floor
range and the availability window, which bound the floor and date fields and are
rendered into the system prompt. Nothing about the data's dates is hardcoded.

`blue_horizon/agents/booking/search.py` then runs one of two fixed queries, one for a
dated stay and one without dates, with the filter values bound as parameters. Each
query returns `[booking.agent].default_search_results` rooms, or as many as the model
asks for with `limit` up to `[booking.agent].max_search_results`, each with a fixed set
of guest-facing fields, plus `matching_count`, the number of rooms that matched, so a
"how many" question needs no second query. A `limit` above the maximum is clamped rather
than rejected, and the result then carries a `limit_note` telling the model to say so.
The note is in the result rather than the prompt so that it reaches a client with its
own prompt too. Internal identifiers, the room-level status, the
rate columns, and renovation dates are never selected.

This bounds what one search can cost. The free-form `run_sql` tool it replaced capped
rows, not size: a single aggregated row could carry millions of characters. It also
keeps the rules in code rather than in the prompt, which matters for any client that
brings its own prompt, such as a future MCP server. `search.py` imports nothing from
LangChain so that such a server can wrap it directly.

Rejected arguments come back as a result with `error_kind` `invalid_arguments` and a
message naming the problem (for example, the availability window), not as an
exception, so the model can correct itself or ask the guest. A
`ToolCallLimitMiddleware` caps `search_rooms` at
`[booking.agent].max_search_calls_per_turn` calls per guest turn; later calls are
refused and the model must answer with what it has.

The grant in layer 1 is still the guarantee that search cannot write. The fixed query
and the read-only transaction each search runs in are what the model can actually
reach.

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

**A proposal is retired only once the write's outcome is actually known.** Confirming
calls `write_ops.commit_booking` (or `cancel_`/`modify_booking`) and only then removes
the proposal from the pending index - not before. That ordering is what makes a `503`
retryable: if the database cannot be reached at all
(`write_ops.BookingUnavailableError`), the request was never evaluated against
`room_availability`, so the proposal is left pending instead of retired, and a second
`POST /v1/booking/confirm` on the same `proposal_id` is a genuine retry rather than a
`404`. Only a determinate refusal (`write_ops.BookingWriteError` - the write was
evaluated and the nights are unavailable) retires the proposal, because that outcome is
already known and a retry would just re-ask a question that has been answered.

**Every confirm outcome that settles a proposal is written into the thread.** A success
appends the app-authored receipt (`receipts.receipt_message`), and a `BookingWriteError`
appends an app-authored refusal note (`receipts.refusal_message`): "Not confirmed:", the
write's reason, and that the request is closed. Both go through
`OrchestrationManager.append_assistant_message`, from `POST /v1/booking/confirm` and
from the eval harnesses' auto-confirm alike. Without the refusal note the thread's last
assistant message is the model's own "for you to review", and when the guest asks again
the model tells them to press a Confirm button that no longer exists. The note is
silent on whether anything changed, since after a lost acknowledgment an earlier
attempt on the same proposal may have landed.

**The total the guest saw is checked before the write commits.** The confirm path passes
the proposal's total to the `write_ops` function, which compares it with the total it
computes under lock. If they differ, it raises `write_ops.PricingMismatchError` inside
the transaction, so the write rolls back. Prices are never written, so only a bug can
cause a mismatch, for example preview pricing drifting from commit pricing. The error is
a `BookingWriteError`, so the proposal is retired and the guest gets a `409`, but it is
logged as an error rather than as an ordinary refusal.

**Invariant 6 (double-booking is structurally impossible) is what makes retaining a
pending proposal across a failed attempt safe.** A retried commit re-prices every
night from scratch under the same `SELECT ... FOR UPDATE` lock described below, so it
can never write a second, conflicting row - it can only either succeed cleanly (nothing
was written the first time) or, if the first attempt's `COMMIT` actually landed and
only the acknowledgment was lost, discover the guest's own booking already exists.

That lost-acknowledgment case is settled by a **reconciliation read**,
`write_ops.find_booking_for_rooms`, called from the confirm path only for a `"book"`
action whose retry raises `BookingUnavailableError` again. It looks up whether every
requested room-stay already belongs to one booking owned by the guest, and it is
**exact, not heuristic**: the `booking_rooms_no_overlap` GiST exclusion constraint makes
`(room_id, [check_in, check_out))` unique across the live rows in `booking_rooms`, so a
match can only be the guest's own prior commit. Found means the commit landed - the
proposal is retired and the real receipt returned, `already_confirmed=False` since this
is the first time the guest sees it. Not found means it did not - the proposal stays
pending. A failed reconciliation read itself is left unresolved rather than reported as
either outcome, since collapsing "checked and found nothing" together with "could not
check" would tell the guest something this code never actually established. Scoped to
`"book"` only: `cancel` and `modify` do not share that uniqueness property, and their
own state checks (already cancelled, unknown `booking_room_id`) are already
self-describing.

### 4. The eval suite checks for the failure mode anyway

`booking_no_unbacked_success_claims` explicitly looks for a success claim with no
receipt behind it. Defense in depth is only credible if something tests that the
defenses hold.

## Tool surface

| Tool | Access | Returns |
|---|---|---|
| `search_rooms` | `bh_agent_ro`, fixed parameterized query | Up to `limit` rooms (clamped to `max_search_results`) plus `matching_count` |
| `list_my_bookings` | Server-injected `customer_id` via `RunnableConfig` | This guest's reservations |
| `propose_booking` | None (in-process) | A proposal id, or a refusal past the [room-night limit](#room-night-limit) |
| `propose_cancellation` | None (in-process) | A proposal id |
| `propose_modification` | None (in-process) | A proposal id, or a refusal past the [room-night limit](#room-night-limit) |

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

## Room-night limit

One guest may hold at most `[booking.limits].max_room_nights_per_guest` room-nights (60
by default) across bookings made in the last `room_night_window_hours` (24). A
room-night is one room for one night, so a three-room, two-night stay counts six.
Without the limit, one visitor could book every room of a type for weeks and leave every
other visitor a sold-out hotel until the nightly reset.

The count covers the guest's bookings whose `booked_at` falls in the window and whose
status is not `cancelled`. The window leaves out the seeded guests' loaded history:
the loader stamps every pre-existing booking with a `booked_at` of midnight UTC on the
first date `room_availability` covers, long before any booking the app makes. The window
also lines up with the nightly reset. A cancelled booking does not count, since it released its nights.

`commit_booking` and `modify_booking` enforce it inside their transactions, through
`write_ops._require_room_night_allowance`, and raise `write_ops.BookingLimitError`, a
`BookingWriteError`, when a write would go over. The confirm path treats it like any
other refusal: the proposal is retired, the guest gets a `409` whose `detail` names the
limit and the total the write would have reached, and the refusal note goes into the
thread. `cancel_booking` is never checked, since it only releases nights. The
`ProposalStore` holds the limit and passes it to both writes, so the eval harness's
auto-confirm applies it too.

A modification is checked only when it adds room-nights, so a guest already over the
limit (after it was lowered, say) can still move or shorten a stay. The booking being
modified counts in full when it was itself made in the window. A booking made before the
window counts only for the nights the modification adds, so a seeded guest can extend an
old reservation, but only by what the limit leaves.

**Each write first takes the guest's advisory lock,**
`pg_advisory_xact_lock(namespace, customer_id)`, held until the transaction ends.
Without it, two confirms for one guest, from two tabs or from two visitors sharing a
guest, could each read the total before the other committed, and both would pass. A
`SELECT ... FOR UPDATE` on the guest's `customers` row would serialize them the same
way, but it needs `UPDATE` on `customers`, which `bh_agent_rw` deliberately lacks. The
advisory lock needs no grant. It is taken before any row lock, so it adds no new
lock-ordering deadlock.

`propose_booking` and `propose_modification` run the same check without the lock
(`write_ops.check_room_night_limit`) and refuse with its reason, so the model can tell
the guest before a dialog appears. That check is a courtesy; the one inside the write
decides.

## Guest identity

Guests are identified without a real login, since this is a demo. A session with no
guest yet claims a random one that nobody else currently holds, tracked in an in-memory
claim registry guarded by a lock. One in-memory dict is sufficient because the whole
Space runs as one Streamlit process, so every browser session already lives on a thread
of that same process. The dict and its lock are created by an `st.cache_resource`
function rather than held in a module global, because Streamlit runs the script in a
fresh module on every rerun and a global would start empty each time. The hourly turn
limit's registry is held the same way.

A claim is held by the visitor, not the browser session: the Google account when signed
in, the session otherwise, the same key the hourly turn limit counts by. A session that
starts without a guest, after a hard refresh or in a second tab, gets back the guest its
visitor already holds. Nothing about the guest is put in the URL, so a shared link does
not carry the sender's guest with it. Without auth the session id is kept in the `sid`
URL parameter, since nothing else survives a refresh.

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
