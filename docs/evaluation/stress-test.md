# Stress test

Simulates concurrent users performing BOOK, MODIFY, and CANCEL operations against the
booking agent, then checks database invariants and reconciles expected against actual
state.

```bash
python -m eval.stress --config eval/stress_config.toml
```

Artifacts are written to `eval/outputs/stress_<timestamp>/` and logs to `eval/logs/`.

## Configuration

| Setting | Default | Description |
|---|---|---|
| `stress.workload.users` | 50 | Concurrent simulated users |
| `stress.workload.ops_per_user` | 5 | Operations per user |
| `stress.targets.hot_target_count` | 10 | High-contention target rooms |
| `stress.targets.hot_target_probability` | 0.8 | Probability of picking a hot target |

## Results

50 simultaneous sessions with 5 booking operations each - 250 operations total, with
`book`/`modify`/`cancel` weighted 50/25/25 - biased 80% of the time toward a hot subset
of 10 room and date targets to force contention. Every operation went through the full
propose-then-auto-confirm path, the harness's stand-in for a human clicking Confirm, so
this exercises real commits under load rather than just proposals.

Run of 2026-11-11, in `eval/outputs/stress_20261111_203349/`. Overall status **PASS**.
It is the first run since `llm_concurrency` went from 15 to 25, and since the booking
path gained the bookable-window refusal in the `propose_*` tools, the cap on model calls
per turn, the per-guest room-night limit, and moderation and an injection check on every
guest message. None of those refusals fired: every reply was a proposal or a "this room
isn't available" refusal.

| Result | Value |
|--------|-------|
| Operations completed (of 250) | 250 (0 errored) |
| Successful bookings/modifies/cancels | 76 (30.4%) |
| Correctly-refused conflicts | 174 (69.6%) |
| Double-booking violations | 0 |
| Reservations with null status | 0 |
| Expected bookings confirmed in DB | 23 / 23 |
| Wall-clock elapsed | 73.7 s (3.39 ops/s) |
| Per-operation latency (mean / median / p95) | 10.7 s / 10.7 s / 13.1 s |

Per-operation latency here is far higher than the
[single-turn eval latency](index.md#latency-end-to-end-ms) because each stress operation
is a full propose-then-confirm round trip under 50-way concurrency against the same
`llm_concurrency` semaphore, not one turn on an idle system. It measures saturation
behavior, not the latency a guest experiences. The server logs show it directly: of a
mean 10.6 s turn, 4.9 s (46%) was spent waiting for one of the 25 semaphore slots. In
the 2026-10-16 run, with 15 slots, the wait was 10.9 s of a 16.2 s turn (67%).

The high conflict rate is expected and by design. With 80% of operations targeting the
same 10 hot slots, most of them are *supposed* to lose the race. What matters is that
they lose it cleanly: a refusal a guest can understand, not a partial write or a raw
database error.

The conflict rate is six points above the previous runs, and the extra conflicts are
bookings that drew a hot slot. This run's random mix had 169 bookings, against 146 on
2026-10-16, and 142 of them targeted a hot slot, against 115. A hot slot can be won once
unless a cancel frees it, so the extra attempts lost: 17 hot-slot bookings won, against
22. Bookings into other slots won 26 of 27, against 31 of 31, and modifies conflicted
less (48, against 66). The higher `llm_concurrency` changed how hot-slot bookings lost
rather than how many: 18 lost at confirm after a proposal, against 11, since more
sessions searched while a slot was still free and only lost at the row lock.

!!! note "A 2026-10-16 run failed reconciliation"
    The first of two 2026-10-16 runs reconciled 26 of 27 expected bookings. No write was wrong.
    A confirm that lost its race retired the proposal but left nothing in the thread, so
    when the guest asked again the agent told them to press a Confirm button that no
    longer existed. The harness then counted that reply as a success. Both are fixed: a
    refused confirm now writes an app-authored note into the thread, and the harness
    counts a success only when a confirm succeeded. The 2026-10-16 run in the table
    below is the rerun. See
    [Design Decisions](../design-decisions.md#a-refused-confirm-left-the-agent-pointing-at-a-dead-proposal).

!!! note "51 conflicts were flagged as suspicious"
    Reconciliation flags a conflict as *suspicious* when none of the nights it targeted
    are booked in the final database state, which can mean the agent refused a room that
    was actually available. It can also happen legitimately, when a booking was made and
    then cancelled later in the same run, so the signal is informational rather than a
    failure, and the run passed reconciliation.

    The summary lists 10 flagged operations in detail (the detail list is capped by
    `reconcile_max_detail`). Six target room 1111, 2025-03-13 to 03-15, three target
    room 401, 2025-03-20 to 03-22, and one targets room 1706, 2025-04-07 to 04-09. The
    operations log shows the cancel-later explanation directly for all three. One guest
    booked room 1111 and cancelled it later in the run, and the same happened to room
    1706. Room 401 was booked and cancelled once, then booked by a second guest who
    modified the stay to another room. Each ended free, so every conflict against it in
    between reads as suspicious against the final state. Distinguishing that
    properly from an over-refusal would need the reconciliation pass to check nights
    against the state at the time of the operation rather than the final state. That is
    not implemented.

    The count is higher than in the runs before 2026-10-16 (13 to 18) partly by
    construction. The harness now counts any operation with no successful confirm as a
    conflict, including one where the agent asked a clarifying question, so more
    operations are candidates. It rose again from 31 to 51 with the conflict count.

### What the earlier runs show

The previous run, from 2026-08-10, is worth keeping in view because it executed
**before** the `booking_rooms_no_overlap` exclusion constraint was added on 2026-08-15.
At that point nothing but `commit_booking` and `modify_booking`'s own
`SELECT ... FOR UPDATE` locking stood between 250 contended operations and an
overlapping pair, and it recorded zero violations. Its detection query was already the
real `booking_rooms` self-join rather than the tautological `room_availability` grouping
it replaced, so that zero is meaningful.

Superseded runs' artifacts are not kept in `eval/outputs/`: each new run replaces the
last, on the reasoning that superseded results stay available in git history rather than
cluttering the output directory. Retrieve them with
`git show 0e2e46a^:eval/outputs/stress_20260810_011711/stress_summary.json` (2026-08-10),
`git show 6b7397f^:eval/outputs/stress_20260823_040837/stress_summary.json`
(2026-08-23), and
`git show 6b7397f:eval/outputs/stress_20260915_174520/stress_summary.json` (2026-09-15),
and `git show 891779f:eval/outputs/stress_20261016_204228/stress_summary.json`
(2026-10-16).

| | 2026-08-10 | 2026-08-23 | 2026-09-15 | 2026-10-16 | 2026-11-11 |
|---|---|---|---|---|---|
| GiST constraint present | no | yes | yes | yes | yes |
| Search tool | `run_sql` | `run_sql` | `run_sql` | `search_rooms` | `search_rooms` |
| `llm_concurrency` | 15 | 15 | 15 | 15 | 25 |
| Successes | 98 (39.2%) | 88 (35.2%) | 90 (36.0%) | 91 (36.4%) | 76 (30.4%) |
| Conflicts | 152 (60.8%) | 162 (64.8%) | 160 (64.0%) | 159 (63.6%) | 174 (69.6%) |
| Errors | 0 | 0 | 0 | 0 | 0 |
| Double-booking violations | 0 | 0 | 0 | 0 | 0 |
| Reconciliation | 26 / 26 | 22 / 22 | 24 / 24 | 29 / 29 | 23 / 23 |
| Suspicious conflicts | 18 | 14 | 13 | 31 | 51 |
| Mean per-operation latency | 15.0 s | 15.5 s | 43.8 s | 16.3 s | 10.7 s |

The 2026-08-10 run is the stronger evidence that the application-level locking is
correct on its own. The later runs' zeros are expected rather than informative, since
the constraint now makes an overlapping pair unrepresentable regardless of what the
locking does. The success-versus-conflict split moved by up to four points between the
first four runs, which is ordinary variance in which sessions win contended slots, not a
behavioral change: the workload re-randomizes its hot targets and its operation mix each
run. The 2026-11-11 run's six-point move is the same variance, traced above to more
bookings drawing a hot slot.

The 2026-09-15 run's latency was nearly three times the others' under the same workload,
concurrency, and models. With the 2026-10-16 run back at 16.3 s, that run looks like an
outlier, most likely provider-side, rather than a regression. Its cause was never
isolated. The 2026-11-11 run's drop to 10.7 s follows the semaphore wait, which fell
from 10.9 s to 4.9 s with 25 slots instead of 15, despite the moderation and injection
calls each turn now adds.

## What the invariant checks mean

### Double-booking violations

Two `booking_rooms` rows for the same room with overlapping date ranges, meaning
`commit_booking` or `modify_booking` failed to enforce mutual exclusion under load.

This is now a schema-level impossibility rather than something checked for after the
fact. `booking_rooms` carries a GiST exclusion constraint, `booking_rooms_no_overlap`
(see
[`schema.sql`](https://github.com/iannellis/Blue-Horizon-AI-Concierge/blob/main/blue_horizon/load_data/schema.sql),
applied by `setup_schema` in
[`booking_pgsql.py`](https://github.com/iannellis/Blue-Horizon-AI-Concierge/blob/main/blue_horizon/load_data/booking_pgsql.py)),
that Postgres enforces on every insert and update, on top of the `FOR UPDATE` locking on
`room_availability` that `commit_booking` and `modify_booking` already perform. If the
constraint is ever hit - only possible if `room_availability` and `booking_rooms` have
drifted out of sync - `write_ops` catches it and turns it into the same clean
`BookingWriteError` refusal a guest sees for any other unavailable night.

### The detection query's own history

The violation check queries `booking_rooms` directly, via the shared
[`eval/db_invariants.py`](https://github.com/iannellis/Blue-Horizon-AI-Concierge/blob/main/eval/db_invariants.py)
helper that the eval harness also uses, so the two cannot drift apart.

An earlier version instead grouped `room_availability` by `(room_id, date)`. That was a
tautology: the table's own `UNIQUE` constraint makes such a group impossible to violate
regardless of what the agent did. The check could never have gone red.

The rewritten check was trusted only after being confirmed to go red against a
hand-inserted overlapping pair. That verification is now a permanent regression test in
[`tests/booking/test_db_invariants.py`](https://github.com/iannellis/Blue-Horizon-AI-Concierge/blob/main/tests/booking/test_db_invariants.py),
which today tests the constraint's own refusal directly, since a hand-inserted
overlapping pair can no longer even be inserted. See
[the pre-constraint version](https://github.com/iannellis/Blue-Horizon-AI-Concierge/blob/e67fb9b33279219a0d4b6b8fc8a85bbb92733dfa/tests/booking/test_db_invariants.py)
for how it worked before.

That same test file separately verifies the *detection query* still works, by
temporarily dropping `booking_rooms_no_overlap` inside a transaction that always rolls
back, inserting an overlapping pair, and asserting the query flags it. The constraint is
deliberately not `DEFERRABLE`, which would have allowed an easier test at the cost of
letting overlapping rows exist transiently mid-transaction in production too.

With the constraint in place, this check is belt-and-suspenders rather than the primary
guarantee. A genuine violation is unreachable through normal writes, so it mainly serves
as an audit signal if the constraint is ever dropped or bypassed. Its passing during
normal stress and eval runs is expected to be uninformative about agent behavior.

### Null-status reservations

A `room_availability` row without a valid status field, indicating a partially-written
or corrupted booking.

### Reconciliation

Separately from the invariant checks, the harness reconciles every thread's expected
final booking against the database. In the current run, all 23 threads that ended with a
successful book or modify had that exact room and date range sitting `Booked` in the
database. A success here means the harness's confirm committed, never what the agent's
reply said, so reconciliation checks the harness's bookkeeping against the database as
much as it checks the agent.

## Transient SSL warnings

An occasional "SSL connection closed" warning appears right after a Neon branch reset,
and is retried automatically by design. It is self-healing and not a failure.

`stress_failures.jsonl` flags any `search_rooms` call that returned an error, which is
not the same thing as a connectivity failure. A correctly-refused conflict is not a
failure; a search with invalid arguments that the agent corrected on retry will still
appear there. Each operation's search calls are logged under `search_calls`.
