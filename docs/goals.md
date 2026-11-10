# Goals

## Why this project exists

Blue Horizon is a portfolio project. It is meant to show that I can design an AI agent
that is safe to trust with a real action and is measured well enough for that trust to
be checked rather than assumed.

A hotel concierge is the vehicle because it combines the two kinds of work most agents
do: answering questions from a body of documents, and acting on a system of record.
Answering questions tests grounding and retrieval. Booking a room tests something
harder: whether a model with access to a database can be stopped from doing damage
with it, however it behaves.

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

## Where each goal shows up

| Goal | Where to look |
|---|---|
| A guest can trust it with a reservation | The propose/confirm split in the [Booking Agent](architecture/booking-agent.md) and the [API Reference](api.md) |
| Correctness is structural | Database roles, the fixed search queries, and the double-booking guards in the [Booking Agent](architecture/booking-agent.md); the reasoning behind each in [Design Decisions](design-decisions.md) |
| Measured continuously | [Evaluation](evaluation/index.md) for results and baselines, the [Stress Test](evaluation/stress-test.md) for concurrent booking, [Deployment and CI](guides/deployment.md) for the gate |

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
