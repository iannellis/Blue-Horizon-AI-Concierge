# Running locally

## Prerequisites

- Python 3.13
- A running Redis instance
- A PostgreSQL database (Neon or local)
- The `bh_agent_rw` and `bh_agent_ro` roles created in that database, with passwords set
- Dependencies installed via [uv](https://docs.astral.sh/uv/):

```bash
uv sync --group ui
```

Add `--all-groups` to install the eval, notebook, and lint groups as well.

Set the environment variables described in [Configuration](configuration.md) in a `.env`
file at the project root. Both the API and the UI read it.

## 1. Load Redis

Populate Redis with hotel information:

```bash
python -m blue_horizon.load_data.information_redis
```

## 2. Load PostgreSQL

Populate PostgreSQL with room, availability, customer, and pre-existing booking data:

```bash
python -m blue_horizon.load_data.booking_pgsql
```

This requires `PGSQL_ROOT_PARENT_DB_URL` and runs against the **Parent** branch only.

Besides `rooms` and `room_availability`, this seeds `customers`, `bookings`, and
`booking_rooms`.

`customers` gets **every** source customer, not just the `seeded_customer_count` (25 by
default) that the UI offers, because most pre-existing bookings belong to a customer
outside any small subset. Loading only a subset leaves most of `room_availability`
marked `Booked` with no matching reservation. See
[Design Goals and Decisions](../design-decisions.md#loading-only-the-seeded-guests-corrupted-the-availability-data).

The `seeded_customer_count` source customers with the richest pre-existing booking
history - not an arbitrary first N - are remapped to a dense `customer_id` range of
`[1, seeded_customer_count]` and loaded first, so the UI's guest assignment and the eval
and stress harnesses' simulated-guest selection keep working unchanged. Every other
customer is loaded too, with an arbitrary but unique id after that block.

Pre-existing reservations for every loaded customer go into `bookings` and
`booking_rooms`, clamped to the date range `room_availability` covers and filtered to
the maximum non-overlapping subset per room, since the source data contains overlapping
stays for a given room and the `booking_rooms_no_overlap` constraint refuses those.

Loading real bookings can leave `room_availability` out of sync with what actually got
kept, in either direction, so the loader reconciles it against the loaded
`booking_rooms` rows before finishing. It also creates the `prevent_maintenance_booking`
trigger on `booking_rooms`, refusing any insert or update that would cover a night
`room_availability` marks `Maintenance`.

## 3. Grant the database roles

The `reload_sql_tables` step above reapplies the grants automatically as its last step,
in the same transaction. To apply them by hand - for instance, against a branch that was
reset rather than reloaded:

```bash
psql "$PGSQL_ROOT_PARENT_DB_URL" -f blue_horizon/load_data/regrant_booking_agent_role.sql
```

The roles themselves are **not** created by this step. They must already exist with
passwords set, since `PGSQL_RO_DB_URL` and `PGSQL_RW_DB_URL` authenticate as them. The
script only reapplies least-privilege grants to already-present roles.

Because `booking_pgsql.py` drops and recreates tables, dropping their grants with them,
this step must follow every reload. Forgetting it once left the Parent branch with zero
grants and silently broke every branch reset downstream of it, which is why the reload
now performs the regrant itself.

## 4. Start the API

```bash
fastapi run blue_horizon/api/app.py --port 8000
```

## 5. Start the UI

```bash
streamlit run ui/app.py
```

The UI connects to `http://localhost:8000` by default. Override with the
`BLUE_HORIZON_API_URL` environment variable.

## Notebooks

The `notebooks/` directory contains the development and exploration work:

| Notebook | Contents |
|---|---|
| `eda.ipynb` | Exploratory data analysis of the source data |
| `neonsql.ipynb` | PostgreSQL setup and natural-language querying experiments |
| `information_agent.ipynb` | Information agent development |
| `booking_agent.ipynb` | Booking agent development |
| `orchestration.ipynb` | Router and graph development |
| `full_agent.ipynb` | End-to-end walkthrough |

They require the `notebook` dependency group and an editable install of the project,
which `uv sync` provides.
