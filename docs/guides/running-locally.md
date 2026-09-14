# Running locally

## Prerequisites

- Python 3.13
- A running Redis instance
- A PostgreSQL database (Neon or local). Step 2 creates the roles it needs.
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

## 2. Create the database roles

Nothing in this repository creates the two roles. **Both `bh_agent_ro` and `bh_agent_rw`
must exist, with passwords set, on all three branches**, before step 3 runs.

```sql
CREATE ROLE bh_agent_ro LOGIN PASSWORD '<password>';
CREATE ROLE bh_agent_rw LOGIN PASSWORD '<password>';
```

Give them the same passwords the matching URLs use.

Both roles also carry a role-level `statement_timeout` of 10s, applied with `ALTER ROLE`
rather than kept in `app_config.toml` so that it survives PgBouncer transaction pooling,
where a per-connection `SET` would not. You do not set it by hand: step 3 applies it on
Parent through `regrant_booking_agent_role.sql`, and a branch reset carries it down to
every child branch along with the grants. It takes effect on newly established
connections, so an already running pool keeps the old value until its connections are
recycled.

`search_path` needs no such treatment. The roles use the PostgreSQL default of
`"$user", public`, which already resolves to `public` because no schema is named after
either role, and the code paths that depend on it issue `SET search_path TO public`
themselves.

On Neon these can equivalently be created from the console's **Roles** tab, which acts on
one branch at a time. That distinction matters more than it looks:

!!! danger "On Neon, a role belongs to a branch, not to the project"
    A branch has a role only if the role was created on that branch directly, or if it
    already inherited the role from its parent when the branch was created. Creating a
    role on Parent does **not** immediately reach branches that already exist. This project hit
    exactly that: `bh_agent_ro` had to be created three separate times, once on each
    branch, because both child branches predated the role.

    **All three branches need both roles.** The reason differs per branch, and Parent is
    the one most easily missed, because nothing ever logs in to Parent as either role:

    | Branch | Why both roles must exist there | Connected to by |
    |---|---|---|
    | Parent | Step 3's loader `GRANT`s to both by name and fails if either is missing | `PGSQL_ROOT_PARENT_DB_URL`, the schema owner, which is neither role |
    | Production | The running app authenticates as both | `PGSQL_RW_DB_URL`, `PGSQL_RO_DB_URL` |
    | Development | The `db_integration` tests and the eval harness authenticate as both | `PGSQL_RW_EVAL_DB_URL`, `PGSQL_RO_EVAL_DB_URL` |

    Parent matters beyond its own load: every other branch is reset from it, and a reset
    copies Parent's grants and role-level settings, such as `statement_timeout`, onto the
    child. A Parent left without grants therefore silently breaks every branch reset
    downstream. See
    [The three root URLs](configuration.md#the-three-root-urls).

## 3. Load PostgreSQL

Populate PostgreSQL with room, availability, customer, and pre-existing booking data:

```bash
python -m blue_horizon.load_data.booking_pgsql
```

This requires `PGSQL_ROOT_PARENT_DB_URL` and runs against the **Parent** branch only.

!!! warning "Missing roles fail the whole load, not just the grants"
    `reload_sql_tables` does everything in one transaction and commits only at the end,
    and its last step grants privileges to `bh_agent_ro` and `bh_agent_rw`. If either role
    is absent, that grant raises `role "bh_agent_ro" does not exist`, the transaction rolls
    back, and **no data is loaded at all**. The symptom is a failure at the very end of a
    long load, with an empty database left behind rather than a partly loaded one.

Besides `rooms` and `room_availability`, this seeds `customers`, `bookings`, and
`booking_rooms`.

`customers` gets **every** source customer, not just the `seeded_customer_count` (15 by
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

## 4. Reapply the grants by hand (not normally needed)

Step 3 already reapplies the grants as its last step, in the same transaction, so this
is only for an out-of-band fix, such as a branch that lost its grants without a reload:

```bash
psql "$PGSQL_ROOT_PARENT_DB_URL" -f blue_horizon/load_data/regrant_booking_agent_role.sql
```

This grants privileges to the roles from step 2 and sets their `statement_timeout`. It
does not create them, and it fails the same way step 3 does if they are absent.

Because `booking_pgsql.py` drops and recreates tables, dropping their grants with them,
the regrant must follow every reload. Forgetting it once left the Parent branch with zero
grants and silently broke every branch reset downstream of it, which is why the reload
now performs the regrant itself.

## 5. Start the API

```bash
fastapi run blue_horizon/api/app.py --port 8000
```

!!! danger "On Windows, add `--reload`"
    Without it, every real database connection fails with `Psycopg cannot use the
    'ProactorEventLoop' to run in async mode`. This is not a bug in this app: on Windows,
    uvicorn's default single-process loop factory hard-codes `ProactorEventLoop`, and it
    passes that in directly as an explicit `loop_factory`, which overrides any event loop
    policy the app itself sets, and it does so before the app module is even imported.
    `--reload` runs the server in a subprocess uvicorn manages differently, and on that
    path it uses `SelectorEventLoop` instead, which is what `psycopg`'s async mode
    requires. `--workers` (mutually exclusive with `--reload`) takes the same path, but
    `--reload` is the one you want for local development anyway.

    ```bash
    fastapi run blue_horizon/api/app.py --port 8000 --reload
    ```

    Production deploys on Linux, where uvicorn's default loop already works, so this is a
    local Windows-only concern. See `deploy/supervisord.conf`, which runs the same
    command with no `--reload`.

## 6. Start the UI

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
