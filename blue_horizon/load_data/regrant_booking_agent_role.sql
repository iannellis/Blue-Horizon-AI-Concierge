-- Reapply least-privilege grants for the booking agent roles after
-- `blue_horizon/load_data/booking_pgsql.py` recreates the `rooms`,
-- `room_availability`, `customers`, `bookings`, `booking_rooms`, and
-- enum-type objects.
--
-- Two roles, not one. This is the enforcement boundary: `bh_agent_ro` is the
-- only role the model's free-text `run_sql` tool ever connects as, and its
-- grants are exactly the guardrail's table allowlist (`rooms`,
-- `room_availability`, SELECT only). `bh_agent_rw` is used only by
-- server-side write_ops functions the model never touches directly.
--
-- What it does:
-- 1. Ensures both target roles can use the `public` schema.
-- 2. Re-grants USAGE on the enum types recreated by the loader.
-- 3. Removes any table/sequence privileges that exceed each role's contract.
-- 4. Grants exactly the permissions each role needs:
--      bh_agent_ro:
--        - SELECT on `public.rooms`
--        - SELECT on `public.room_availability`
--      bh_agent_rw:
--        - SELECT on `public.rooms`, `public.customers`
--        - SELECT, UPDATE on `public.room_availability`
--        - SELECT, INSERT, UPDATE on `public.bookings`
--        - SELECT, INSERT, UPDATE, DELETE on `public.booking_rooms` (a fully
--          cancelled room-stay is deleted, not flagged -- see
--          `write_ops.cancel_booking`)
--        - USAGE on the `customers`, `bookings`, `booking_rooms` identity
--          sequences and on the `booking_status` enum
--
-- Usage:
-- - `blue_horizon/load_data/booking_pgsql.py`'s `reload_sql_tables()` now runs
--   this automatically, as its last step, against the same connection it
--   used to reload the data -- so a normal data reload can no longer leave
--   the booking-agent roles without grants. Run it by hand only for an
--   out-of-band fix (e.g. a branch that somehow lost its grants without a
--   reload) or while debugging.
-- - Run this on the branch whose permissions you want to fix, typically
--   immediately after reloading data (a branch's *parent*, per the plan's
--   branch-discipline notes — the loader drops and recreates tables, so
--   running this against a branch that then gets reset from that parent
--   would be wasted).
-- - If your role names differ from `bh_agent_ro` / `bh_agent_rw`, change the
--   two variables below.
-- - If you grant directly to login roles instead of NOLOGIN grant roles, use
--   those login role names here instead.

DO $$
DECLARE
    ro_role text := 'bh_agent_ro';
    rw_role text := 'bh_agent_rw';
BEGIN
    -- --- Schema usage for both roles -----------------------------------
    EXECUTE format('GRANT USAGE ON SCHEMA public TO %I', ro_role);
    EXECUTE format('GRANT USAGE ON SCHEMA public TO %I', rw_role);

    -- --- Enum type usage for both roles ----------------------------------
    EXECUTE format(
        'GRANT USAGE ON TYPE '
        || 'public.room_type, '
        || 'public.room_bed_type, '
        || 'public.room_status_type, '
        || 'public.availability_status_type '
        || 'TO %I',
        ro_role
    );
    EXECUTE format(
        'GRANT USAGE ON TYPE '
        || 'public.room_type, '
        || 'public.room_bed_type, '
        || 'public.room_status_type, '
        || 'public.availability_status_type, '
        || 'public.booking_status '
        || 'TO %I',
        rw_role
    );

    -- --- bh_agent_ro: read-only, rooms/room_availability only ------------
    EXECUTE format('REVOKE ALL PRIVILEGES ON TABLE public.rooms FROM %I', ro_role);
    EXECUTE format(
        'REVOKE ALL PRIVILEGES ON TABLE public.room_availability FROM %I', ro_role
    );
    EXECUTE format(
        'REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA public FROM %I', ro_role
    );
    EXECUTE format(
        'REVOKE ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public FROM %I', ro_role
    );

    EXECUTE format('GRANT SELECT ON TABLE public.rooms TO %I', ro_role);
    EXECUTE format(
        'GRANT SELECT ON TABLE public.room_availability TO %I', ro_role
    );

    -- --- bh_agent_rw: search + booking writes -----------------------------
    EXECUTE format(
        'REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA public FROM %I', rw_role
    );
    EXECUTE format(
        'REVOKE ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public FROM %I', rw_role
    );

    EXECUTE format('GRANT SELECT ON TABLE public.rooms TO %I', rw_role);
    EXECUTE format('GRANT SELECT ON TABLE public.customers TO %I', rw_role);
    EXECUTE format(
        'GRANT SELECT, UPDATE ON TABLE public.room_availability TO %I', rw_role
    );
    EXECUTE format(
        'GRANT SELECT, INSERT, UPDATE ON TABLE public.bookings TO %I', rw_role
    );
    EXECUTE format(
        'GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE public.booking_rooms TO %I',
        rw_role
    );

    EXECUTE format(
        'GRANT USAGE ON SEQUENCE public.bookings_booking_id_seq TO %I', rw_role
    );
    EXECUTE format(
        'GRANT USAGE ON SEQUENCE public.booking_rooms_booking_room_id_seq TO %I',
        rw_role
    );
END $$;
