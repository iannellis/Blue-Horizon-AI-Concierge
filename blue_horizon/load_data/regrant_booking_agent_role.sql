-- Reapply least-privilege grants for the rooms runtime agent role after
-- `blue_horizon/load_data/booking_pgsql.py` recreates the `rooms`,
-- `room_availability`, and enum-type objects.
--
-- What it does:
-- 1. Ensures the target role can use the `public` schema.
-- 2. Re-grants USAGE on the room-related enum types recreated by the loader.
-- 3. Removes any table/sequence privileges that exceed the runtime contract.
-- 4. Grants exactly the permissions the rooms agent needs:
--      - SELECT on `public.rooms`
--      - SELECT, UPDATE on `public.room_availability`
--
-- Usage:
-- - Run this on the branch whose permissions you want to fix, typically
--   `production` immediately after reloading rooms data.
-- - If your grant role is not named `bh_agent_rw`, change `grant_role` below.
-- - If you grant directly to a login role instead of a NOLOGIN grant role,
--   use that login role name here instead.

DO $$
DECLARE
    grant_role text := 'bh_agent_rw';
BEGIN
    EXECUTE format(
        'GRANT USAGE ON SCHEMA public TO %I',
        grant_role
    );

    EXECUTE format(
        'GRANT USAGE ON TYPE '
        || 'public.room_type, '
        || 'public.room_bed_type, '
        || 'public.room_status_type, '
        || 'public.availability_status_type '
        || 'TO %I',
        grant_role
    );

    EXECUTE format(
        'REVOKE ALL PRIVILEGES ON TABLE public.rooms FROM %I',
        grant_role
    );
    EXECUTE format(
        'REVOKE ALL PRIVILEGES ON TABLE public.room_availability FROM %I',
        grant_role
    );
    EXECUTE format(
        'REVOKE ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public FROM %I',
        grant_role
    );

    EXECUTE format(
        'GRANT SELECT ON TABLE public.rooms TO %I',
        grant_role
    );
    EXECUTE format(
        'GRANT SELECT, UPDATE ON TABLE public.room_availability TO %I',
        grant_role
    );
END $$;
