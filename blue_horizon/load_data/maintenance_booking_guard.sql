-- Trigger refusing any `booking_rooms` insert or update that would cover a
-- night `room_availability` marks `Maintenance`.
--
-- Belt-and-suspenders alongside `booking_rooms_no_overlap` (see
-- `schema.sql`): `write_ops._price_one_room` already refuses to price any
-- night whose `room_availability.status` isn't `Available`, so a real
-- `commit_booking`/`modify_booking` call should never reach this trigger.
-- It exists as a schema-level backstop in case `booking_rooms` and
-- `room_availability` ever drift out of sync, or a row is written some
-- other way than through `write_ops`.
--
-- Executed by
-- `blue_horizon.load_data.booking_pgsql.setup_maintenance_booking_guard`,
-- deliberately kept out of `schema.sql` and installed as a separate,
-- later step in `reload_sql_tables`: it runs only after pre-existing
-- bookings are loaded and `room_availability` is reconciled against them.
-- If this trigger were active during that initial load, a pre-existing
-- booking touching a Maintenance-status night in the seed data would be
-- rejected outright -- silently dropping real historical bookings, instead
-- of just guarding future live writes made through `write_ops`.

DROP TRIGGER IF EXISTS booking_rooms_no_maintenance ON booking_rooms;
DROP FUNCTION IF EXISTS prevent_maintenance_booking();

CREATE FUNCTION prevent_maintenance_booking()
RETURNS TRIGGER AS $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM room_availability
        WHERE room_id = NEW.room_id
          AND date >= NEW.check_in
          AND date < NEW.check_out
          AND status = 'Maintenance'
    ) THEN
        RAISE EXCEPTION
            'Room % is under maintenance for one or more nights in [%, %)',
            NEW.room_id, NEW.check_in, NEW.check_out
            USING ERRCODE = 'check_violation';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER booking_rooms_no_maintenance
BEFORE INSERT OR UPDATE ON booking_rooms
FOR EACH ROW
EXECUTE FUNCTION prevent_maintenance_booking();
