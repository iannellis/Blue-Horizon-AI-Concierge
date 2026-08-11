"""Tests for SQL guardrail validation logic.

All tests are pure: no I/O, no mocking.
"""


import pytest

from blue_horizon.agents.booking.guardrails import validate_sql

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ALLOW = True   # allow_only_hotel_tables=True
_ANY   = False  # allow_only_hotel_tables=False


# ---------------------------------------------------------------------------
# Single-statement check
# ---------------------------------------------------------------------------


class TestStatementSeparator:
    """validate_sql must block multi-statement input."""

    def test_multi_statement_blocked(self) -> None:
        """Semicolon between two statements is rejected."""
        with pytest.raises(ValueError, match="single SQL statement"):
            validate_sql("SELECT 1; DROP TABLE rooms", allow_only_hotel_tables=_ANY)

    def test_trailing_semicolon_is_fine(self) -> None:
        """A trailing semicolon with no second statement passes."""
        validate_sql("SELECT 1 FROM rooms;", allow_only_hotel_tables=_ANY)

    def test_semicolon_inside_string_literal_is_fine(self) -> None:
        """Semicolons inside string literals are not treated as separators."""
        validate_sql(
            "SELECT * FROM rooms WHERE name = 'hello; world'",
            allow_only_hotel_tables=_ALLOW,
        )

    def test_semicolon_inside_block_comment_is_fine(self) -> None:
        """Semicolons inside block comments are not treated as separators."""
        validate_sql(
            "/* step 1; step 2 */ SELECT * FROM rooms",
            allow_only_hotel_tables=_ALLOW,
        )


# ---------------------------------------------------------------------------
# Forbidden token checks
# ---------------------------------------------------------------------------


class TestForbiddenTokens:
    """validate_sql must block known catalog/extension escape hatches."""

    @pytest.mark.parametrize(
        "token",
        [
            "pg_sleep",
            "information_schema",
            "pg_catalog",
            "dblink",
        ],
    )
    def test_forbidden_token_blocked(self, token: str) -> None:
        """Each forbidden token causes a ValueError."""
        query = f"SELECT {token}(1) FROM rooms"  # noqa: S608
        with pytest.raises(ValueError, match="Forbidden SQL token"):
            validate_sql(query, allow_only_hotel_tables=_ANY)

    def test_forbidden_token_case_insensitive(self) -> None:
        """Token check is case-insensitive."""
        with pytest.raises(ValueError, match="Forbidden SQL token"):
            validate_sql("SELECT PG_SLEEP(5) FROM rooms", allow_only_hotel_tables=_ANY)


# ---------------------------------------------------------------------------
# DDL / privileged keyword checks
# ---------------------------------------------------------------------------


class TestUnsupportedStatements:
    """validate_sql must reject statement types outside the runtime contract."""

    @pytest.mark.parametrize(
        "statement",
        [
            "CREATE TABLE evil (id INT)",
            "DROP TABLE rooms",
            "ALTER TABLE rooms ADD COLUMN x INT",
            "TRUNCATE rooms",
            "GRANT ALL ON rooms TO attacker",
            "REVOKE SELECT ON rooms FROM user1",
            "VACUUM rooms",
            "ANALYZE rooms",
            "COPY rooms TO '/tmp/dump.csv'",
            "DO $$ BEGIN NULL; END $$",
            "CALL some_proc()",
            "EXECUTE 'SELECT 1'",
        ],
    )
    def test_unsupported_statement_blocked(self, statement: str) -> None:
        """Non-runtime statements are always blocked regardless of table policy."""
        with pytest.raises(ValueError, match="Only SELECT statements"):
            validate_sql(statement, allow_only_hotel_tables=_ANY)

    def test_ddl_keyword_case_insensitive(self) -> None:
        """Statement-type blocking is case-insensitive."""
        with pytest.raises(ValueError, match="Only SELECT statements"):
            validate_sql("drop table rooms", allow_only_hotel_tables=_ANY)


# ---------------------------------------------------------------------------
# Valid queries that should pass
# ---------------------------------------------------------------------------


class TestValidQueries:
    """validate_sql must not raise for legitimate read queries."""

    def test_simple_select_passes(self) -> None:
        """A plain SELECT passes all checks."""
        validate_sql("SELECT id, name FROM rooms", allow_only_hotel_tables=_ALLOW)

    def test_select_with_where_passes(self) -> None:
        """SELECT with WHERE clause passes."""
        validate_sql(
            "SELECT * FROM rooms WHERE status = 'available'",
            allow_only_hotel_tables=_ALLOW,
        )

    def test_join_on_allowed_tables_passes(self) -> None:
        """JOIN across the two allowed tables passes."""
        validate_sql(
            "SELECT r.id FROM rooms r JOIN room_availability ra ON r.id = ra.room_id",
            allow_only_hotel_tables=_ALLOW,
        )

    def test_room_availability_allowed(self) -> None:
        """room_availability is on the allowlist."""
        validate_sql(
            "SELECT * FROM room_availability WHERE room_id = 1",
            allow_only_hotel_tables=_ALLOW,
        )

    def test_select_with_set_returning_function_passes(self) -> None:
        """generate_series() after FROM is not treated as a disallowed table."""
        validate_sql(
            "SELECT * FROM rooms, generate_series(1, 5) AS gs",
            allow_only_hotel_tables=_ALLOW,
        )


# ---------------------------------------------------------------------------
# Table allowlist enforcement
# ---------------------------------------------------------------------------


class TestTableAllowlist:
    """Table restriction applies when allow_only_hotel_tables=True."""

    def test_disallowed_table_blocked(self) -> None:
        """References to tables outside the allowlist are rejected."""
        with pytest.raises(ValueError, match="Table not allowed"):
            validate_sql(
                "SELECT * FROM users",
                allow_only_hotel_tables=_ALLOW,
            )

    def test_schema_qualified_disallowed_table_blocked(self) -> None:
        """Schema-qualified references to disallowed tables are rejected."""
        with pytest.raises(ValueError, match="Table not allowed"):
            validate_sql(
                "SELECT * FROM public.users",
                allow_only_hotel_tables=_ALLOW,
            )

    def test_table_restriction_disabled_allows_other_tables(self) -> None:
        """With allow_only_hotel_tables=False any table is allowed."""
        validate_sql(
            "SELECT * FROM some_other_table",
            allow_only_hotel_tables=_ANY,
        )

    def test_quoted_allowed_table_passes(self) -> None:
        """Double-quoted allowed table names are normalised correctly."""
        validate_sql(
            'SELECT * FROM "rooms"',
            allow_only_hotel_tables=_ALLOW,
        )

    def test_quoted_disallowed_table_blocked(self) -> None:
        """Double-quoted disallowed table names must be rejected."""
        with pytest.raises(ValueError, match="Table not allowed"):
            validate_sql('SELECT * FROM "users"', allow_only_hotel_tables=_ALLOW)

    @pytest.mark.parametrize("table", ["customers", "bookings", "booking_rooms"])
    def test_write_ops_tables_blocked(self, table: str) -> None:
        """The write-only tables are deliberately off the allowlist.

        `run_sql` must never reach `customers`, `bookings`, or `booking_rooms`
        -- guest identity and reservation state come only from `write_ops`
        via the `bh_agent_rw` role, never from free-form SQL the model
        authors.
        """
        query = f"SELECT * FROM {table}"  # noqa: S608
        with pytest.raises(ValueError, match="Table not allowed"):
            validate_sql(query, allow_only_hotel_tables=_ALLOW)


# ---------------------------------------------------------------------------
# CTE (WITH clause) handling
# ---------------------------------------------------------------------------


class TestCteHandling:
    """CTE names are added to the allowed set and must not be flagged."""

    def test_cte_name_bypasses_table_restriction(self) -> None:
        """A CTE defined in WITH is exempt from the table allowlist."""
        validate_sql(
            """
            WITH available AS (
                SELECT * FROM rooms WHERE status = 'available'
            )
            SELECT * FROM available
            """,
            allow_only_hotel_tables=_ALLOW,
        )

    def test_cte_with_recursive_passes(self) -> None:
        """RECURSIVE CTEs are also handled."""
        validate_sql(
            """
            WITH RECURSIVE cte AS (
                SELECT id FROM rooms WHERE id = 1
                UNION ALL
                SELECT r.id FROM rooms r JOIN cte ON r.id = cte.id + 1
            )
            SELECT * FROM cte
            """,
            allow_only_hotel_tables=_ALLOW,
        )

    def test_multiple_ctes_all_pass(self) -> None:
        """Multiple CTE names are all added to the allowed set."""
        validate_sql(
            """
            WITH a AS (SELECT * FROM rooms),
                 b AS (SELECT * FROM room_availability)
            SELECT * FROM a JOIN b ON a.id = b.room_id
            """,
            allow_only_hotel_tables=_ALLOW,
        )

    def test_cte_referencing_disallowed_base_table_blocked(self) -> None:
        """The CTE body may only reference allowed tables."""
        with pytest.raises(ValueError, match="Table not allowed"):
            validate_sql(
                """
                WITH evil AS (SELECT * FROM users)
                SELECT * FROM evil
                """,
                allow_only_hotel_tables=_ALLOW,
            )


# ---------------------------------------------------------------------------
# Regression coverage for current regex limitations
# ---------------------------------------------------------------------------


class TestRegexRegressionCoverage:
    """Regression tests for the known regex limitations in validate_sql."""

    def test_privileged_keyword_inside_string_literal_passes(self) -> None:
        """Keyword text inside a string literal must not be treated as DDL."""
        validate_sql(
            "SELECT 'drop' AS note FROM rooms",
            allow_only_hotel_tables=_ALLOW,
        )

    def test_update_only_syntax_still_blocked_as_a_write(self) -> None:
        """ONLY is valid PostgreSQL syntax, but UPDATE itself is now rejected.

        Writes moved entirely to `write_ops` in Phase 1 -- `run_sql` is
        read-only, so this now fails the statement-type check before the
        allowlist is ever consulted.
        """
        with pytest.raises(ValueError, match="Only SELECT statements"):
            validate_sql(
                "UPDATE ONLY room_availability SET status = 'Booked' WHERE room_id = 1",
                allow_only_hotel_tables=_ALLOW,
            )

    def test_update_from_disallowed_table_blocked_as_a_write(self) -> None:
        """UPDATE is rejected as a write before its FROM clause is even checked."""
        with pytest.raises(ValueError, match="Only SELECT statements"):
            validate_sql(
                (
                    "UPDATE room_availability AS ra "
                    "SET status = 'Booked' "
                    "FROM audit_log AS al "
                    "WHERE ra.room_id = al.room_id"
                ),
                allow_only_hotel_tables=_ALLOW,
            )
