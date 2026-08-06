"""SQL guardrails for the booking agent.

Validates SQL statements against conservative security rules before execution
using a PostgreSQL-aware AST parser. Supports room search plus booking-state
updates while blocking unsupported statements, catalog access, and
multi-statement input.
"""

from __future__ import annotations

from typing import Final

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

_ALLOWED_TABLES: Final[set[str]] = {"rooms", "room_availability"}
_ALLOWED_SCHEMAS: Final[set[str]] = {"public"}
_FORBIDDEN_FUNCTIONS: Final[set[str]] = {
    "dblink",
    "information_schema",
    "pg_catalog",
    "pg_sleep",
}
_FORBIDDEN_SCHEMAS: Final[set[str]] = {"information_schema", "pg_catalog"}
_ALLOWED_STATEMENT_KEYS: Final[set[str]] = {"select", "update"}
_WRITE_STATEMENT_KEYS: Final[set[str]] = {"update"}


def validate_sql(query: str, *, allow_only_hotel_tables: bool) -> None:
    """Validate a SQL statement against conservative guardrails.

    Guardrails enforced:
      - Exactly one statement per call.
      - Blocks disallowed function calls such as ``pg_sleep`` and ``dblink``.
      - Blocks system-catalog access.
      - Blocks all statement types other than ``SELECT`` and ``UPDATE``.
      - Optionally restricts table references to a hotel-table allowlist, with
        CTE names treated as allowed references.

    Args:
        query: SQL statement to validate.
        allow_only_hotel_tables: Whether to enforce a table allowlist.

    Raises:
        ValueError: If the SQL violates any guardrail.

    """
    statement = _parse_single_statement(query)
    _validate_statement_type(statement)
    _validate_forbidden_functions(statement)
    _validate_forbidden_schemas(statement)

    if allow_only_hotel_tables:
        _validate_table_allowlist(statement)


def _is_write_sql(query: str) -> bool:
    """Determine whether the SQL statement is a write statement.

    Args:
        query: SQL statement.

    Returns:
        True when the parsed statement is ``UPDATE``. Returns False when
        parsing fails or the statement is not write-oriented.

    """
    try:
        statement = _parse_single_statement(query)
    except ValueError:
        return False

    return statement.key in _WRITE_STATEMENT_KEYS


def _parse_single_statement(query: str) -> exp.Expr:
    """Parse a SQL string and require exactly one statement.

    Args:
        query: Raw SQL string.

    Returns:
        The parsed SQL expression.

    Raises:
        ValueError: If parsing fails or the input contains zero or multiple
            statements.

    """
    try:
        statements = [
            statement
            for statement in sqlglot.parse(query, read="postgres")
            if statement is not None
        ]
    except ParseError as exc:
        msg = "Invalid SQL statement."
        raise ValueError(msg) from exc

    if len(statements) != 1:
        msg = (
            "Only a single SQL statement is allowed per call (no semicolons). "
            "For multi-step workflows, call the tool multiple times."
        )
        raise ValueError(msg)

    return statements[0]


def _validate_statement_type(statement: exp.Expr) -> None:
    """Reject statement types outside the allowed DML/read subset.

    Args:
        statement: Parsed SQL expression.

    Raises:
        ValueError: If the parsed statement is not allowed for the booking agent.

    """
    if statement.key not in _ALLOWED_STATEMENT_KEYS:
        msg = "Only SELECT and UPDATE statements are allowed."
        raise ValueError(msg)


def _validate_forbidden_functions(statement: exp.Expr) -> None:
    """Reject forbidden function calls found anywhere in the statement.

    Args:
        statement: Parsed SQL expression.

    Raises:
        ValueError: If a forbidden function call is present.

    """
    for function in statement.find_all(exp.Func):
        function_name = (function.name or function.sql_name()).lower()
        if function_name in _FORBIDDEN_FUNCTIONS:
            msg = f"Forbidden SQL token detected: {function_name}"
            raise ValueError(msg)


def _validate_forbidden_schemas(statement: exp.Expr) -> None:
    """Reject references to forbidden schemas.

    Args:
        statement: Parsed SQL expression.

    Raises:
        ValueError: If a forbidden schema is referenced by a table expression.

    """
    for table in statement.find_all(exp.Table):
        schema_name = table.db.lower() if table.db else ""
        if schema_name in _FORBIDDEN_SCHEMAS:
            msg = f"Forbidden SQL token detected: {schema_name}"
            raise ValueError(msg)


def _validate_table_allowlist(statement: exp.Expr) -> None:
    """Reject references to tables outside the hotel-table allowlist.

    Args:
        statement: Parsed SQL expression.

    Raises:
        ValueError: If the statement references a disallowed schema or table.

    """
    allowed = {
        cte.alias_or_name.lower()
        for cte in statement.find_all(exp.CTE)
        if cte.alias_or_name
    }
    allowed |= _ALLOWED_TABLES

    for table in statement.find_all(exp.Table):
        table_name = table.name.lower() if table.name else ""
        if not table_name:
            continue

        if table_name in allowed:
            continue

        schema_name = table.db.lower() if table.db else ""
        if schema_name and schema_name not in _ALLOWED_SCHEMAS:
            msg = f"Table not allowed: {table.sql(dialect='postgres')}"
            raise ValueError(msg)

        msg = f"Table not allowed: {table.sql(dialect='postgres')}"
        raise ValueError(msg)
