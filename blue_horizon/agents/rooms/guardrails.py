"""SQL guardrails for the rooms agent.

Validates SQL statements against conservative security rules before execution.
Supports booking/cancel write operations while blocking DDL, catalog access,
and multi-statement calls.
"""

from __future__ import annotations

import re
from typing import Final

import sqlparse

_FORBIDDEN_TOKENS: Final[tuple[str, ...]] = (
    "pg_sleep",
    "information_schema",
    "pg_catalog",
    "dblink",
)

_FORBIDDEN_KEYWORDS = re.compile(
    r"\b("
    r"alter|create|drop|truncate|grant|revoke|vacuum|analyze|"
    r"copy|do|call|execute"
    r")\b",
    re.IGNORECASE,
)

_ALLOWED_TABLES: Final[set[str]] = {"rooms", "room_availability"}

_IDENTIFIER = r'(?:"[^"]+"|[a-zA-Z_][a-zA-Z0-9_]*)'
# Matches table/view references after FROM/JOIN/UPDATE/INTO.
# The negative lookahead (?!\s*\() excludes set-returning function calls such as
# generate_series(...) or unnest(...), which appear syntactically in the same
# position as table names but are not subject to the table allowlist.
_TABLE_REF = re.compile(
    rf"\b(from|join|update|into)\s+({_IDENTIFIER})(?:\s*\.\s*({_IDENTIFIER}))?(?!\s*\()\b",
    re.IGNORECASE,
)


def validate_sql(query: str, *, allow_only_hotel_tables: bool) -> None:
    """Validate a SQL statement against conservative guardrails.

    Guardrails enforced:
      - Exactly one statement per call (no statement separators).
      - Blocks common catalog/extension escape hatches.
      - Blocks DDL and privileged commands.
      - Optionally restricts table references to an allowlist (CTEs are allowed).

    Args:
        query: SQL statement to validate.
        allow_only_hotel_tables: Whether to enforce a table allowlist.

    Raises:
        ValueError: If the SQL violates any guardrail.

    """
    if _contains_statement_separator(query):
        msg = (
            "Only a single SQL statement is allowed per call (no semicolons). "
            "For multi-step workflows, call the tool multiple times."
        )
        raise ValueError(msg)

    q = _normalize_sql(query)
    q_lower = q.lower()

    for token in _FORBIDDEN_TOKENS:
        if token in q_lower:
            msg = f"Forbidden SQL token detected: {token}"
            raise ValueError(msg)

    if _FORBIDDEN_KEYWORDS.search(q):
        msg = "DDL/privileged SQL statements are not allowed."
        raise ValueError(msg)

    if allow_only_hotel_tables:
        cte_names = _extract_cte_names(query)
        allowed = _ALLOWED_TABLES | cte_names

        for _, part1, part2 in _TABLE_REF.findall(q):
            table_token = part2 or part1
            table = _normalize_identifier(table_token)
            if table not in allowed:
                msg = f"Table not allowed: {table_token}"
                raise ValueError(msg)


def _is_write_sql(query: str) -> bool:
    """Determine whether the SQL statement looks like a write (DML).

    Args:
        query: SQL statement.

    Returns:
        True if the statement contains INSERT, UPDATE, or DELETE.

    """
    q = _normalize_sql(query).lower()
    return bool(re.search(r"\b(insert|update|delete)\b", q))


def _extract_cte_names(query: str) -> set[str]:
    """Extract CTE names from WITH clauses in a SQL query.

    Args:
        query: SQL query string.

    Returns:
        Set of normalized CTE names found in the query.

    """
    cte_pattern = re.compile(
        r"(?:\bWITH\s+(?:RECURSIVE\s+)?|,\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s+AS\s*\(",
        re.IGNORECASE,
    )
    return {
        _normalize_identifier(match.group(1)) for match in cte_pattern.finditer(query)
    }


def _normalize_sql(query: str) -> str:
    """Normalize SQL text for lightweight validation.

    Args:
        query: Raw SQL string.

    Returns:
        SQL with collapsed whitespace.

    """
    return " ".join(query.strip().split())


def _normalize_identifier(identifier: str) -> str:
    """Normalize a SQL identifier for comparisons.

    Args:
        identifier: Identifier token, possibly quoted.

    Returns:
        Normalized identifier name.

    """
    token = identifier.strip()
    if token.startswith('"') and token.endswith('"') and len(token) >= 2:  # noqa: PLR2004
        token = token[1:-1]
    return token.lower()


def _contains_statement_separator(query: str) -> bool:
    """Detect whether a SQL string contains a statement separator.

    Delegates to ``sqlparse.split`` which correctly handles semicolons inside
    string literals, quoted identifiers, line comments (``--``), and block
    comments (``/* … */``).

    Args:
        query: Raw SQL string.

    Returns:
        True if more than one non-empty statement is found; otherwise False.

    """
    return len([s for s in sqlparse.split(query) if s.strip()]) > 1
