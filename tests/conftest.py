"""Shared pytest configuration for the whole test suite.

Protects the Production Neon branch from `db_integration`-marked tests run
locally. Those tests (`tests/booking/test_write_ops.py`,
`tests/api/test_app.py`) read `PGSQL_RW_DB_URL`/`PGSQL_RO_DB_URL` directly,
which is also what the running app uses -- so on a developer machine, where
`.env` points those at the `Production` branch, a local `pytest -m
db_integration` silently books and cancels real rows there. In CI this is a
non-issue: the `db-integration-tests` job in `.github/workflows/ci.yml`
explicitly maps `PGSQL_RW_EVAL_DB_URL`/`PGSQL_RO_EVAL_DB_URL` onto
`PGSQL_RW_DB_URL`/`PGSQL_RO_DB_URL` for that job and resets the `Development`
branch first.

This hook reproduces that same override locally: when
`PGSQL_RW_EVAL_DB_URL`/`PGSQL_RO_EVAL_DB_URL` are present in the environment,
it copies them onto `PGSQL_RW_DB_URL`/`PGSQL_RO_DB_URL` before any test
reads them, so `db_integration` tests land on the `Development` branch by
default instead. It is a no-op wherever the eval URLs are not set (e.g. the
`unit-tests` CI job, which never defines them).
"""

from __future__ import annotations

import os

_EVAL_URL_OVERRIDES = {
    "PGSQL_RW_DB_URL": "PGSQL_RW_EVAL_DB_URL",
    "PGSQL_RO_DB_URL": "PGSQL_RO_EVAL_DB_URL",
}


def pytest_configure(config: object) -> None:  # noqa: ARG001
    """Point `db_integration` tests at the eval database, if configured.

    Args:
        config: The pytest `Config` object (unused; required by the
            `pytest_configure` hook signature).

    """
    for target_var, eval_var in _EVAL_URL_OVERRIDES.items():
        eval_url = os.environ.get(eval_var)
        if eval_url:
            os.environ[target_var] = eval_url
