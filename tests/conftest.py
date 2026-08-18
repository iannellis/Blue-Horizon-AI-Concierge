"""Shared pytest configuration for the whole test suite.

Protects the Production Neon branch from `db_integration`-marked tests run
locally. Those tests (`tests/booking/test_write_ops.py`,
`tests/api/test_app.py`) read `PGSQL_RW_DB_URL`/`PGSQL_RO_DB_URL` directly,
which is also what the running app uses -- so on a developer machine, where
`.env` points those at the `Production` branch, a local `pytest -m
db_integration` silently books and cancels real rows there. `PGSQL_ROOT_DB_URL`
(read only by `tests/booking/test_db_invariants.py`'s `root_db_url` fixture,
for the one test that needs DDL privilege) carries the same risk if it were
ever pointed at Production directly -- see `blue_horizon/config.py`'s
`pgsql_root_parent_db_url`, which is deliberately a *different* variable
reserved for the Parent branch, precisely so it can never collide with this
one. In CI this is a non-issue: the `db-integration-tests` job in
`.github/workflows/ci.yml` explicitly maps `PGSQL_RW_EVAL_DB_URL`/
`PGSQL_RO_EVAL_DB_URL`/`PGSQL_ROOT_EVAL_DB_URL` onto `PGSQL_RW_DB_URL`/
`PGSQL_RO_DB_URL`/`PGSQL_ROOT_DB_URL` for that job and resets the
`Development` branch first.

This hook reproduces that same override locally: when
`PGSQL_RW_EVAL_DB_URL`/`PGSQL_RO_EVAL_DB_URL`/`PGSQL_ROOT_EVAL_DB_URL` are
present in the environment, it copies each onto its plain-named counterpart
before any test reads them, so `db_integration` tests land on the
`Development` branch by default instead. It is a no-op wherever the eval URL
is not set (e.g. the `unit-tests` CI job, which never defines any of them, or
a developer who hasn't added `PGSQL_ROOT_EVAL_DB_URL` locally -- in which
case `root_db_url`-gated tests just skip, same as if `PGSQL_ROOT_DB_URL` were
never set at all).
"""

from __future__ import annotations

import os

_EVAL_URL_OVERRIDES = {
    "PGSQL_RW_DB_URL": "PGSQL_RW_EVAL_DB_URL",
    "PGSQL_RO_DB_URL": "PGSQL_RO_EVAL_DB_URL",
    "PGSQL_ROOT_DB_URL": "PGSQL_ROOT_EVAL_DB_URL",
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
