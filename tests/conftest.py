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

It must call `load_dotenv()` itself, first, rather than relying on the
environment already having these variables when pytest starts. Without that
call, nothing in a plain local `pytest` invocation ever reads `.env` before
this hook runs -- `pytest_configure` fires before test collection, so it
sees whatever was already exported in the shell and nothing more, finds
every `_EVAL` variable unset, and overrides nothing. The variables that were
actually missing were only ever populated later and by accident, by whichever
`db_integration` test happened to import a module with its own module-level
`load_dotenv()` call first (e.g. `tests/api/test_app.py` importing
`blue_horizon.api.app`) -- and `load_dotenv()`'s default behaviour is to fill
in only variables that are still unset, so at that point it wrote `.env`'s
literal `PGSQL_RW_DB_URL`/`PGSQL_RO_DB_URL` (Production) straight into
`os.environ`, with this hook's override already having had its one chance
and missed it. `PGSQL_ROOT_DB_URL` has no literal entry in `.env` at all --
only this hook's derivation from `PGSQL_ROOT_EVAL_DB_URL` can ever set it --
so that failure mode was silent for the read/write variables and only
visibly broke (as a permanent skip) for the root one. Calling `load_dotenv()`
here first closes that window: this hook now always sees `.env`'s contents
before deciding whether to override them.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

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
    # Must run before the override loop below, and before any test module
    # gets the chance to call load_dotenv() on its own -- see this module's
    # docstring for the failure mode that created.
    load_dotenv()

    for target_var, eval_var in _EVAL_URL_OVERRIDES.items():
        eval_url = os.environ.get(eval_var)
        if eval_url:
            os.environ[target_var] = eval_url
