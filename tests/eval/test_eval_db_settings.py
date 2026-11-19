"""Tests for the eval database URL overrides in `eval.config`.

No database connection is required: only environment parsing is exercised.
"""

# ruff: noqa: S101

import pytest
from pydantic import ValidationError

from eval.config import _EvalDbSettings  # pyright: ignore[reportPrivateUsage]

_OVERRIDE_VARS = ("PGSQL_RW_EVAL_DB_URL", "PGSQL_RO_EVAL_DB_URL")


class TestEvalDbOverrides:
    """A blank override is an error, not the same as an unset one.

    Callers combine an override with ``override or app_default``, so a blank
    value accepted here would silently fall back to the application URL,
    which in a local ``.env`` points at Production.
    """

    @pytest.mark.parametrize("env_var", _OVERRIDE_VARS)
    def test_blank_override_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch, env_var: str,
    ) -> None:
        """Setting an override to an empty string fails validation."""
        monkeypatch.setenv(env_var, "")
        with pytest.raises(ValidationError):
            _EvalDbSettings(_env_file=None)  # pyright: ignore[reportCallIssue]

    def test_unset_overrides_stay_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With neither variable set, both overrides are ``None``."""
        for env_var in _OVERRIDE_VARS:
            monkeypatch.delenv(env_var, raising=False)
        settings = _EvalDbSettings(_env_file=None)  # pyright: ignore[reportCallIssue]
        assert settings.pgsql_rw_eval_db_url is None
        assert settings.pgsql_ro_eval_db_url is None
