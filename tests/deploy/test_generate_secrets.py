"""Tests for deploy/generate_secrets.py.

_build_redirect_uri and _derive_cookie_secret are pure functions tested in
isolation.  main() is tested by patching Path.home() to a temporary directory
so no real ~/.streamlit/secrets.toml is written during the test run.
"""

# ruff: noqa: S101, S105, S106, S107

import hashlib
import pathlib

import pytest

from deploy.generate_secrets import (
    _GOOGLE_METADATA_URL,
    _build_redirect_uri,
    _derive_cookie_secret,
    main,
)

_SPACE_VARS = ("SPACE_HOST", "SPACE_AUTHOR_NAME", "SPACE_REPO_NAME")
_AUTH_VARS = ("GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET", "COOKIE_SECRET")


# ---------------------------------------------------------------------------
# _build_redirect_uri
# ---------------------------------------------------------------------------


class TestBuildRedirectUri:
    """_build_redirect_uri constructs the OAuth callback URL from env vars."""

    def test_uses_space_host_when_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SPACE_HOST → https://<host>/oauth2callback."""
        for var in _SPACE_VARS:
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("SPACE_HOST", "myuser-myspace.hf.space")
        assert _build_redirect_uri() == (
            "https://myuser-myspace.hf.space/oauth2callback"
        )

    def test_uses_author_and_repo_when_space_host_absent(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """SPACE_AUTHOR_NAME + SPACE_REPO_NAME → correct hf.space URL."""
        for var in _SPACE_VARS:
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("SPACE_AUTHOR_NAME", "alice")
        monkeypatch.setenv("SPACE_REPO_NAME", "myspace")
        assert _build_redirect_uri() == "https://alice-myspace.hf.space/oauth2callback"

    def test_lowercases_author_and_repo(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Author and repo names are lowercased in the URL."""
        for var in _SPACE_VARS:
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("SPACE_AUTHOR_NAME", "Alice")
        monkeypatch.setenv("SPACE_REPO_NAME", "MySpace")
        assert _build_redirect_uri() == "https://alice-myspace.hf.space/oauth2callback"

    def test_falls_back_to_localhost_when_no_space_vars(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """No Space env vars → localhost fallback for local development."""
        for var in _SPACE_VARS:
            monkeypatch.delenv(var, raising=False)
        assert _build_redirect_uri() == "http://localhost:8501/oauth2callback"


# ---------------------------------------------------------------------------
# _derive_cookie_secret
# ---------------------------------------------------------------------------


class TestDeriveCookieSecret:
    """_derive_cookie_secret returns a stable secret for signing cookies."""

    def test_returns_cookie_secret_env_var_when_set(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """COOKIE_SECRET env var is returned as-is."""
        monkeypatch.setenv("COOKIE_SECRET", "my-explicit-secret")
        assert _derive_cookie_secret("any-client-secret") == "my-explicit-secret"

    def test_derives_from_client_secret_when_env_var_absent(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When COOKIE_SECRET is unset, derives secret from the client secret."""
        monkeypatch.delenv("COOKIE_SECRET", raising=False)
        client_secret = "google-client-secret"
        expected = hashlib.sha256(client_secret.encode()).hexdigest()
        assert _derive_cookie_secret(client_secret) == expected

    def test_derivation_is_deterministic(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The same client secret always produces the same derived secret."""
        monkeypatch.delenv("COOKIE_SECRET", raising=False)
        result_a = _derive_cookie_secret("stable-secret")
        result_b = _derive_cookie_secret("stable-secret")
        assert result_a == result_b


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


class TestMain:
    """main() writes secrets.toml or exits early when credentials are absent."""

    def _run_main(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: pathlib.Path,
        *,
        client_id: str | None = "test-client-id",
        client_secret: str | None = "test-client-secret",
    ) -> pathlib.Path:
        """Set up env and patch Path.home(), then run main().

        Args:
            monkeypatch: Pytest monkeypatch fixture.
            tmp_path: Temporary directory used as the fake home directory.
            client_id: Value for GOOGLE_CLIENT_ID, or None to leave unset.
            client_secret: Value for GOOGLE_CLIENT_SECRET, or None to leave unset.

        Returns:
            The path where secrets.toml would be written.

        """
        for var in _AUTH_VARS + _SPACE_VARS:
            monkeypatch.delenv(var, raising=False)

        if client_id is not None:
            monkeypatch.setenv("GOOGLE_CLIENT_ID", client_id)
        if client_secret is not None:
            monkeypatch.setenv("GOOGLE_CLIENT_SECRET", client_secret)

        monkeypatch.setenv("COOKIE_SECRET", "fixed-cookie-secret")
        monkeypatch.setattr(pathlib.Path, "home", lambda: tmp_path)

        main()
        return tmp_path / ".streamlit" / "secrets.toml"

    def test_no_file_written_when_client_id_absent(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path,
    ) -> None:
        """Missing GOOGLE_CLIENT_ID → secrets.toml is not written."""
        secrets_path = self._run_main(monkeypatch, tmp_path, client_id=None)
        assert not secrets_path.exists()

    def test_no_file_written_when_client_secret_absent(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path,
    ) -> None:
        """Missing GOOGLE_CLIENT_SECRET → secrets.toml is not written."""
        secrets_path = self._run_main(monkeypatch, tmp_path, client_secret=None)
        assert not secrets_path.exists()

    def test_writes_secrets_toml(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path,
    ) -> None:
        """With valid credentials, secrets.toml is created."""
        secrets_path = self._run_main(monkeypatch, tmp_path)
        assert secrets_path.exists()

    def test_toml_contains_client_id(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path,
    ) -> None:
        """Written TOML includes the Google client ID."""
        secrets_path = self._run_main(monkeypatch, tmp_path, client_id="my-client-id")
        assert "my-client-id" in secrets_path.read_text()

    def test_toml_contains_client_secret(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path,
    ) -> None:
        """Written TOML includes the Google client secret."""
        secrets_path = self._run_main(
            monkeypatch, tmp_path, client_secret="my-client-secret",
        )
        assert "my-client-secret" in secrets_path.read_text()

    def test_toml_contains_google_metadata_url(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path,
    ) -> None:
        """Written TOML references the Google OIDC discovery document URL."""
        secrets_path = self._run_main(monkeypatch, tmp_path)
        assert _GOOGLE_METADATA_URL in secrets_path.read_text()

    def test_toml_contains_redirect_uri(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path,
    ) -> None:
        """Written TOML includes a redirect_uri entry."""
        secrets_path = self._run_main(monkeypatch, tmp_path)
        assert "redirect_uri" in secrets_path.read_text()
