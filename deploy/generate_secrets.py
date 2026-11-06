"""Generate secrets.toml from environment variables at container startup.

Intended to run once before Streamlit starts (see supervisord.conf).  Reads
OAuth credentials stored as HuggingFace Space secrets and writes them into
the format Streamlit's native auth system expects: shared settings under
``[auth]`` and one ``[auth.<provider>]`` section per configured provider,
which the UI passes by name to ``st.login``.

If no provider's credentials are present the script exits silently, allowing
local development to proceed without authentication.

Environment variables read:
    GOOGLE_CLIENT_ID: Google OAuth 2.0 client ID.
    GOOGLE_CLIENT_SECRET: Google OAuth 2.0 client secret.
    OAUTH_CLIENT_ID: Hugging Face OAuth client ID.  Injected by the platform
        when the Space's metadata sets ``hf_oauth: true``.
    OAUTH_CLIENT_SECRET: Hugging Face OAuth client secret, injected likewise.
    COOKIE_SECRET: Secret used to sign Streamlit session cookies.  Falls back
        to a SHA-256 hash of the first configured client secret when not
        set, so container restarts do not invalidate existing sessions.
    SPACE_HOST: Full HuggingFace Spaces hostname injected by the platform
        (e.g. ``username-spacename.hf.space``).  Used to build the OAuth
        redirect URI.  When absent, ``SPACE_AUTHOR_NAME`` and
        ``SPACE_REPO_NAME`` are used instead.  Falls back to localhost for
        local development.
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import textwrap

_GOOGLE_METADATA_URL: str = (
    "https://accounts.google.com/.well-known/openid-configuration"
)
_HF_METADATA_URL: str = "https://huggingface.co/.well-known/openid-configuration"

# Provider name (as passed to st.login), client ID var, client secret var,
# OIDC discovery URL. Order is the order of the sign-in buttons.
_PROVIDERS: tuple[tuple[str, str, str, str], ...] = (
    ("google", "GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET", _GOOGLE_METADATA_URL),
    ("huggingface", "OAUTH_CLIENT_ID", "OAUTH_CLIENT_SECRET", _HF_METADATA_URL),
)


def _build_redirect_uri() -> str:
    """Construct the OAuth callback URL for this deployment.

    Returns:
        The fully-qualified redirect URI that a provider should send users
        back to after a successful login.

    """
    if space_host := os.environ.get("SPACE_HOST"):
        return f"https://{space_host}/oauth2callback"

    author = os.environ.get("SPACE_AUTHOR_NAME", "")
    repo = os.environ.get("SPACE_REPO_NAME", "")
    if author and repo:
        return f"https://{author.lower()}-{repo.lower()}.hf.space/oauth2callback"

    return "http://localhost:8501/oauth2callback"


def _derive_cookie_secret(client_secret: str) -> str:
    """Return a stable cookie-signing secret.

    Uses the ``COOKIE_SECRET`` environment variable when set; otherwise
    derives a deterministic value from *client_secret* so that container
    restarts do not invalidate active sessions.

    Args:
        client_secret: An OAuth client secret, used as entropy when no
            dedicated ``COOKIE_SECRET`` is configured.

    Returns:
        A hex string suitable for use as Streamlit's ``cookie_secret``.

    """
    if explicit := os.environ.get("COOKIE_SECRET"):
        return explicit
    return hashlib.sha256(client_secret.encode()).hexdigest()


def main() -> None:
    """Write secrets.toml to the locations Streamlit checks for auth config.

    Writes to ``~/.streamlit/`` (global) and ``ui/.streamlit/`` (script-
    relative), covering both lookup paths Streamlit uses.

    A provider is written only when both its client ID and secret are set.
    Exits without writing if none is, so local development without
    authentication is unaffected.

    """
    configured = [
        (name, os.environ[id_var], os.environ[secret_var], metadata_url)
        for name, id_var, secret_var, metadata_url in _PROVIDERS
        if os.environ.get(id_var) and os.environ.get(secret_var)
    ]
    if not configured:
        return

    content = textwrap.dedent(f"""\
        [auth]
        redirect_uri = "{_build_redirect_uri()}"
        cookie_secret = "{_derive_cookie_secret(configured[0][2])}"
    """)
    for name, client_id, client_secret, metadata_url in configured:
        content += textwrap.dedent(f"""
            [auth.{name}]
            client_id = "{client_id}"
            client_secret = "{client_secret}"
            server_metadata_url = "{metadata_url}"
        """)

    app_root = pathlib.Path(__file__).resolve().parent.parent
    for secrets_dir in (
        pathlib.Path.home() / ".streamlit",
        app_root / "ui" / ".streamlit",
    ):
        secrets_dir.mkdir(parents=True, exist_ok=True)
        (secrets_dir / "secrets.toml").write_text(content)


if __name__ == "__main__":
    main()
