"""Generate ~/.streamlit/secrets.toml from environment variables at container startup.

Intended to run once before Streamlit starts (see supervisord.conf).  Reads
Google OAuth credentials stored as HuggingFace Space secrets and writes them
into the format Streamlit's native auth system expects.

If the required credentials are absent the script exits silently, allowing
local development to proceed without authentication.

Environment variables read:
    GOOGLE_CLIENT_ID: Google OAuth 2.0 client ID.
    GOOGLE_CLIENT_SECRET: Google OAuth 2.0 client secret.
    COOKIE_SECRET: Secret used to sign Streamlit session cookies.  Falls back
        to a SHA-256 hash of ``GOOGLE_CLIENT_SECRET`` when not set, so
        container restarts do not invalidate existing sessions.
    SPACE_HOST: Full HuggingFace Spaces hostname injected by the platform
        (e.g. ``username-spacename.hf.space``).  Used to build the OAuth
        redirect URI.  When absent, ``SPACE_AUTHOR_NAME`` and
        ``SPACE_REPO_NAME`` are used instead.  Falls back to localhost for
        local development.
"""

from __future__ import annotations

import hashlib
import logging
import os
import pathlib
import textwrap

_GOOGLE_METADATA_URL: str = (
    "https://accounts.google.com/.well-known/openid-configuration"
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
_log = logging.getLogger(__name__)


def _build_redirect_uri() -> str:
    """Construct the OAuth callback URL for this deployment.

    Returns:
        The fully-qualified redirect URI that Google should send users back to
        after a successful login.

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
        client_secret: The Google OAuth client secret, used as entropy when
            no dedicated ``COOKIE_SECRET`` is configured.

    Returns:
        A hex string suitable for use as Streamlit's ``cookie_secret``.

    """
    if explicit := os.environ.get("COOKIE_SECRET"):
        return explicit
    return hashlib.sha256(client_secret.encode()).hexdigest()


def main() -> None:
    """Write ~/.streamlit/secrets.toml from environment variables.

    Exits without writing if ``GOOGLE_CLIENT_ID`` or
    ``GOOGLE_CLIENT_SECRET`` are absent, so local development without
    authentication is unaffected.

    """
    client_id = os.environ.get("GOOGLE_CLIENT_ID")
    client_secret = os.environ.get("GOOGLE_CLIENT_SECRET")

    if not client_id or not client_secret:
        _log.info(
            "GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET not set; "
            "skipping secrets.toml generation.",
        )
        return

    secrets_dir = pathlib.Path.home() / ".streamlit"
    secrets_dir.mkdir(parents=True, exist_ok=True)

    content = textwrap.dedent(f"""\
        [auth]
        redirect_uri = "{_build_redirect_uri()}"
        cookie_secret = "{_derive_cookie_secret(client_secret)}"

        [auth.providers.google]
        client_id = "{client_id}"
        client_secret = "{client_secret}"
        server_metadata_url = "{_GOOGLE_METADATA_URL}"
    """)

    secrets_path = secrets_dir / "secrets.toml"
    secrets_path.write_text(content)
    _log.info("Secrets written to %s", secrets_path)


if __name__ == "__main__":
    main()
