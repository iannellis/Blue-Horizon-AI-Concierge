"""Configuration loading and prompt rendering for the booking agent."""

from __future__ import annotations

from typing import TYPE_CHECKING

from blue_horizon.config import BookingSqlConfig, load_app_config

if TYPE_CHECKING:
    from pathlib import Path
    from string import Template

    from blue_horizon.agents.booking.search import RoomsMetadata


def load_booking_config(config_path: Path | str | None = None) -> BookingSqlConfig:
    """Load the booking SQL configuration section.

    For when using the booking agent standalone.

    Args:
        config_path: Optional path to override the packaged config. If unset,
            ``app_config.toml`` from the package resources is used.

    Returns:
        BookingSqlConfig: Parsed configuration for the booking agent.

    """
    app_config = load_app_config(path=config_path)
    return app_config.booking


def render_system_prompt(
    *,
    template: Template,
    top_k: int,
    meta: RoomsMetadata,
) -> str:
    """Render the system prompt template with runtime substitutions.

    The searchable values themselves are not rendered: the `search_rooms`
    tool schema lists them, so the prompt needs only the bounds a guest's
    wording has to be mapped onto.

    Args:
        template: String Template loaded from the booking prompt resource.
        top_k: Maximum number of rooms a search returns.
        meta: Rooms metadata supplying the date window and top floor.

    Returns:
        Rendered system prompt string with all placeholders substituted.

    """
    return template.safe_substitute(
        top_k=top_k,
        first_night=meta.first_night.isoformat(),
        last_night=meta.last_night.isoformat(),
        last_check_out=meta.last_check_out.isoformat(),
        max_floor=meta.max_floor,
    )
