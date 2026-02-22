"""Configuration loading and prompt rendering for the rooms SQL agent."""

from __future__ import annotations

from typing import TYPE_CHECKING

from blue_horizon.agents.prompt_utils import load_prompt_template
from blue_horizon.config import RoomsSqlConfig, load_app_config

if TYPE_CHECKING:
    from pathlib import Path
    from string import Template


def load_rooms_config(config_path: Path | str | None = None) -> RoomsSqlConfig:
    """Load the rooms SQL configuration section.

    For when using the rooms agent standalone.

    Args:
        config_path: Optional path to override the packaged config. If unset,
            ``app_config.toml`` from the package resources is used.

    Returns:
        RoomsSqlConfig: Parsed configuration for the rooms agent.

    """
    app_config = load_app_config(path=config_path)
    return app_config.rooms


def render_system_prompt(  # noqa: PLR0913
    *,
    template: Template,
    top_k: int,
    enum_values: dict[str, list[str]],
    basic_amenities: list[str],
    additional_amenities: list[str],
    view_types: list[str],
) -> str:
    """Render the system prompt template with runtime substitutions.

    Args:
        template: String Template loaded from the rooms prompt resource.
        top_k: Maximum number of rooms to surface in the prompt.
        enum_values: Mapping of enum type name to list of valid values.
        basic_amenities: Distinct basic amenity values from the database.
        additional_amenities: Distinct additional amenity values from the database.
        view_types: Distinct view type values from the database.

    Returns:
        Rendered system prompt string with all placeholders substituted.

    """
    return template.safe_substitute(
        top_k=top_k,
        room_type=enum_values.get("room_type", []),
        basic_amenities=basic_amenities,
        additional_amenities=additional_amenities,
        room_bed_type=enum_values.get("room_bed_type", []),
        view_types=view_types,
        room_status_type=enum_values.get("room_status_type", []),
        availability_status_type=enum_values.get("availability_status_type", []),
    )


def load_prompt_template_for_rooms(resource: str) -> Template:
    """Load the rooms system prompt template from package resources.

    Args:
        resource: Package resource path to the prompt template file.

    Returns:
        Loaded string Template.

    Raises:
        FileNotFoundError: If the resource cannot be located.
        RuntimeError: If the template cannot be loaded.

    """
    return load_prompt_template(resource)
