"""Configuration loading and prompt rendering for the information RAG agent."""

from __future__ import annotations

from typing import TYPE_CHECKING

from blue_horizon.agents.prompt_utils import load_prompt_template
from blue_horizon.config import InfoRagConfig, load_app_config

if TYPE_CHECKING:
    from pathlib import Path


def load_info_config(config_path: Path | str | None = None) -> InfoRagConfig:
    """Load the info configuration section.

    For when using the information agent standalone.

    Args:
        config_path: Optional path to override the packaged config. If unset,
            ``app_config.toml`` from the package resources is used.

    Returns:
        InfoRagConfig: Parsed configuration for the information agent.

    """
    app_config = load_app_config(path=config_path)
    return app_config.info


def build_system_prompt(*, top_k: int, prompt_resource: str) -> str:
    """Render the system prompt with runtime substitutions.

    Args:
        top_k: Maximum number of retrieval items referenced by the prompt.
        prompt_resource: Package resource path to the prompt template file.

    Returns:
        Rendered system prompt with runtime placeholders filled in.

    Raises:
        FileNotFoundError: If the prompt template resource cannot be located.
        RuntimeError: If the template cannot be loaded or substituted.

    """
    template = load_prompt_template(prompt_resource)
    return template.safe_substitute(top_k=top_k)
