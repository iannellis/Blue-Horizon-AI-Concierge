"""Utility helpers for loading prompt templates from package resources."""

import importlib.resources as importlib_resources
from functools import lru_cache
from string import Template
from typing import Final

from blue_horizon.agents.exceptions import ConfigurationError

DEFAULT_PACKAGE: Final[str] = "blue_horizon"


def prompt_resource_path(folder: str, filename: str) -> str:
    """Join a prompts folder and filename into a packaged-resource path.

    Every agent resource class resolves its prompt path this same way, and
    `eval/run_experiment.py` mirrors it independently to fingerprint the
    prompt text the agent actually reads -- both must join identically or a
    fingerprint could silently point at the wrong file.

    Args:
        folder: Prompts folder from config, possibly empty or slash-wrapped.
        filename: Prompt file name within that folder.

    Returns:
        str: ``f"{folder}/{filename}"`` with a stripped folder, or bare
        `filename` if the folder is empty.

    """
    folder = folder.strip("/")
    return f"{folder}/{filename}" if folder else filename


def load_prompt_template(
    relative_path: str,
    *,
    base_package: str = DEFAULT_PACKAGE,
) -> Template:
    """Load a prompt template from a packaged resource.

    Args:
        relative_path: Path to the template relative to the package.
        base_package: Package containing the template.

    Returns:
        Parsed string.Template for the prompt.

    Raises:
        ConfigurationError: If the template content cannot be loaded. A
            missing or unreadable packaged prompt file is a deployment
            defect, not something a retry can fix.

    """
    text = load_packaged_text(relative_path, base_package=base_package)
    return Template(text)


@lru_cache(maxsize=10)
def load_packaged_text(
    relative_path: str,
    *,
    base_package: str = DEFAULT_PACKAGE,
) -> str:
    """Read and cache UTF-8 text from a packaged resource.

    Args:
        relative_path: Path to the resource file relative to the package.
        base_package: Package containing the resource.

    Returns:
        The decoded text content of the resource.

    Raises:
        ConfigurationError: If the resource cannot be found or read. This is
            a packaging defect (a prompt file missing from the shipped
            image), not a dependency outage, so it is deliberately not an
            `OperationalError`: retrying will not make the file appear.

    """
    try:
        traversable = importlib_resources.files(base_package).joinpath(relative_path)
        return traversable.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        msg = f"Resource not found: {base_package}/{relative_path}"
        raise ConfigurationError(msg) from exc
    except OSError as exc:
        msg = f"Failed to read resource: {base_package}/{relative_path}"
        raise ConfigurationError(msg) from exc
