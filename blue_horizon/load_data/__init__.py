"""Namespace package for loading data into databases."""

from pathlib import Path


def _repo_root() -> Path:
    """Return the repository root directory.

    Returns:
        Absolute path two levels above this package.

    """
    return Path(__file__).parents[2]
