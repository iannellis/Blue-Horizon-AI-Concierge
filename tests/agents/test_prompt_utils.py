"""Tests for blue_horizon/agents/prompt_utils.py."""

# ruff: noqa: S101

from blue_horizon.agents.prompt_utils import prompt_resource_path


class TestPromptResourcePath:
    """prompt_resource_path() joins a prompts folder and filename."""

    def test_joins_folder_and_filename(self) -> None:
        """A plain folder is joined with the filename by a single slash."""
        assert (
            prompt_resource_path("system_prompts", "rooms_sql_prompt.txt")
            == "system_prompts/rooms_sql_prompt.txt"
        )

    def test_empty_folder_returns_bare_filename(self) -> None:
        """An empty folder is dropped, leaving just the filename."""
        assert (
            prompt_resource_path("", "rooms_sql_prompt.txt") == "rooms_sql_prompt.txt"
        )

    def test_strips_leading_and_trailing_slashes(self) -> None:
        """A folder wrapped in slashes is stripped before joining."""
        assert (
            prompt_resource_path("/system_prompts/", "rooms_sql_prompt.txt")
            == "system_prompts/rooms_sql_prompt.txt"
        )

    def test_folder_of_only_slashes_returns_bare_filename(self) -> None:
        """A folder that strips down to empty behaves like an empty folder."""
        assert (
            prompt_resource_path("//", "rooms_sql_prompt.txt") == "rooms_sql_prompt.txt"
        )
