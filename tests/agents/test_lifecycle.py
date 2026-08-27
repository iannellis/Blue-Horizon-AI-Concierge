"""Tests for blue_horizon/agents/_lifecycle.py."""

# ruff: noqa: S101

import pytest

from blue_horizon.agents._lifecycle import require


class TestRequire:
    """require() narrows an Optional or raises with a useful message."""

    def test_passes_non_none_value_through(self) -> None:
        """A non-None value is returned unchanged."""
        assert require("a rendered prompt", "System prompt") == "a rendered prompt"

    def test_passes_through_falsy_but_non_none_value(self) -> None:
        """A falsy-but-real value (0, "", []) is not mistaken for uninitialized."""
        assert require(0, "Count") == 0

    def test_raises_runtime_error_on_none(self) -> None:
        """None raises RuntimeError naming the resource and how to fix it."""
        with pytest.raises(RuntimeError, match="System prompt"):
            require(None, "System prompt")

    def test_error_message_names_startup_check(self) -> None:
        """The error message tells the caller how to initialize the resource."""
        with pytest.raises(RuntimeError, match="await startup_check"):
            require(None, "Read pool")
