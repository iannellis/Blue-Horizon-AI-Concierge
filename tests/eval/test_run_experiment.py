"""Tests for eval/run_experiment.py helper functions."""

# ruff: noqa: S101

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

from eval.run_experiment import _has_booking_cases

if TYPE_CHECKING:
    from langsmith.schemas import Example


class TestHasBookingCases:
    """_has_booking_cases() detects booking traffic from tags or expected routes."""

    def test_booking_route_without_booking_tag_is_detected(self) -> None:
        """A booking turn triggers DB preparation even when tags are absent."""
        examples = cast(
            "list[Example]",
            [
                SimpleNamespace(
                    inputs={
                        "tags": [],
                        "turns": [
                            {"user": "Book a room", "expected_route": "booking"},
                        ],
                    },
                ),
            ],
        )

        result = _has_booking_cases(examples)

        assert result is True

    def test_info_only_examples_return_false(self) -> None:
        """Examples with no booking tag or booking route do not trigger resets."""
        examples = cast(
            "list[Example]",
            [
                SimpleNamespace(
                    inputs={
                        "tags": ["info"],
                        "turns": [
                            {
                                "user": "What time is breakfast?",
                                "expected_route": "info",
                            },
                        ],
                    },
                ),
            ],
        )

        result = _has_booking_cases(examples)

        assert result is False
