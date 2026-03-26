"""Tests for eval/run_experiment.py helper functions."""

# ruff: noqa: S101

from types import SimpleNamespace

from eval.run_experiment import _has_rooms_cases


class TestHasRoomsCases:
    """_has_rooms_cases() detects rooms traffic from tags or expected routes."""

    def test_rooms_route_without_rooms_tag_is_detected(self) -> None:
        """A rooms turn triggers DB preparation even when tags are absent."""
        examples = [
            SimpleNamespace(
                inputs={
                    "tags": [],
                    "turns": [{"user": "Book a room", "expected_route": "rooms"}],
                },
            ),
        ]

        result = _has_rooms_cases(examples)

        assert result is True

    def test_info_only_examples_return_false(self) -> None:
        """Examples with no rooms tag or rooms route do not trigger resets."""
        examples = [
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
        ]

        result = _has_rooms_cases(examples)

        assert result is False
