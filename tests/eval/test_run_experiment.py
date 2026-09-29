"""Tests for eval/run_experiment.py helper functions."""

# ruff: noqa: S101

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

from eval.run_experiment import _has_booking_cases, _score_example_locally

if TYPE_CHECKING:
    from langsmith.schemas import Example, Run


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


class TestScoreExampleLocally:
    """_score_example_locally() keeps scoring when one evaluator raises."""

    def test_failed_evaluator_is_skipped_and_others_still_score(self) -> None:
        """A raising evaluator adds no feedback; the rest are still collected."""

        def _failing(_run: object, _example: object) -> list[dict[str, object]]:
            msg = "judge returned text instead of a function call"
            raise RuntimeError(msg)

        async def _passing(_run: object, _example: object) -> list[dict[str, object]]:
            return [{"key": "route_accuracy", "score": 1.0}]

        run = cast("Run", SimpleNamespace(outputs={}))
        example = cast("Example", SimpleNamespace(id="example-1", inputs={}))

        feedback = asyncio.run(
            _score_example_locally(run, example, [_failing, _passing]),
        )

        assert feedback == [{"key": "route_accuracy", "score": 1.0}]
