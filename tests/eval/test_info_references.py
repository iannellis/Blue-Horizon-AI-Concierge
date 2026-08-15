"""Tests for eval/evaluators/_info_references.py.

Covers pure helpers (_normalize_text, _split_expected_reference,
_build_reference_candidate_text, _reference_subset_match) and the
end-to-end eval_info_reference_subset() function via minimal mock objects.
"""

# ruff: noqa: S101

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

from eval.evaluators._info_references import (
    _build_reference_candidate_text,
    _normalize_text,
    _reference_subset_match,
    _split_expected_reference,
    eval_info_reference_subset,
)

if TYPE_CHECKING:
    from langsmith.schemas import Example, Run

    from eval.config import EvalConfig

# ---------------------------------------------------------------------------
# Helpers for constructing minimal mock objects
# ---------------------------------------------------------------------------


def _make_run(turn_outputs: list[dict[str, Any]]) -> Run:
    """Build a minimal run-like object.

    Args:
        turn_outputs: Per-turn output dicts to embed in outputs.

    Returns:
        Run: A `SimpleNamespace` with an `outputs` attribute, cast to `Run`
        since `eval_info_reference_subset` only ever reads that attribute
        off it.

    """
    return cast("Run", SimpleNamespace(outputs={"turn_outputs": turn_outputs}))


def _make_example(
    turns: list[dict[str, Any]],
    *,
    reference_answers: list[Any] | None = None,
) -> Example:
    """Build a minimal example-like object.

    Args:
        turns: Per-turn input dicts, each optionally containing a reference field.
        reference_answers: Optional top-level reference answer list.

    Returns:
        Example: A `SimpleNamespace` with an `inputs` attribute, cast to
        `Example` since `eval_info_reference_subset` only ever reads that
        attribute off it.

    """
    inputs: dict[str, Any] = {"turns": turns}
    if reference_answers is not None:
        inputs["reference_answers"] = reference_answers
    return cast("Example", SimpleNamespace(inputs=inputs))


def _make_cfg(
    *,
    json_value_max: int = 10_000,
    info_filter_failures_max: int = 50,
    reference_chars: int = 2_000,
) -> EvalConfig:
    """Build a minimal cfg-like object.

    Args:
        json_value_max: Maximum JSON string length.
        info_filter_failures_max: Maximum failure entries to keep.
        reference_chars: Maximum reference string length.

    Returns:
        EvalConfig: A `SimpleNamespace` with `evaluator_limits` and `ragas`
        attributes, cast to `EvalConfig` since `eval_info_reference_subset`
        only ever reads those attributes off it.

    """
    limits = SimpleNamespace(
        json_value_max=json_value_max,
        info_filter_failures_max=info_filter_failures_max,
    )
    ragas = SimpleNamespace(reference_chars=reference_chars)
    return cast("EvalConfig", SimpleNamespace(evaluator_limits=limits, ragas=ragas))


# ---------------------------------------------------------------------------
# _normalize_text
# ---------------------------------------------------------------------------


class TestNormalizeText:
    """_normalize_text lowercases and collapses whitespace."""

    def test_empty_string_returns_empty(self) -> None:
        """Empty input returns an empty string."""
        assert _normalize_text("") == ""

    def test_lowercases_text(self) -> None:
        """Mixed-case text is fully lowercased."""
        assert _normalize_text("Hello WORLD") == "hello world"

    def test_collapses_multiple_spaces(self) -> None:
        """Multiple consecutive spaces are collapsed to one."""
        assert _normalize_text("a   b   c") == "a b c"

    def test_strips_leading_trailing_whitespace(self) -> None:
        """Leading and trailing whitespace is removed."""
        assert _normalize_text("  hello  ") == "hello"

    def test_collapses_newlines(self) -> None:
        """Newlines and tabs are treated as whitespace and collapsed."""
        assert _normalize_text("line one\nline two\ttab") == "line one line two tab"

    def test_already_normalized_unchanged(self) -> None:
        """Text that is already normalised passes through unchanged."""
        assert _normalize_text("spa hours 9am") == "spa hours 9am"


# ---------------------------------------------------------------------------
# _split_expected_reference
# ---------------------------------------------------------------------------


class TestSplitExpectedReference:
    """_split_expected_reference splits on ' | ' and strips whitespace."""

    def test_empty_string_returns_empty_list(self) -> None:
        """An empty string produces no snippets."""
        assert _split_expected_reference("") == []

    def test_single_snippet(self) -> None:
        """A reference with no separator returns a single-element list."""
        assert _split_expected_reference("pool open 9am") == ["pool open 9am"]

    def test_two_snippets(self) -> None:
        """A ' | '-separated reference is split into two parts."""
        result = _split_expected_reference("pool open 9am | spa requires booking")
        assert result == ["pool open 9am", "spa requires booking"]

    def test_three_snippets(self) -> None:
        """Three-part references split correctly."""
        result = _split_expected_reference("a | b | c")
        assert result == ["a", "b", "c"]

    def test_whitespace_only_parts_filtered(self) -> None:
        """Parts that are entirely whitespace are dropped."""
        result = _split_expected_reference("  | valid snippet |   ")
        assert result == ["valid snippet"]

    def test_strips_surrounding_whitespace_from_parts(self) -> None:
        """Each part has leading/trailing whitespace stripped."""
        result = _split_expected_reference("  first  |  second  ")
        assert result == ["first", "second"]


# ---------------------------------------------------------------------------
# _build_reference_candidate_text
# ---------------------------------------------------------------------------


class TestBuildReferenceCandidateText:
    """_build_reference_candidate_text assembles context + assistant text."""

    def test_no_contexts_no_assistant_returns_empty(self) -> None:
        """A turn with neither contexts nor assistant text returns ''."""
        assert _build_reference_candidate_text({}) == ""

    def test_assistant_text_only(self) -> None:
        """Only assistant_text → that text is returned directly."""
        turn = {"assistant_text": "The pool closes at 10pm."}
        assert _build_reference_candidate_text(turn) == "The pool closes at 10pm."

    def test_contexts_only(self) -> None:
        """Only contexts_used → joined context string is returned."""
        turn = {"contexts_used": ["context A", "context B"]}
        result = _build_reference_candidate_text(turn)
        assert "context A" in result
        assert "context B" in result

    def test_both_contexts_and_assistant(self) -> None:
        """Both contexts and assistant_text are combined."""
        turn = {
            "contexts_used": ["relevant context"],
            "assistant_text": "Direct answer.",
        }
        result = _build_reference_candidate_text(turn)
        assert "relevant context" in result
        assert "Direct answer." in result

    def test_none_contexts_ignored(self) -> None:
        """None items in contexts_used are skipped."""
        turn = {"contexts_used": [None, "good context", None]}
        result = _build_reference_candidate_text(turn)
        assert "good context" in result

    def test_non_list_contexts_ignored(self) -> None:
        """contexts_used that is not a list is treated as absent."""
        turn = {"contexts_used": "not a list", "assistant_text": "answer"}
        assert _build_reference_candidate_text(turn) == "answer"


# ---------------------------------------------------------------------------
# _reference_subset_match
# ---------------------------------------------------------------------------


class TestReferenceSubsetMatch:
    """_reference_subset_match checks whether all snippets appear in candidate."""

    def test_all_snippets_present_returns_true(self) -> None:
        """When every snippet is found, matched=True and missing is empty."""
        matched, missing = _reference_subset_match(
            ["pool open 9am", "spa requires booking"],
            "The pool open 9am daily. The spa requires booking in advance.",
        )
        assert matched is True
        assert missing == []

    def test_missing_snippet_returns_false(self) -> None:
        """A snippet absent from the candidate produces matched=False."""
        matched, missing = _reference_subset_match(
            ["pool open 9am", "gym free weights"],
            "The pool open 9am daily.",
        )
        assert matched is False
        assert "gym free weights" in missing

    def test_case_insensitive_match(self) -> None:
        """Matching is case-insensitive via normalisation."""
        matched, missing = _reference_subset_match(
            ["Pool Open 9AM"],
            "the pool open 9am every day",
        )
        assert matched is True
        assert missing == []

    def test_whitespace_insensitive_match(self) -> None:
        """Extra whitespace in snippet or candidate is normalised away."""
        matched, missing = _reference_subset_match(
            ["pool  open  9am"],
            "the  pool open  9am daily",
        )
        assert matched is True
        assert missing == []

    def test_empty_snippets_list_returns_true(self) -> None:
        """With no snippets to check, the result is trivially True."""
        matched, missing = _reference_subset_match([], "some text")
        assert matched is True
        assert missing == []

    def test_multiple_missing_snippets_all_reported(self) -> None:
        """All missing snippets are listed, not just the first."""
        matched, missing = _reference_subset_match(
            ["a", "b", "c"],
            "only a is here",
        )
        assert matched is False
        assert "b" in missing
        assert "c" in missing


# ---------------------------------------------------------------------------
# eval_info_reference_subset (end-to-end with mocks)
# ---------------------------------------------------------------------------


class TestEvalInfoReferenceSubset:
    """eval_info_reference_subset() returns correct LangSmith feedback dicts."""

    def test_no_reference_turns_returns_skipped(self) -> None:
        """Turns with no reference field produce the 'skipped' result."""
        run = _make_run([{"contexts_used": ["some context"], "assistant_text": "ok"}])
        example = _make_example([{"user": "What is the wifi?"}])
        cfg = _make_cfg()
        results = eval_info_reference_subset(run, example, cfg=cfg)
        keys = [r["key"] for r in results]
        assert "info_reference_subset_skipped" in keys

    def test_all_snippets_found_pass_rate_one(self) -> None:
        """All snippets present → pass_rate=1.0."""
        snippet = "pool open 9am"
        run = _make_run([{
            "contexts_used": [f"The {snippet} daily."],
            "assistant_text": "yes",
        }])
        example = _make_example([{"user": "pool hours?", "reference": snippet}])
        cfg = _make_cfg()
        results = eval_info_reference_subset(run, example, cfg=cfg)
        scores = {r["key"]: r.get("score") for r in results}
        assert scores.get("info_reference_subset_pass_rate") == 1.0

    def test_snippet_missing_pass_rate_zero(self) -> None:
        """A snippet not found in contexts or response → pass_rate=0.0."""
        run = _make_run([{
            "contexts_used": ["unrelated context"],
            "assistant_text": "I don't know.",
        }])
        example = _make_example([{
            "user": "gym hours?",
            "reference": "gym open 6am to midnight",
        }])
        cfg = _make_cfg()
        results = eval_info_reference_subset(run, example, cfg=cfg)
        scores = {r["key"]: r.get("score") for r in results}
        assert scores.get("info_reference_subset_pass_rate") == 0.0

    def test_partial_pass_rate(self) -> None:
        """One passing and one failing turn produces a 0.5 pass rate."""
        snippet_found = "pool open 9am"
        snippet_missing = "gym open 6am"
        run = _make_run([
            {"contexts_used": [f"The {snippet_found}."], "assistant_text": "yes"},
            {"contexts_used": ["unrelated"], "assistant_text": "no"},
        ])
        example = _make_example([
            {"user": "pool?", "reference": snippet_found},
            {"user": "gym?", "reference": snippet_missing},
        ])
        cfg = _make_cfg()
        results = eval_info_reference_subset(run, example, cfg=cfg)
        scores = {r["key"]: r.get("score") for r in results}
        assert scores.get("info_reference_subset_pass_rate") == 0.5  # noqa: PLR2004

    def test_reference_from_reference_answers_list(self) -> None:
        """reference_answers top-level list is used when turn has no reference."""
        snippet = "breakfast 7am"
        run = _make_run([{
            "contexts_used": [f"Breakfast served from {snippet}."],
            "assistant_text": "yes",
        }])
        example = _make_example(
            [{"user": "breakfast?"}],
            reference_answers=[snippet],
        )
        cfg = _make_cfg()
        results = eval_info_reference_subset(run, example, cfg=cfg)
        scores = {r["key"]: r.get("score") for r in results}
        assert scores.get("info_reference_subset_pass_rate") == 1.0

    def test_pipe_separated_reference_all_must_match(self) -> None:
        """All ' | '-separated snippets must be present to pass."""
        run = _make_run([{
            "contexts_used": ["pool open 9am daily"],
            "assistant_text": "yes",
        }])
        example = _make_example([{
            "user": "pool and spa?",
            "reference": "pool open 9am | spa requires booking",
        }])
        cfg = _make_cfg()
        results = eval_info_reference_subset(run, example, cfg=cfg)
        scores = {r["key"]: r.get("score") for r in results}
        assert scores.get("info_reference_subset_pass_rate") == 0.0

    def test_scored_turns_count(self) -> None:
        """info_reference_subset_turns reflects the number of scored turns."""
        snippet = "wifi free"
        run = _make_run([
            {"contexts_used": [snippet], "assistant_text": "yes"},
            {"contexts_used": [snippet], "assistant_text": "yes"},
        ])
        example = _make_example([
            {"user": "wifi?", "reference": snippet},
            {"user": "wifi again?", "reference": snippet},
        ])
        cfg = _make_cfg()
        results = eval_info_reference_subset(run, example, cfg=cfg)
        scores = {r["key"]: r.get("score") for r in results}
        assert scores.get("info_reference_subset_turns") == 2.0  # noqa: PLR2004
