"""Tests for pure helper functions inside the information agent factory.

_best_by_text, _build_context_block, and _latest_user_text are module-level
helpers with no I/O or LLM calls.
"""

# ruff: noqa: S101

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from blue_horizon.agents.information.factory import (
    _best_by_text,
    _build_context_block,
    _latest_user_text,
)
from blue_horizon.agents.information.models import RetrievalItem, Source

# ---------------------------------------------------------------------------
# Shared fixtures / factories
# ---------------------------------------------------------------------------


def _item(text: str, score: float, source: Source = Source.FAQ) -> RetrievalItem:
    """Construct a minimal RetrievalItem for testing.

    Args:
        text: Item text content.
        score: Similarity score.
        source: Knowledge source (defaults to FAQ).

    Returns:
        A RetrievalItem with empty metadata.

    """
    return RetrievalItem(source=source, text=text, score=score, metadata={})


# ---------------------------------------------------------------------------
# _best_by_text
# ---------------------------------------------------------------------------


class TestBestByText:
    """_best_by_text deduplicates retrieval results, keeping the highest score."""

    def test_empty_batches_returns_empty(self) -> None:
        """All-empty batches produce an empty output list."""
        assert _best_by_text([[], [], []]) == []

    def test_single_batch_no_duplicates_passes_through(self) -> None:
        """Unique texts in a single batch are all kept."""
        items = [_item("a", 0.9), _item("b", 0.8)]
        result = _best_by_text([items])
        assert len(result) == 2  # noqa: PLR2004
        texts = {r.text for r in result}
        assert texts == {"a", "b"}

    def test_duplicate_within_same_batch_keeps_first_seen(self) -> None:
        """When the same text appears twice in one batch, the first (higher) wins."""
        items = [_item("dup", 0.9), _item("dup", 0.5)]
        result = _best_by_text([items])
        assert len(result) == 1
        assert result[0].score == 0.9  # noqa: PLR2004

    def test_duplicate_across_batches_keeps_higher_score(self) -> None:
        """The item with the higher score wins across batches."""
        batch1 = [_item("shared", 0.6)]
        batch2 = [_item("shared", 0.85)]
        result = _best_by_text([batch1, batch2])
        assert len(result) == 1
        assert result[0].score == 0.85  # noqa: PLR2004

    def test_lower_score_does_not_overwrite_higher(self) -> None:
        """Processing order does not let a lower-scored duplicate overwrite."""
        batch1 = [_item("x", 0.95)]
        batch2 = [_item("x", 0.2)]
        result = _best_by_text([batch1, batch2])
        assert result[0].score == 0.95  # noqa: PLR2004

    def test_multiple_batches_no_overlap_all_returned(self) -> None:
        """Non-overlapping batches are merged without loss."""
        batches = [
            [_item("a", 0.9), _item("b", 0.8)],
            [_item("c", 0.7)],
            [_item("d", 0.6), _item("e", 0.5)],
        ]
        result = _best_by_text(batches)
        assert len(result) == 5  # noqa: PLR2004
        assert {r.text for r in result} == {"a", "b", "c", "d", "e"}

    def test_result_preserves_metadata(self) -> None:
        """The winning item's metadata is kept unchanged."""
        item = RetrievalItem(
            source=Source.AMENITIES,
            text="pool",
            score=0.99,
            metadata={"price": 0.0},
        )
        result = _best_by_text([[item]])
        assert result[0].metadata == {"price": 0.0}

    def test_no_batches_returns_empty(self) -> None:
        """Zero batches produce an empty list."""
        assert _best_by_text([]) == []


# ---------------------------------------------------------------------------
# _build_context_block
# ---------------------------------------------------------------------------


class TestBuildContextBlock:
    """_build_context_block serialises RetrievalItems into a prompt context."""

    def test_empty_list_returns_empty_string(self) -> None:
        """No items produces an empty string."""
        assert _build_context_block([]) == ""

    def test_single_item_contains_required_fields(self) -> None:
        """Each item block contains source, score, text, and metadata."""
        item = _item("spa treatment", 0.88)
        block = _build_context_block([item])
        assert "source=" in block
        assert "score=" in block
        assert "text=spa treatment" in block
        assert "metadata=" in block

    def test_multiple_items_separated_by_double_newline(self) -> None:
        """Multiple items are joined by double newlines."""
        items = [_item("a", 0.9), _item("b", 0.7)]
        block = _build_context_block(items)
        # Two items → one "\n\n" separator
        assert "\n\n" in block
        parts = block.split("\n\n")
        assert len(parts) == 2  # noqa: PLR2004

    def test_score_value_appears_in_output(self) -> None:
        """The exact score value is rendered in the block."""
        item = _item("gym", 0.42)
        block = _build_context_block([item])
        assert "0.42" in block

    def test_source_value_appears_in_output(self) -> None:
        """The source enum value string appears in the block."""
        item = _item("pool", 0.5, source=Source.AMENITIES)
        block = _build_context_block([item])
        assert "amenities" in block


# ---------------------------------------------------------------------------
# _latest_user_text
# ---------------------------------------------------------------------------


class TestLatestUserText:
    """_latest_user_text finds the most recent HumanMessage content."""

    def test_no_messages_returns_empty_string(self) -> None:
        """Empty message list returns an empty string."""
        assert _latest_user_text([]) == ""

    def test_no_human_message_returns_empty_string(self) -> None:
        """A list containing only AI/System messages returns an empty string."""
        messages = [AIMessage(content="Hello"), SystemMessage(content="System prompt")]
        assert _latest_user_text(messages) == ""

    def test_single_human_message_returned(self) -> None:
        """A single HumanMessage's content is returned."""
        messages = [HumanMessage(content="What is the wifi password?")]
        assert _latest_user_text(messages) == "What is the wifi password?"

    def test_returns_most_recent_human_message(self) -> None:
        """The last HumanMessage in the list is returned, not the first."""
        messages = [
            HumanMessage(content="First question"),
            AIMessage(content="First answer"),
            HumanMessage(content="Second question"),
        ]
        assert _latest_user_text(messages) == "Second question"

    def test_ignores_ai_messages_after_last_human(self) -> None:
        """AI messages appearing after the last human turn do not interfere."""
        messages = [
            HumanMessage(content="Are there pool towels?"),
            AIMessage(content="Yes, we have towels."),
        ]
        assert _latest_user_text(messages) == "Are there pool towels?"
