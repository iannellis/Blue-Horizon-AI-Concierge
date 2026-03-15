"""Tests for orchestration response formatting helpers.

format_chat_response and _extract_text_content are pure functions.
LangChain message objects are instantiated directly; no LLM calls are made.
"""

# ruff: noqa: S101

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from blue_horizon.agents.orchestration.formatting import (
    _extract_text_content,
    format_chat_response,
)

# ---------------------------------------------------------------------------
# _extract_text_content
# ---------------------------------------------------------------------------


class TestExtractTextContent:
    """_extract_text_content normalises message content to a plain string."""

    def test_plain_string_returned_as_is(self) -> None:
        """A bare string passes through unchanged."""
        assert _extract_text_content("Hello world") == "Hello world"

    def test_empty_string_returned_as_is(self) -> None:
        """An empty string is returned as an empty string."""
        assert _extract_text_content("") == ""

    def test_list_single_text_block(self) -> None:
        """A list containing one text block returns the text."""
        content = [{"type": "text", "text": "Hi there"}]
        assert _extract_text_content(content) == "Hi there"

    def test_list_reasoning_block_excluded(self) -> None:
        """Non-text block types (reasoning) are dropped."""
        content = [
            {"type": "thinking", "thinking": "Let me reason…"},
            {"type": "text", "text": "The answer is 42."},
        ]
        assert _extract_text_content(content) == "The answer is 42."

    def test_list_multiple_text_blocks_joined_with_space(self) -> None:
        """Multiple text blocks are joined by a single space."""
        content = [
            {"type": "text", "text": "Part one."},
            {"type": "text", "text": "Part two."},
        ]
        assert _extract_text_content(content) == "Part one. Part two."

    def test_list_no_text_blocks_returns_empty_string(self) -> None:
        """A list with no text blocks yields an empty string."""
        content = [{"type": "thinking", "thinking": "silent reasoning"}]
        assert _extract_text_content(content) == ""

    def test_empty_list_returns_empty_string(self) -> None:
        """An empty list yields an empty string."""
        assert _extract_text_content([]) == ""

    def test_list_items_without_type_key_skipped(self) -> None:
        """Dict items missing the 'type' key are skipped."""
        content = [{"text": "orphaned"}, {"type": "text", "text": "kept"}]
        assert _extract_text_content(content) == "kept"


# ---------------------------------------------------------------------------
# format_chat_response
# ---------------------------------------------------------------------------


class TestFormatChatResponse:
    """format_chat_response strips internal messages and normalises content."""

    def test_empty_messages_returns_empty_list(self) -> None:
        """Result with no messages produces an empty messages list."""
        result = format_chat_response({"messages": []})
        assert result == {"messages": []}

    def test_missing_messages_key_returns_empty_list(self) -> None:
        """A result dict without a 'messages' key is handled gracefully."""
        result = format_chat_response({})
        assert result == {"messages": []}

    def test_tool_messages_filtered_out(self) -> None:
        """ToolMessage objects do not appear in the formatted output."""
        messages = [
            HumanMessage(content="book a room"),
            ToolMessage(content="[tool output]", tool_call_id="tc1"),
            AIMessage(content="Done!"),
        ]
        result = format_chat_response({"messages": messages})
        types = [m["type"] for m in result["messages"]]
        assert "tool" not in types
        assert len(result["messages"]) == 2  # noqa: PLR2004

    def test_human_and_ai_messages_kept(self) -> None:
        """HumanMessage and AIMessage are both retained."""
        messages = [HumanMessage(content="hi"), AIMessage(content="hello")]
        result = format_chat_response({"messages": messages})
        types = [m["type"] for m in result["messages"]]
        assert types == ["human", "ai"]

    def test_human_content_extracted(self) -> None:
        """Human message content is correctly extracted."""
        messages = [HumanMessage(content="Is the pool open?")]
        result = format_chat_response({"messages": messages})
        assert result["messages"][0]["content"] == "Is the pool open?"

    def test_ai_string_content_extracted(self) -> None:
        """AI message with a plain string content is extracted directly."""
        messages = [AIMessage(content="Yes, the pool is open.")]
        result = format_chat_response({"messages": messages})
        assert result["messages"][0]["content"] == "Yes, the pool is open."

    def test_ai_list_content_text_extracted(self) -> None:
        """AI message with list content has text blocks extracted."""
        messages = [
            AIMessage(content=[{"type": "text", "text": "Rooms available."}]),
        ]
        result = format_chat_response({"messages": messages})
        assert result["messages"][0]["content"] == "Rooms available."

    def test_ai_reasoning_block_stripped(self) -> None:
        """Reasoning blocks in AI list content are stripped from output."""
        messages = [
            AIMessage(
                content=[
                    {"type": "thinking", "thinking": "private reasoning"},
                    {"type": "text", "text": "Public answer."},
                ],
            ),
        ]
        result = format_chat_response({"messages": messages})
        assert result["messages"][0]["content"] == "Public answer."

    def test_multi_turn_conversation(self) -> None:
        """Multiple turns are all kept in order."""
        messages = [
            HumanMessage(content="Q1"),
            AIMessage(content="A1"),
            HumanMessage(content="Q2"),
            AIMessage(content="A2"),
        ]
        result = format_chat_response({"messages": messages})
        assert len(result["messages"]) == 4  # noqa: PLR2004
        assert [m["content"] for m in result["messages"]] == ["Q1", "A1", "Q2", "A2"]

    def test_output_schema_has_type_and_content_keys(self) -> None:
        """Every formatted message dict has exactly 'type' and 'content' keys."""
        messages = [HumanMessage(content="test"), AIMessage(content="reply")]
        result = format_chat_response({"messages": messages})
        for msg in result["messages"]:
            assert set(msg.keys()) == {"type", "content"}
