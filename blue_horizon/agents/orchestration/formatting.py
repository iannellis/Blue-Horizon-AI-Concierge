"""API response formatting utilities for the orchestration agent."""

from __future__ import annotations

from typing import Any


def format_chat_response(result: dict[str, Any]) -> dict[str, Any]:
    """Extract and format the human/AI message history from an orchestrator result.

    Strips internal LangChain message metadata, tool messages, and reasoning blocks,
    returning only the fields needed by the chat API client.

    Args:
        result: Raw state dict returned by ``OrchestrationManager.ainvoke``.

    Returns:
        Dict with a ``messages`` key containing only human and AI messages, each
        normalised to ``{"type": str, "content": str}``.

    """
    messages = result.get("messages", [])
    formatted = []
    for msg in messages:
        msg_type = getattr(msg, "type", None)
        if msg_type not in {"human", "ai"}:
            continue
        formatted.append(
            {
                "type": msg_type,
                "content": _extract_text_content(getattr(msg, "content", "")),
            },
        )
    return {"messages": formatted}


def _extract_text_content(content: str | list[Any]) -> str:
    """Normalise a LangChain message content value to a plain string.

    When a model is configured with a reasoning effort parameter, the content
    field is a list that may contain reasoning blocks alongside text blocks.
    This function discards non-text blocks and joins the remaining text.

    Args:
        content: Either a plain string or a list of content block dicts as
            returned by LangChain/OpenAI messages.

    Returns:
        Plain text string with text blocks joined by a single space.

    """
    if isinstance(content, str):
        return content
    return " ".join(
        item["text"]
        for item in content
        if isinstance(item, dict) and item.get("type") == "text"
    )
