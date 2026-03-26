"""Text and payload extraction utilities for the eval harness.

Provides helpers for previewing, sanitizing, and extracting content from
LangChain messages, orchestration results, and tool outputs.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.messages import AIMessage


def _preview(obj: object, max_len: int = 200) -> str:
    """Create a short, redacted preview of a value.

    Args:
        obj: Value to preview.
        max_len: Maximum preview length.

    Returns:
        Sanitized preview string.

    """
    if max_len <= 0:
        return ""
    text = _safe_str(obj)
    text = _redact_secrets(text)
    text = " ".join(text.split())
    if len(text) <= max_len:
        return text
    return f"{text[: max_len - 1]}…"


def _safe_str(obj: object) -> str:
    """Safely convert an object to a string.

    Args:
        obj: Object to convert.

    Returns:
        String representation of the object.

    """
    if isinstance(obj, str):
        return obj
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return repr(obj)


def _redact_secrets(text: str) -> str:
    """Redact obvious secret-like substrings from a text blob.

    Args:
        text: Input text to sanitize.

    Returns:
        Redacted text safe for logging.

    """
    redacted = re.sub(r"sk-[A-Za-z0-9]{10,}", "[REDACTED]", text)
    redacted = re.sub(r"Bearer\s+\S+", "Bearer [REDACTED]", redacted)
    redacted = re.sub(
        r"(\w+://)([^:@\s]+):([^@\s]+)@",
        r"\1[REDACTED]:[REDACTED]@",
        redacted,
    )
    secret_keys = (
        "OPENAI_API_KEY",
        "GOOGLE_API_KEY",
        "LANGSMITH_API_KEY",
    )
    for key in secret_keys:
        if key in redacted:
            redacted = redacted.replace(key, "[REDACTED]")
    return redacted


def _input_keys(obj: object, max_keys: int = 20) -> list[str]:
    """Extract a capped list of input keys from a mapping.

    Args:
        obj: Tool input payload.
        max_keys: Maximum number of keys to return.

    Returns:
        Sorted list of keys when available, otherwise an empty list.

    """
    if not isinstance(obj, Mapping):
        return []
    keys = sorted(str(k) for k in obj)
    return keys[:max_keys]


def _extract_assistant_text(messages: Sequence[object]) -> str:
    """Extract the last assistant message content from a list of messages.

    Args:
        messages: Ordered list of LangChain messages.

    Returns:
        The last assistant message content, or an empty string if none found.

    """
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return _stringify_message_content(message.content)
        role = getattr(message, "type", None) or getattr(message, "role", None)
        if role in {"ai", "assistant"}:
            return _stringify_message_content(getattr(message, "content", ""))
    return ""


def _get_tool_name(metadata: dict[str, Any]) -> str | None:
    """Extract the tool name from callback metadata.

    Args:
        metadata: Callback metadata dict.

    Returns:
        Tool name string if available.

    """
    name = metadata.get("name") or metadata.get("tool_name")
    if isinstance(name, str):
        return name
    serialized = metadata.get("serialized")
    if isinstance(serialized, dict):
        serialized_name = serialized.get("name")
        if isinstance(serialized_name, str):
            return serialized_name
    return None


def _extract_assistant_text_from_result(result: object) -> str:
    """Extract the final assistant text from an orchestration result object.

    Args:
        result: Orchestration result object, typically a mapping containing a
            ``"messages"`` entry.

    Returns:
        Final assistant message text, or an empty string when unavailable.

    """
    if not isinstance(result, Mapping):
        return ""

    messages = result.get("messages", [])
    if not isinstance(messages, Sequence):
        return ""

    return _extract_assistant_text(messages)


def _stringify_message_content(content: object) -> str:
    """Normalize assistant message content into plain text.

    Args:
        content: Message content, which may be a string or a list of blocks.

    Returns:
        Text content with non-text blocks omitted.

    """
    if isinstance(content, str):
        return content.strip()

    if not isinstance(content, list):
        return str(content).strip()

    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
            continue
        if isinstance(item, dict):
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)

    return " ".join(parts).strip()
