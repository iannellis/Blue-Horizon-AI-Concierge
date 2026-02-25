"""Text and payload extraction utilities for the eval harness.

Provides helpers for previewing, sanitizing, and extracting content from
LangChain messages and tool outputs.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage


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
    redacted = re.sub(r"Bearer\\s+\\S+", "Bearer [REDACTED]", redacted)
    redacted = re.sub(
        r"(\\w+://)([^:@\\s]+):([^@\\s]+)@",
        r"\\1[REDACTED]:[REDACTED]@",
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


def _extract_assistant_text(messages: list[BaseMessage]) -> str:
    """Extract the last assistant message content from a list of messages.

    Args:
        messages: Ordered list of LangChain messages.

    Returns:
        The last assistant message content, or an empty string if none found.

    """
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return str(message.content)
    return ""


def _extract_tool_payload(output: object) -> object:
    """Extract the tool payload from LangChain tool outputs.

    Args:
        output: Tool output payload, which may be a ToolMessage.

    Returns:
        The raw payload content for downstream parsing.

    """
    if isinstance(output, ToolMessage):
        if output.artifact is not None:
            return output.artifact
        return output.content
    return output


def _parse_tool_content(content: object) -> object:
    """Parse tool message content into Python objects when possible.

    Args:
        content: Tool content payload, often a JSON string.

    Returns:
        Parsed object when JSON is detected, otherwise the original content.

    """
    parsed: object = content
    if isinstance(content, str):
        stripped = content.strip()
        if stripped.startswith(("[", "{")) and stripped.endswith(("]", "}")):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = content
    elif isinstance(content, list):
        for block in content:
            if not isinstance(block, Mapping):
                continue
            if block.get("type") == "json" and "json" in block:
                parsed = block.get("json")
                break
            if block.get("type") == "text" and "text" in block:
                text = block.get("text")
                if isinstance(text, str):
                    parsed = _parse_tool_content(text)
                    break
    return parsed


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
