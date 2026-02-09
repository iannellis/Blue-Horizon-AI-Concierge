"""Shared utility functions for eval package."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping


def truncate(text: object, limit: int) -> str:
    """Convert a value to a truncated string.

    Handles None values and coerces non-strings to strings before truncating.

    Args:
        text: Value to coerce into a string and truncate.
        limit: Maximum character length (including ellipsis).

    Returns:
        Truncated string value with "..." appended when trimming occurs.

    """
    if limit <= 0:
        return ""
    if text is None:
        return ""
    text_str = str(text) if not isinstance(text, str) else text
    if len(text_str) <= limit:
        return text_str
    if limit <= 3:  # noqa: PLR2004
        return text_str[:limit]
    return f"{text_str[: limit - 3]}..."


def coerce_int(value: object) -> int | None:
    """Coerce a value to int when possible.

    Args:
        value: Raw value to coerce.

    Returns:
        Integer value when coercible, otherwise None.

    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value.strip()))
        except ValueError:
            return None
    return None


def coerce_float(value: object) -> float | None:
    """Coerce a value to float when possible.

    Args:
        value: Raw value to coerce.

    Returns:
        Float value when coercible, otherwise None.

    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def coerce_strict_bool(value: object) -> bool | None:
    """Coerce a value to bool using strict, string-only rules.

    Args:
        value: Raw value to coerce.

    Returns:
        Boolean value when coercible, otherwise None.

    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "y", "1"}:
            return True
        if normalized in {"false", "no", "n", "0"}:
            return False
    return None


def json_safe(value: object) -> object:
    """Serialize unsupported types into JSON-safe structures recursively.

    Handles nested structures, pydantic models, dataclasses, and arbitrary
    objects by converting them to JSON-serializable types.

    Args:
        value: Value to serialize.

    Returns:
        JSON-serializable representation.

    """
    if value is None:
        serialized: object = None
    elif isinstance(value, (str, int, float, bool)):
        serialized = value
    elif isinstance(value, Mapping):
        serialized = {key: json_safe(val) for key, val in value.items()}
    elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        serialized = [json_safe(item) for item in value]
    else:
        model_dump = getattr(value, "model_dump", None)
        dict_dump = getattr(value, "dict", None)
        if callable(model_dump):
            serialized = json_safe(model_dump())
        elif callable(dict_dump):
            serialized = json_safe(dict_dump())
        elif hasattr(value, "__dict__"):
            serialized = json_safe(vars(value))
        else:
            serialized = str(value)
    return serialized


def json_value(obj: object, *, max_len: int | None = None) -> str:
    """Serialize a value into JSON for safe LangSmith feedback.

    Args:
        obj: Object to serialize.
        max_len: Optional maximum length for the JSON string.

    Returns:
        JSON string representation of the object, truncated when requested.

    """
    try:
        payload = json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001
        try:
            payload = json.dumps(repr(obj), ensure_ascii=False)
        except Exception:  # noqa: BLE001
            payload = json.dumps("<unserializable>", ensure_ascii=False)

    if max_len is not None and max_len > 0 and len(payload) > max_len:
        return f"{payload[:max_len]}..."
    return payload
