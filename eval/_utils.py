"""Shared utility functions for eval package."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


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
    """Coerce a value to a finite float when possible.

    The single numeric coercer for the eval package: LangSmith feedback
    scores, Ragas metric results (once unwrapped -- see
    `eval.evaluators._rag._metric_result_to_float`), and result-aggregation
    payloads all funnel through this.

    Args:
        value: Raw value to coerce.

    Returns:
        Float value when coercible and finite, otherwise None. Booleans are
        rejected -- a score field holding True/False is a shape bug
        upstream, not a 1.0/0.0 score. NaN and infinite values are also
        rejected, since a judge or retrieval score that isn't a real number
        cannot be averaged or compared against a threshold meaningfully.

    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        result = float(value)
    elif isinstance(value, str):
        try:
            result = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


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

    The `model_dump`/`dict` check must run before the `Iterable` check: a
    Pydantic v2 `BaseModel` is itself iterable (yielding `(field_name,
    value)` pairs), so checking `Iterable` first silently turns every model
    -- e.g. a LangSmith `EvaluationResult` -- into a flat list of
    `[name, value]` pairs instead of a proper `{field: value}` dict.

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
    else:
        model_dump = getattr(value, "model_dump", None)
        dict_dump = getattr(value, "dict", None)
        if callable(model_dump):
            serialized = json_safe(model_dump())
        elif callable(dict_dump):
            serialized = json_safe(dict_dump())
        elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            serialized = [json_safe(item) for item in value]
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


def json_detail_metric(*, key: str, data: object, max_len: int) -> dict[str, Any]:
    """Build one JSON-value LangSmith metric, truncating if it exceeds max_len.

    Shared by every evaluator that reports a details/failures list as a
    single ``value`` metric (`_info_filters`, `_info_references`,
    `_injection`, `_rag`, `_booking`): each used to repeat this same
    serialize-then-check-length dance inline.

    Args:
        key: LangSmith metric key.
        data: JSON-serializable detail payload (typically a list of records).
        max_len: Maximum serialized length before truncation.

    Returns:
        A LangSmith value metric dict: `{"key": key, "value": <json>}`, with
        `"comment": "JSON truncated"` added when the payload was cut down.

    """
    raw = json_value(data)
    if len(raw) <= max_len:
        return {"key": key, "value": raw}
    return {
        "key": key,
        "value": json_value(data, max_len=max_len),
        "comment": "JSON truncated",
    }


def configure_logging(run_name: str, log_dir: Path) -> None:
    """Configure file and stderr logging for an evaluation or stress run.

    Attaches both a file handler (``<log_dir>/<run_name>.log``) and a stderr
    stream handler so that progress is visible in the terminal as well as
    persisted to disk.

    Args:
        run_name: Name of the current run, used as the log filename stem.
        log_dir: Directory where log files should be written.

    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_name}.log"

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(file_handler)
    root.addHandler(stream_handler)
