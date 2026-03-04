"""Filter normalization utilities for the eval harness.

Provides strict canonical-key normalization and coercion for info-tool
amenity/service filters used by evaluators.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from eval._utils import (
    coerce_float as _coerce_float,
)
from eval._utils import (
    coerce_int as _coerce_int,
)
from eval._utils import (
    coerce_strict_bool as _coerce_strict_bool,
)

# Filter keys present in ParsedQuery that map to retrieval metadata filters.
_INFO_FILTER_KEYS: frozenset[str] = frozenset(
    {
        "booking_required",
        "min_price",
        "max_price",
        "max_notice_hours",
        "min_duration_minutes",
        "max_duration_minutes",
    },
)


def _normalize_info_filters_strict(
    filters: Mapping[str, object] | None,
) -> tuple[dict[str, object] | None, list[str]]:
    """Normalize amenity/service filters using strict canonical keys.

    Args:
        filters: Raw filters dict passed to an info tool, if any.

    Returns:
        Tuple of (normalized filters or None, unknown key list).

    """
    if not filters:
        return None, []
    if not isinstance(filters, dict):
        return None, ["<non_dict_filters>"]

    canonical_keys = {
        "booking_required",
        "min_price",
        "max_price",
        "max_notice_hours",
        "min_duration_minutes",
        "max_duration_minutes",
    }

    norm: dict[str, object] = {}
    unknown_keys: list[str] = []

    for key, value in filters.items():
        canonical = _canonicalize_filter_key(str(key))
        if canonical not in canonical_keys:
            unknown_keys.append(str(key))
            continue

        coerced = _coerce_strict_filter_value(canonical, value)
        if coerced is None:
            unknown_keys.append(f"{key}:<bad_value>")
            continue
        norm[canonical] = coerced

    _swap_range_if_needed(norm, "min_price", "max_price")
    _swap_range_if_needed(norm, "min_duration_minutes", "max_duration_minutes")

    return (norm or None), unknown_keys


def _canonicalize_filter_key(key: str) -> str:
    """Normalize filter keys for strict comparisons.

    Args:
        key: Raw filter key.

    Returns:
        Canonicalized key string.

    """
    normalized = key.strip().lower().replace(" ", "_").replace("-", "_")
    normalized = re.sub(r"_+", "_", normalized)
    if normalized == "duration_mintues":
        return "duration_minutes"
    return normalized


def _coerce_strict_filter_value(
    canonical_key: str,
    value: object,
) -> object | None:
    """Coerce filter values based on a strict canonical key.

    Args:
        canonical_key: Canonical filter key.
        value: Raw filter value.

    Returns:
        Coerced value when parseable, otherwise None.

    """
    if canonical_key == "booking_required":
        return _coerce_strict_bool(value)
    if canonical_key.endswith("_price"):
        return _coerce_float(value)
    return _coerce_int(value)


def _swap_range_if_needed(
    norm: dict[str, object],
    min_key: str,
    max_key: str,
) -> None:
    """Swap min/max values if they are reversed.

    Args:
        norm: Normalized filters dict updated in place.
        min_key: Minimum value key.
        max_key: Maximum value key.

    """
    min_val = norm.get(min_key)
    max_val = norm.get(max_key)
    if (
        isinstance(min_val, (int, float))
        and not isinstance(min_val, bool)
        and isinstance(max_val, (int, float))
        and not isinstance(max_val, bool)
        and float(min_val) > float(max_val)
    ):
        norm[min_key], norm[max_key] = norm[max_key], norm[min_key]


def _extract_filters_from_tool_inputs(inputs: object) -> dict[str, object] | None:
    """Extract a filters dict from tool inputs payloads.

    Args:
        inputs: Tool inputs payload, often a mapping.

    Returns:
        Filters dict if present, otherwise None.

    """
    if not isinstance(inputs, Mapping):
        return None
    raw_filters = inputs.get("filters")
    if isinstance(raw_filters, Mapping):
        return dict(raw_filters)

    excluded_keys = {"query", "k", "top_k"}
    extracted = {
        str(key): value
        for key, value in inputs.items()
        if str(key) not in excluded_keys
    }
    return extracted or None
