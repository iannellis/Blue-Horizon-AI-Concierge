"""Tests for eval/_utils.py helper functions.

All functions are pure; no I/O or mocking required.
"""

# ruff: noqa: S101, FBT003

import json
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from eval._utils import (
    coerce_float,
    coerce_int,
    coerce_strict_bool,
    json_safe,
    json_value,
    truncate,
)

# ---------------------------------------------------------------------------
# truncate
# ---------------------------------------------------------------------------


class TestTruncate:
    """truncate() coerces values to strings and enforces a character limit."""

    def test_none_returns_empty_string(self) -> None:
        """None input produces an empty string."""
        assert truncate(None, 100) == ""

    def test_zero_limit_returns_empty_string(self) -> None:
        """A limit of 0 always returns an empty string."""
        assert truncate("hello", 0) == ""

    def test_negative_limit_returns_empty_string(self) -> None:
        """A negative limit returns an empty string."""
        assert truncate("hello", -1) == ""

    def test_short_text_returned_as_is(self) -> None:
        """Text shorter than limit passes through unchanged."""
        assert truncate("hi", 10) == "hi"

    def test_text_exactly_at_limit_not_truncated(self) -> None:
        """Text exactly equal to limit is returned unchanged."""
        assert truncate("hello", 5) == "hello"

    def test_text_over_limit_gets_ellipsis(self) -> None:
        """Text longer than limit is sliced and suffixed with '...'."""
        result = truncate("abcdefgh", 6)
        assert result == "abc..."
        assert len(result) == 6  # noqa: PLR2004

    def test_limit_of_one_no_ellipsis(self) -> None:
        """With limit<=3, raw slice is returned without ellipsis."""
        assert truncate("hello", 1) == "h"

    def test_limit_of_two_no_ellipsis(self) -> None:
        """limit=2 returns first two characters, no ellipsis."""
        assert truncate("hello", 2) == "he"

    def test_limit_of_three_no_ellipsis(self) -> None:
        """limit=3 returns first three characters, no ellipsis."""
        assert truncate("hello", 3) == "hel"

    def test_limit_of_four_uses_ellipsis(self) -> None:
        """limit=4 uses one character + '...'."""
        assert truncate("hello", 4) == "h..."

    def test_non_string_coerced(self) -> None:
        """Non-string values are coerced via str() before truncating."""
        assert truncate(42, 10) == "42"

    def test_int_truncated_when_over_limit(self) -> None:
        """An integer that is longer than limit when str()-ed is truncated."""
        result = truncate(123456789, 5)
        assert result == "12..."
        assert len(result) == 5  # noqa: PLR2004


# ---------------------------------------------------------------------------
# coerce_int
# ---------------------------------------------------------------------------


class TestCoerceInt:
    """coerce_int() converts values to int where unambiguously possible."""

    def test_int_returned_directly(self) -> None:
        """An int is returned unchanged."""
        assert coerce_int(7) == 7  # noqa: PLR2004

    def test_float_truncated_to_int(self) -> None:
        """A float is truncated (not rounded) to int."""
        assert coerce_int(3.9) == 3  # noqa: PLR2004

    def test_string_integer(self) -> None:
        """A string containing an integer is parsed."""
        assert coerce_int("42") == 42  # noqa: PLR2004

    def test_string_float_truncated(self) -> None:
        """A string like '3.7' is parsed as float then truncated to int."""
        assert coerce_int("3.7") == 3  # noqa: PLR2004

    def test_string_with_whitespace(self) -> None:
        """Leading/trailing whitespace in string is stripped before parsing."""
        assert coerce_int("  5  ") == 5  # noqa: PLR2004

    def test_non_numeric_string_returns_none(self) -> None:
        """A non-numeric string returns None."""
        assert coerce_int("abc") is None

    def test_none_returns_none(self) -> None:
        """None returns None."""
        assert coerce_int(None) is None

    def test_bool_true_returns_none(self) -> None:
        """True (a bool subclass of int) returns None, not 1."""
        assert coerce_int(True) is None

    def test_bool_false_returns_none(self) -> None:
        """False (a bool subclass of int) returns None, not 0."""
        assert coerce_int(False) is None

    def test_list_returns_none(self) -> None:
        """An unsupported type returns None."""
        assert coerce_int([1, 2]) is None


# ---------------------------------------------------------------------------
# coerce_float
# ---------------------------------------------------------------------------


class TestCoerceFloat:
    """coerce_float() converts values to float where unambiguously possible."""

    def test_int_becomes_float(self) -> None:
        """An int is widened to float."""
        assert coerce_float(3) == 3.0  # noqa: PLR2004

    def test_float_returned_directly(self) -> None:
        """A float is returned as-is."""
        assert coerce_float(1.5) == 1.5  # noqa: PLR2004

    def test_string_float(self) -> None:
        """A string containing a float is parsed."""
        assert coerce_float("3.14") == pytest.approx(3.14)

    def test_string_integer(self) -> None:
        """A string containing an integer is parsed as float."""
        assert coerce_float("1") == 1.0

    def test_string_with_whitespace(self) -> None:
        """Whitespace-padded string is handled."""
        assert coerce_float("  2.5  ") == pytest.approx(2.5)

    def test_non_numeric_string_returns_none(self) -> None:
        """A non-numeric string returns None."""
        assert coerce_float("abc") is None

    def test_none_returns_none(self) -> None:
        """None returns None."""
        assert coerce_float(None) is None

    def test_bool_true_returns_none(self) -> None:
        """True returns None to prevent bool-as-number coercion."""
        assert coerce_float(True) is None

    def test_bool_false_returns_none(self) -> None:
        """False returns None."""
        assert coerce_float(False) is None


# ---------------------------------------------------------------------------
# coerce_strict_bool
# ---------------------------------------------------------------------------


class TestCoerceStrictBool:
    """coerce_strict_bool() converts values using an explicit allowlist."""

    # --- literal bool passthrough ---
    def test_true_passthrough(self) -> None:
        """Literal True is returned as True."""
        assert coerce_strict_bool(True) is True

    def test_false_passthrough(self) -> None:
        """Literal False is returned as False."""
        assert coerce_strict_bool(False) is False

    # --- truthy string values ---
    @pytest.mark.parametrize("value", ["true", "True", "TRUE", "yes", "y", "1"])
    def test_truthy_string(self, value: str) -> None:
        """Canonical truthy strings are coerced to True."""
        assert coerce_strict_bool(value) is True

    # --- falsy string values ---
    @pytest.mark.parametrize("value", ["false", "False", "FALSE", "no", "n", "0"])
    def test_falsy_string(self, value: str) -> None:
        """Canonical falsy strings are coerced to False."""
        assert coerce_strict_bool(value) is False

    # --- strings with surrounding whitespace ---
    def test_whitespace_stripped(self) -> None:
        """Surrounding whitespace is stripped before matching."""
        assert coerce_strict_bool("  true  ") is True

    # --- non-canonical inputs ---
    def test_ambiguous_string_returns_none(self) -> None:
        """An unrecognised string returns None."""
        assert coerce_strict_bool("maybe") is None

    def test_none_returns_none(self) -> None:
        """None is not coercible."""
        assert coerce_strict_bool(None) is None

    def test_integer_one_returns_none(self) -> None:
        """Plain int 1 is not treated as True (only bool and str accepted)."""
        assert coerce_strict_bool(1) is None

    def test_integer_zero_returns_none(self) -> None:
        """Plain int 0 is not treated as False."""
        assert coerce_strict_bool(0) is None


# ---------------------------------------------------------------------------
# json_safe
# ---------------------------------------------------------------------------


class TestJsonSafe:
    """json_safe() recursively converts values to JSON-serialisable structures."""

    def test_none_stays_none(self) -> None:
        """None is preserved as None."""
        assert json_safe(None) is None

    def test_str_passthrough(self) -> None:
        """Strings pass through unchanged."""
        assert json_safe("hello") == "hello"

    def test_int_passthrough(self) -> None:
        """Integers pass through unchanged."""
        assert json_safe(42) == 42  # noqa: PLR2004

    def test_float_passthrough(self) -> None:
        """Floats pass through unchanged."""
        assert json_safe(1.5) == 1.5  # noqa: PLR2004

    def test_bool_passthrough(self) -> None:
        """Booleans pass through unchanged."""
        assert json_safe(True) is True

    def test_dict_recurses(self) -> None:
        """Dict values are recursively processed."""
        assert json_safe({"a": 1, "b": None}) == {"a": 1, "b": None}

    def test_list_recurses(self) -> None:
        """List items are recursively processed."""
        assert json_safe([1, "two", None]) == [1, "two", None]

    def test_nested_structure(self) -> None:
        """Nested dicts and lists are all converted."""
        data = {"outer": [{"inner": 42}]}
        assert json_safe(data) == data

    def test_pydantic_model_serialised_as_key_value_pairs(self) -> None:
        """Pydantic v2 BaseModel is Iterable; json_safe yields key-value pair lists.

        In Pydantic v2, BaseModel implements __iter__ yielding (key, value) tuples,
        so json_safe hits the Iterable branch before the model_dump fallback,
        producing a list of [key, value] pairs rather than a dict.
        """

        class M(BaseModel):
            x: int = 1

        result = json_safe(M())
        assert result == [["x", 1]]

    def test_dataclass_uses_dict(self) -> None:
        """Dataclass instances are serialised via __dict__ (vars())."""

        @dataclass
        class DC:
            name: str = "test"
            value: int = 99

        result = json_safe(DC())
        assert isinstance(result, dict)
        assert result["name"] == "test"
        assert result["value"] == 99  # noqa: PLR2004

    def test_simple_namespace_uses_dict(self) -> None:
        """Objects with __dict__ fall back to vars()."""
        obj = SimpleNamespace(foo="bar", num=7)
        result = json_safe(obj)
        assert result == {"foo": "bar", "num": 7}

    def test_bytes_converted_to_string(self) -> None:
        """Bytes are not treated as an iterable; they fall through to str()."""
        result = json_safe(b"hello")
        assert isinstance(result, str)

    def test_arbitrary_object_falls_back_to_str(self) -> None:
        """Objects with no known serialisation protocol become strings."""

        class Opaque:
            def __repr__(self) -> str:
                return "Opaque()"

            # No __dict__, model_dump, or dict
            __slots__ = ()

        result = json_safe(Opaque())
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# json_value
# ---------------------------------------------------------------------------


class TestJsonValue:
    """json_value() serialises objects to JSON strings with optional truncation."""

    def test_none_serialised_as_null(self) -> None:
        """None becomes the JSON string 'null'."""
        assert json_value(None) == "null"

    def test_dict_serialised(self) -> None:
        """A dict is serialised to a valid JSON string."""
        result = json_value({"key": "val"})
        assert json.loads(result) == {"key": "val"}

    def test_list_serialised(self) -> None:
        """A list is serialised to a valid JSON array string."""
        result = json_value([1, 2, 3])
        assert json.loads(result) == [1, 2, 3]

    def test_no_max_len_no_truncation(self) -> None:
        """Without max_len, the full JSON string is returned."""
        long_str = "x" * 1000
        result = json_value({"data": long_str})
        assert "..." not in result
        assert long_str in result

    def test_max_len_truncates_with_ellipsis(self) -> None:
        """A max_len shorter than the output appends '...'."""
        result = json_value({"data": "hello world"}, max_len=5)
        assert result.endswith("...")
        assert len(result) == 8  # noqa: PLR2004  # 5 chars + "..."

    def test_max_len_not_exceeded_no_truncation(self) -> None:
        """When output is shorter than max_len, no truncation occurs."""
        result = json_value(42, max_len=100)
        assert result == "42"
        assert "..." not in result

    def test_max_len_zero_no_truncation(self) -> None:
        """max_len=0 is treated as 'no limit' (guard condition is max_len > 0)."""
        result = json_value("hello", max_len=0)
        assert result == '"hello"'
