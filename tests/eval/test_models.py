"""Tests for eval/models.py.

Covers the parsing contract every evaluator relies on: unknown keys survive
via `extra="allow"`, a malformed list entry is dropped rather than failing
the whole list, `contexts_used` is filtered to non-empty strings, and
`ExampleTurn.reference` resolves its legacy alias chain.
"""

# ruff: noqa: S101

from __future__ import annotations

from eval.models import ExampleTurn, ToolSummaryEntry, TurnOutput, _validate_list

_ROWCOUNT = 3
_K_VALUE = 5
_LATENCY_MS = 42


class TestToolSummaryEntry:
    """ToolSummaryEntry declares the fields evaluators read, allows the rest."""

    def test_declared_fields_parse(self) -> None:
        """Declared fields are typed and attribute-accessible."""
        entry = ToolSummaryEntry.model_validate(
            {"tool": "run_sql", "status": "ok", "rowcount": _ROWCOUNT},
        )
        assert entry.tool == "run_sql"
        assert entry.status == "ok"
        assert entry.rowcount == _ROWCOUNT

    def test_unknown_key_survives_as_extra(self) -> None:
        """A key the callback captures but this model doesn't declare survives."""
        entry = ToolSummaryEntry.model_validate({"tool": "run_sql", "sql_query": "x"})
        assert entry.sql_query == "x"  # type: ignore[attr-defined]

    def test_undeclared_speculative_field_is_absent(self) -> None:
        """A field this entry never had at all is absent, not None-by-default."""
        entry = ToolSummaryEntry.model_validate({"tool": "run_sql"})
        assert getattr(entry, "query", "MISSING") == "MISSING"

    def test_filters_unknown_keys_defaults_to_empty_list(self) -> None:
        """Omitted filters_unknown_keys defaults to [], not None."""
        entry = ToolSummaryEntry.model_validate({"tool": "query_amenities"})
        assert entry.filters_unknown_keys == []

    def test_model_dump_includes_extras(self) -> None:
        """model_dump() round-trips extras -- _judge.py's transcript relies on it."""
        entry = ToolSummaryEntry.model_validate({"tool": "run_sql", "k": _K_VALUE})
        dumped = entry.model_dump(mode="json", exclude_none=True)
        assert dumped["k"] == _K_VALUE
        assert dumped["tool"] == "run_sql"


class TestTurnOutput:
    """TurnOutput best-effort parses tool_summary and contexts_used."""

    def test_defaults_are_empty(self) -> None:
        """A turn with no keys at all parses to empty defaults, not an error."""
        turn = TurnOutput()
        assert turn.tool_summary == []
        assert turn.contexts_used == []
        assert turn.route_pred is None

    def test_malformed_tool_summary_entry_is_dropped(self) -> None:
        """A tool_summary entry that fails validation is dropped, not fatal."""
        turn = TurnOutput.model_validate(
            {
                "tool_summary": [
                    {"tool": "query_faq", "status": "ok"},
                    "not a dict",
                    {"rowcount": "not a number"},
                ],
            },
        )
        assert len(turn.tool_summary) == 1
        assert turn.tool_summary[0].tool == "query_faq"

    def test_non_list_tool_summary_becomes_empty(self) -> None:
        """A non-list tool_summary value is treated as absent."""
        turn = TurnOutput.model_validate({"tool_summary": "oops"})
        assert turn.tool_summary == []

    def test_contexts_used_drops_none_and_empty_and_non_str(self) -> None:
        """contexts_used keeps only non-empty strings."""
        turn = TurnOutput.model_validate(
            {"contexts_used": [None, "good", "", 5, "also good"]},
        )
        assert turn.contexts_used == ["good", "also good"]

    def test_non_list_contexts_used_becomes_empty(self) -> None:
        """A non-list contexts_used value is treated as absent."""
        turn = TurnOutput.model_validate({"contexts_used": "oops"})
        assert turn.contexts_used == []

    def test_unknown_key_survives_as_extra(self) -> None:
        """A key not declared on TurnOutput round-trips via extra='allow'."""
        turn = TurnOutput.model_validate({"latency_ms": _LATENCY_MS})
        assert turn.latency_ms == _LATENCY_MS  # type: ignore[attr-defined]


class TestExampleTurn:
    """ExampleTurn resolves the reference alias chain and typed labels."""

    def test_reference_direct_key(self) -> None:
        """The 'reference' key populates .reference directly."""
        turn = ExampleTurn.model_validate({"reference": "R1"})
        assert turn.reference == "R1"

    def test_reference_falls_back_to_expected_answer(self) -> None:
        """Legacy 'expected_answer' key populates .reference when present."""
        turn = ExampleTurn.model_validate({"expected_answer": "R2"})
        assert turn.reference == "R2"

    def test_reference_falls_back_to_ground_truth(self) -> None:
        """Legacy 'ground_truth' key populates .reference when present."""
        turn = ExampleTurn.model_validate({"ground_truth": "R3"})
        assert turn.reference == "R3"

    def test_reference_key_wins_when_multiple_present(self) -> None:
        """'reference' takes precedence over the legacy aliases."""
        turn = ExampleTurn.model_validate(
            {"reference": "R1", "expected_answer": "R2", "ground_truth": "R3"},
        )
        assert turn.reference == "R1"

    def test_reference_absent_is_none(self) -> None:
        """No reference key at all leaves .reference as None."""
        turn = ExampleTurn.model_validate({"user": "hi"})
        assert turn.reference is None

    def test_expected_filters_absent_is_none(self) -> None:
        """A turn without expected_filters parses to None, not {}."""
        turn = ExampleTurn.model_validate({"user": "hi"})
        assert turn.expected_filters is None

    def test_expected_filters_present(self) -> None:
        """A turn with expected_filters keeps the raw dict for later canonicalizing."""
        turn = ExampleTurn.model_validate(
            {"user": "hi", "expected_filters": {"max_price": 150}},
        )
        assert turn.expected_filters == {"max_price": 150}

    def test_unknown_key_survives_as_extra(self) -> None:
        """A key not declared on ExampleTurn round-trips via extra='allow'."""
        turn = ExampleTurn.model_validate({"tags": ["info", "filters"]})
        assert turn.tags == ["info", "filters"]  # type: ignore[attr-defined]


class TestValidateList:
    """_validate_list is the shared best-effort list parser."""

    def test_non_list_input_returns_empty(self) -> None:
        """A non-list value (e.g. a stray string) parses to an empty list."""
        assert _validate_list(ToolSummaryEntry, "not a list") == []

    def test_none_input_returns_empty(self) -> None:
        """None parses to an empty list."""
        assert _validate_list(ToolSummaryEntry, None) == []

    def test_mixed_valid_and_invalid_entries(self) -> None:
        """Invalid entries are dropped; valid ones survive, in order."""
        result = _validate_list(
            ToolSummaryEntry,
            [{"tool": "a"}, 123, {"tool": "b"}, "nope", {"tool": "c"}],
        )
        assert [entry.tool for entry in result] == ["a", "b", "c"]

    def test_empty_list_returns_empty(self) -> None:
        """An empty list stays an empty list."""
        assert _validate_list(ToolSummaryEntry, []) == []
