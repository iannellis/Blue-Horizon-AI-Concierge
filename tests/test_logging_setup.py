"""Tests for `blue_horizon.logging_setup`.

Covers the log context's binding, nesting, and restoration, its propagation
into the asyncio tasks a LangGraph turn runs in, and `configure_logging`'s
idempotence. None of this touches the network.
"""
# ruff: noqa: S101

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, TypedDict

import pytest
from langgraph.graph import END, START, StateGraph

from blue_horizon.config import LoggingConfig
from blue_horizon.logging_setup import LogContextFilter, configure_logging, log_context

if TYPE_CHECKING:
    from collections.abc import Iterator

_THREAD_ID = "thread-1"
_CUSTOMER_ID = 7
_QUIET_LOGGER = "tests.logging_setup.quiet"


def _context_fields() -> tuple[object, object]:
    """Run a fresh record through `LogContextFilter` and read both fields.

    Returns:
        tuple[object, object]: The record's `(thread_id, customer_id)`.

    """
    record = logging.LogRecord("test", logging.INFO, __file__, 1, "msg", None, None)
    assert LogContextFilter().filter(record) is True
    return record.__dict__["thread_id"], record.__dict__["customer_id"]


@pytest.fixture
def restore_logging() -> Iterator[None]:
    """Restore the root logger and the quiet test logger after a test.

    Yields:
        None, while the test runs.

    """
    root = logging.getLogger()
    quiet = logging.getLogger(_QUIET_LOGGER)
    handlers, root_level, quiet_level = list(root.handlers), root.level, quiet.level
    yield
    root.handlers[:] = handlers
    root.setLevel(root_level)
    quiet.setLevel(quiet_level)


class TestLogContext:
    """`log_context` and `LogContextFilter`."""

    def test_unbound_context_renders_placeholder(self) -> None:
        """Outside any turn, both fields render as a placeholder."""
        assert _context_fields() == ("-", "-")

    def test_bound_context_is_attached(self) -> None:
        """Inside a block, both bound values reach the record."""
        with log_context(thread_id=_THREAD_ID, customer_id=_CUSTOMER_ID):
            assert _context_fields() == (_THREAD_ID, _CUSTOMER_ID)

    def test_context_is_restored_on_exit(self) -> None:
        """Leaving the block unbinds both values, even on an exception."""
        with (
            pytest.raises(RuntimeError),
            log_context(thread_id=_THREAD_ID, customer_id=_CUSTOMER_ID),
        ):
            raise RuntimeError
        assert _context_fields() == ("-", "-")

    def test_nested_block_keeps_the_value_it_does_not_name(self) -> None:
        """A nested block naming only the guest keeps the enclosing thread."""
        with (
            log_context(thread_id=_THREAD_ID),
            log_context(customer_id=_CUSTOMER_ID),
        ):
            assert _context_fields() == (_THREAD_ID, _CUSTOMER_ID)

    def test_context_reaches_a_langgraph_node(self) -> None:
        """A node LangGraph runs inside the block sees the bound values.

        This is what lets a line logged inside a tool or retrieval node name
        its guest without either value being passed down explicitly.
        """

        class _State(TypedDict, total=False):
            fields: tuple[object, object]

        async def _node(state: _State) -> dict[str, Any]:
            _ = state
            return {"fields": _context_fields()}

        graph = StateGraph(_State)
        graph.add_node("node", _node)
        graph.add_edge(START, "node")
        graph.add_edge("node", END)
        compiled = graph.compile()

        async def _run() -> dict[str, Any]:
            with log_context(thread_id=_THREAD_ID, customer_id=_CUSTOMER_ID):
                return await compiled.ainvoke({})

        assert asyncio.run(_run())["fields"] == (_THREAD_ID, _CUSTOMER_ID)


@pytest.mark.usefixtures("restore_logging")
class TestConfigureLogging:
    """`configure_logging`."""

    def test_repeat_calls_leave_one_context_handler(self) -> None:
        """However often it is called, the root has one context handler.

        Counted by type rather than against the handler count before the
        call, since an earlier test that ran the app's `lifespan` may already
        have installed one.
        """
        config = LoggingConfig(level="INFO", quiet_loggers=())
        configure_logging(config)
        configure_logging(config)
        context_handlers = [
            h
            for h in logging.getLogger().handlers
            if type(h).__name__ == "_ContextStderrHandler"
        ]
        assert len(context_handlers) == 1

    def test_applies_root_level_and_quiets_named_loggers(self) -> None:
        """The root takes the configured level; quiet loggers sit at WARNING."""
        configure_logging(
            LoggingConfig(level="DEBUG", quiet_loggers=(_QUIET_LOGGER,)),
        )
        assert logging.getLogger().level == logging.DEBUG
        assert logging.getLogger(_QUIET_LOGGER).level == logging.WARNING
