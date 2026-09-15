"""Tests for `blue_horizon.logging_setup`.

Covers the log context's binding, nesting, and restoration, its propagation
into the asyncio tasks a LangGraph turn runs in, `configure_logging`'s
idempotence, and the Axiom shipping path. None of this touches the network:
the OTLP exporter is replaced by a fake that records what it is sent.
"""
# ruff: noqa: S101, SLF001

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, TypedDict

import pytest
from langgraph.graph import END, START, StateGraph
from opentelemetry.sdk._logs.export import LogRecordExporter, LogRecordExportResult

from blue_horizon import logging_setup
from blue_horizon.config import AxiomLoggingConfig, LoggingConfig
from blue_horizon.logging_setup import (
    LogContextFilter,
    configure_logging,
    log_context,
    stop_log_shipping,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from opentelemetry.sdk._logs import ReadableLogRecord

_THREAD_ID = "thread-1"
_CUSTOMER_ID = 7
_QUIET_LOGGER = "tests.logging_setup.quiet"
_SHIPPED_LOGGER = "tests.logging_setup.shipped"
_AXIOM_CONFIG = AxiomLoggingConfig(
    otlp_endpoint="https://example.invalid/v1/logs",
    batch_size=100,
    max_queue_size=1000,
    flush_interval_s=0.05,
    export_timeout_s=1.0,
)


class _RecordingExporter(LogRecordExporter):
    """Record exported batches in place of the OTLP exporter.

    Attributes:
        batches: Each batch passed to `export`, in order.
        fail: When True, `export` raises instead of recording.

    """

    def __init__(self, *, fail: bool = False) -> None:
        """Create an exporter with no batches recorded.

        Args:
            fail: Whether every export should raise.

        """
        self.batches: list[list[ReadableLogRecord]] = []
        self.fail = fail

    def export(self, batch: Sequence[ReadableLogRecord]) -> LogRecordExportResult:
        """Record one batch, or raise if the exporter is set to fail.

        Args:
            batch: The batch's records.

        Returns:
            LogRecordExportResult: Always ``SUCCESS`` when it returns.

        Raises:
            ConnectionError: If `fail` is True.

        """
        if self.fail:
            msg = "axiom unreachable"
            raise ConnectionError(msg)
        self.batches.append(list(batch))
        return LogRecordExportResult.SUCCESS

    def shutdown(self) -> None:
        """Do nothing; there is no connection to close."""

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Report success; nothing is buffered here.

        Args:
            timeout_millis: Unused.

        Returns:
            bool: Always True.

        """
        _ = timeout_millis
        return True

    @property
    def records(self) -> list[ReadableLogRecord]:
        """Every record received, in order."""
        return [record for batch in self.batches for record in batch]


def _attributes(record: ReadableLogRecord) -> Mapping[str, object]:
    """Read a shipped record's attributes.

    Args:
        record: A record the exporter received.

    Returns:
        Mapping[str, object]: Its attributes, empty if it has none.

    """
    return record.log_record.attributes or {}


def _context_fields() -> tuple[object, object]:
    """Run a fresh record through `LogContextFilter` and read both fields.

    Returns:
        tuple[object, object]: The record's `(thread_id, customer_id)`, each
        ``None`` when the filter left it off.

    """
    record = logging.LogRecord("test", logging.INFO, __file__, 1, "msg", None, None)
    assert LogContextFilter().filter(record) is True
    return record.__dict__.get("thread_id"), record.__dict__.get("customer_id")


def _shipping_logger(
    exporter: _RecordingExporter, config: AxiomLoggingConfig = _AXIOM_CONFIG,
) -> tuple[logging.Logger, logging.Handler]:
    """Build a non-propagating logger whose only handler ships to `exporter`.

    Args:
        exporter: The fake the batch processor sends to.
        config: Batching settings for the processor.

    Returns:
        tuple[logging.Logger, logging.Handler]: The logger, and its handler,
        which the caller closes to flush.

    """
    handler = logging_setup._build_shipping_handler(config, exporter)
    logger = logging.getLogger(_SHIPPED_LOGGER)
    logger.handlers[:] = [handler]
    logger.propagate = False
    logger.setLevel(logging.INFO)
    return logger, handler


def _handlers_named(name: str) -> list[logging.Handler]:
    """List the root logger's handlers of one class.

    Args:
        name: The handler class name.

    Returns:
        list[logging.Handler]: Every root handler of that class.

    """
    return [h for h in logging.getLogger().handlers if type(h).__name__ == name]


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

    def test_unbound_context_is_left_off_the_record(self) -> None:
        """Outside any turn, neither field is attached."""
        assert _context_fields() == (None, None)

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
        assert _context_fields() == (None, None)

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
        config = LoggingConfig(level="INFO", quiet_loggers=(), axiom=_AXIOM_CONFIG)
        configure_logging(config)
        configure_logging(config)
        assert len(_handlers_named("_ContextStderrHandler")) == 1

    def test_stderr_line_renders_unbound_context_as_placeholder(self) -> None:
        """A line logged outside any turn shows ``-`` for both fields."""
        configure_logging(
            LoggingConfig(level="INFO", quiet_loggers=(), axiom=_AXIOM_CONFIG),
        )
        (handler,) = _handlers_named("_ContextStderrHandler")
        record = logging.LogRecord("test", logging.INFO, __file__, 1, "msg", None, None)
        handler.filter(record)
        assert "[thread_id=- customer_id=-] msg" in handler.format(record)

    def test_applies_root_level_and_quiets_named_loggers(self) -> None:
        """The root takes the configured level; quiet loggers sit at WARNING."""
        configure_logging(
            LoggingConfig(
                level="DEBUG", quiet_loggers=(_QUIET_LOGGER,), axiom=_AXIOM_CONFIG,
            ),
        )
        assert logging.getLogger().level == logging.DEBUG
        assert logging.getLogger(_QUIET_LOGGER).level == logging.WARNING

    def test_no_shipping_handler_without_both_settings(self) -> None:
        """A key without a dataset, or a blank key, ships nothing."""
        config = LoggingConfig(level="INFO", quiet_loggers=(), axiom=_AXIOM_CONFIG)
        configure_logging(config, axiom_api_key="key", axiom_dataset=None)
        configure_logging(config, axiom_api_key="", axiom_dataset="dataset")
        assert not _handlers_named("_ShippingHandler")

    def test_shipping_handler_added_once_and_removed_by_stop(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With both settings one handler ships; stopping detaches it."""
        monkeypatch.setattr(
            logging_setup,
            "_axiom_exporter",
            lambda *_args: _RecordingExporter(),
        )
        config = LoggingConfig(level="INFO", quiet_loggers=(), axiom=_AXIOM_CONFIG)
        configure_logging(config, axiom_api_key="key", axiom_dataset="dataset")
        configure_logging(config, axiom_api_key="key", axiom_dataset="dataset")
        assert len(_handlers_named("_ShippingHandler")) == 1
        stop_log_shipping()
        assert not _handlers_named("_ShippingHandler")


class TestAxiomExporter:
    """`_axiom_exporter`."""

    def test_sends_the_token_and_dataset_to_the_configured_endpoint(self) -> None:
        """The exporter targets the endpoint with Axiom's two headers."""
        exporter = logging_setup._axiom_exporter(_AXIOM_CONFIG, "xaat-key", "ds")
        assert exporter._endpoint == _AXIOM_CONFIG.otlp_endpoint
        assert exporter._headers == {
            "Authorization": "Bearer xaat-key",
            "X-Axiom-Dataset": "ds",
        }
        exporter.shutdown()


class TestShipping:
    """The shipping handler and its batch processor."""

    def test_record_carries_message_level_and_log_context(self) -> None:
        """A shipped record names its level, message, guest, and thread."""
        exporter = _RecordingExporter()
        logger, handler = _shipping_logger(exporter)
        with log_context(thread_id=_THREAD_ID, customer_id=_CUSTOMER_ID):
            logger.info("proposal %s confirmed", "p-1")
        handler.close()

        (record,) = exporter.records
        attributes = _attributes(record)
        assert record.log_record.body == "proposal p-1 confirmed"
        assert record.log_record.severity_text == "INFO"
        assert attributes["thread_id"] == _THREAD_ID
        assert attributes["customer_id"] == _CUSTOMER_ID
        assert "code.function.name" in attributes
        assert record.resource.attributes["service.name"] == "blue-horizon-api"

    def test_unbound_context_is_omitted(self) -> None:
        """Outside a turn, neither context attribute is shipped."""
        exporter = _RecordingExporter()
        logger, handler = _shipping_logger(exporter)
        logger.info("startup")
        handler.close()

        (record,) = exporter.records
        assert "thread_id" not in _attributes(record)
        assert "customer_id" not in _attributes(record)

    def test_exception_ships_its_traceback(self) -> None:
        """A record logged with an exception carries the formatted traceback."""
        exporter = _RecordingExporter()
        logger, handler = _shipping_logger(exporter)
        try:
            msg = "boom"
            raise ValueError(msg)  # noqa: TRY301
        except ValueError:
            logger.exception("write failed")
        handler.close()

        (record,) = exporter.records
        assert "ValueError: boom" in str(_attributes(record)["exception.stacktrace"])

    def test_batches_respect_batch_size(self) -> None:
        """No batch exceeds `batch_size`, and closing sends every record."""
        exporter = _RecordingExporter()
        config = _AXIOM_CONFIG.model_copy(
            update={"batch_size": 2, "flush_interval_s": 30.0},
        )
        logger, handler = _shipping_logger(exporter, config)
        for n in range(5):
            logger.info("line %d", n)
        handler.close()

        assert all(len(batch) <= config.batch_size for batch in exporter.batches)
        assert [r.log_record.body for r in exporter.records] == [
            f"line {n}" for n in range(5)
        ]

    def test_failed_export_never_raises_to_the_caller(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """An unreachable Axiom is reported by the SDK, not raised."""
        exporter = _RecordingExporter(fail=True)
        logger, handler = _shipping_logger(exporter)
        logger.warning("lost line")
        with caplog.at_level(logging.ERROR):
            handler.close()

        messages = [r.getMessage() for r in caplog.records]
        assert any("Exception while exporting" in m for m in messages)

    def test_close_is_safe_to_repeat(self) -> None:
        """Closing twice, as the lifespan and logging's exit hook may, is harmless."""
        _, handler = _shipping_logger(_RecordingExporter())
        handler.close()
        handler.close()

    def test_export_thread_records_are_not_shipped(self) -> None:
        """A record logged on the SDK's export thread is kept off the pipeline."""
        record = logging.LogRecord("t", logging.WARNING, __file__, 1, "x", None, None)
        record.threadName = "OtelBatchLogRecordProcessor"
        assert logging_setup._not_from_export_thread(record) is False
        record.threadName = "MainThread"
        assert logging_setup._not_from_export_thread(record) is True
