"""Process-wide logging setup and the per-turn log context.

`configure_logging` installs one stderr handler on the root logger, whose
format carries the conversation's `thread_id` and `customer_id` on every
line. Those two values come from `log_context`, which binds them to the
current `contextvars` context. Every asyncio task LangGraph starts for a turn
copies that context, so a line logged inside a node or a tool is attributable
to its guest and conversation without passing either value down every call.

Given an Axiom API key and dataset, it also ships every record to Axiom over
OTLP with the OpenTelemetry SDK, so the log outlives the container. The
handler on the root logger only converts a record and queues it; the SDK's
batch processor sends from its own thread, so logging never waits on the
network and an Axiom outage never raises into the code that logged.
"""

from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.instrumentation.logging.handler import LoggingHandler
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource

if TYPE_CHECKING:
    from collections.abc import Iterator

    from opentelemetry.sdk._logs.export import LogRecordExporter

    from blue_horizon.config import AxiomLoggingConfig, LoggingConfig

_LOG_FORMAT = (
    "%(asctime)s %(levelname)s %(name)s "
    "[thread_id=%(thread_id)s customer_id=%(customer_id)s] %(message)s"
)
# Rendered in place of a context value that is not bound, such as a line
# logged during startup, outside any turn.
_UNBOUND = "-"
# Names this process in Axiom, where the UI writes to the same dataset.
_SERVICE_NAME = "blue-horizon-api"
# The SDK names its batch export thread with this prefix.
_EXPORT_THREAD_PREFIX = "OtelBatch"

_thread_id: ContextVar[str | None] = ContextVar("log_thread_id", default=None)
_customer_id: ContextVar[int | None] = ContextVar("log_customer_id", default=None)


def configure_logging(
    config: LoggingConfig,
    *,
    axiom_api_key: str | None = None,
    axiom_dataset: str | None = None,
) -> None:
    """Install the context-aware handlers on the root logger.

    Safe to call more than once: each handler is added only if the root
    logger does not already have one, while the levels are applied on every
    call. Records are shipped to Axiom only when both `axiom_api_key` and
    `axiom_dataset` are non-empty.

    Args:
        config: Root log level, the third-party loggers held at ``WARNING``,
            and the Axiom endpoint and batching settings.
        axiom_api_key: Axiom API token with ingest permission, or ``None`` to
            log to stderr only.
        axiom_dataset: Axiom dataset to ship to, or ``None`` to log to stderr
            only.

    """
    root = logging.getLogger()
    if not any(isinstance(h, _ContextStderrHandler) for h in root.handlers):
        handler = _ContextStderrHandler()
        handler.setFormatter(
            logging.Formatter(
                _LOG_FORMAT,
                defaults={"thread_id": _UNBOUND, "customer_id": _UNBOUND},
            ),
        )
        handler.addFilter(LogContextFilter())
        root.addHandler(handler)
    if (
        axiom_api_key
        and axiom_dataset
        and not any(isinstance(h, _ShippingHandler) for h in root.handlers)
    ):
        exporter = _axiom_exporter(config.axiom, axiom_api_key, axiom_dataset)
        root.addHandler(_build_shipping_handler(config.axiom, exporter))
    root.setLevel(config.level)
    for name in config.quiet_loggers:
        logging.getLogger(name).setLevel(logging.WARNING)


class _ContextStderrHandler(logging.Handler):
    """Write formatted records to whatever `sys.stderr` is at the time.

    `logging.StreamHandler` keeps the stream it was built with. When something
    swaps `sys.stderr` for a while, as pytest's output capture does, a handler
    built during the swap keeps writing to the swapped-in stream after it has
    been closed. Looking the stream up on each write avoids that.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Format one record and write it to the current `sys.stderr`.

        Args:
            record: The record to write.

        """
        try:
            sys.stderr.write(f"{self.format(record)}\n")
            sys.stderr.flush()
        except Exception:  # noqa: BLE001
            self.handleError(record)


class LogContextFilter(logging.Filter):
    """Attach the bound `thread_id` and `customer_id` to every record.

    Attached to each handler rather than to a logger, so records from every
    logger that reaches the root, third-party ones included, get both fields.
    A value that is not bound is left off the record: the stderr format
    renders it as a placeholder, and the Axiom event simply omits it.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Copy the bound log context onto `record`.

        Args:
            record: The record being handled.

        Returns:
            bool: Always True; this filter annotates records and never drops
            one.

        """
        thread_id = _thread_id.get()
        customer_id = _customer_id.get()
        if thread_id is not None:
            record.__dict__["thread_id"] = thread_id
        if customer_id is not None:
            record.__dict__["customer_id"] = customer_id
        return True


def _axiom_exporter(
    config: AxiomLoggingConfig, api_key: str, dataset: str,
) -> OTLPLogExporter:
    """Build the OTLP exporter for Axiom's logs endpoint.

    Args:
        config: The endpoint and the per-send timeout.
        api_key: Axiom API token with ingest permission.
        dataset: Axiom dataset to ship to.

    Returns:
        OTLPLogExporter: An exporter that authenticates with `api_key` and
        writes to `dataset`.

    """
    return OTLPLogExporter(
        endpoint=config.otlp_endpoint,
        headers={"Authorization": f"Bearer {api_key}", "X-Axiom-Dataset": dataset},
        timeout=config.export_timeout_s,
    )


def _build_shipping_handler(
    config: AxiomLoggingConfig, exporter: LogRecordExporter,
) -> _ShippingHandler:
    """Build the root handler that queues records for `exporter`.

    Args:
        config: Batch size, queue size, and flush interval.
        exporter: Where batches are sent.

    Returns:
        _ShippingHandler: A handler carrying the log context filter, whose
        batch processor thread is already running.

    """
    provider = LoggerProvider(
        resource=Resource.create({"service.name": _SERVICE_NAME}),
        # Shut down by the handler's close() instead, which logging's own
        # exit hook calls, so there is one shutdown path, not two.
        shutdown_on_exit=False,
    )
    provider.add_log_record_processor(
        BatchLogRecordProcessor(
            exporter,
            schedule_delay_millis=config.flush_interval_s * 1000,
            max_export_batch_size=config.batch_size,
            max_queue_size=config.max_queue_size,
        ),
    )
    handler = _ShippingHandler(provider)
    handler.addFilter(LogContextFilter())
    handler.addFilter(_not_from_export_thread)
    return handler


class _ShippingHandler(LoggingHandler):
    """The OpenTelemetry logging handler, owning its logger provider.

    Owning the provider lets `close` flush and shut it down exactly once,
    whether shutdown comes from the API's lifespan or from logging's exit
    hook.
    """

    def __init__(self, provider: LoggerProvider) -> None:
        """Create a handler that emits through `provider`.

        Args:
            provider: The logger provider whose processor ships records.

        """
        super().__init__(logger_provider=provider, log_code_attributes=True)
        self._provider = provider
        self._closed = False

    def close(self) -> None:
        """Send what is queued and shut the provider down, once."""
        if not self._closed:
            self._closed = True
            self._provider.shutdown()
        super().close()


def _not_from_export_thread(record: logging.LogRecord) -> bool:
    """Keep records logged by the SDK's export thread from being shipped.

    The exporter logs its own failures and retries, as does `urllib3`
    beneath it. Shipping those would send a failure report through the
    pipeline that just failed. They still reach stderr.

    Args:
        record: The record being handled.

    Returns:
        bool: False for a record logged on the export thread.

    """
    return not (record.threadName or "").startswith(_EXPORT_THREAD_PREFIX)


def stop_log_shipping() -> None:
    """Detach the shipping handler and send what it has queued.

    Called at shutdown, so the last lines before a restart reach Axiom. A
    no-op when records are not being shipped.
    """
    root = logging.getLogger()
    for handler in [h for h in root.handlers if isinstance(h, _ShippingHandler)]:
        root.removeHandler(handler)
        handler.close()


@contextmanager
def log_context(
    *,
    thread_id: str | None = None,
    customer_id: int | None = None,
) -> Iterator[None]:
    """Bind a conversation and guest to every log line inside the block.

    A value left as ``None`` is not bound, so a nested block that names only
    one of the two keeps the other from the enclosing block.

    Args:
        thread_id: Conversation to attribute log lines to.
        customer_id: Guest to attribute log lines to.

    Yields:
        None. Both values are restored when the block exits.

    """
    thread_token = None if thread_id is None else _thread_id.set(thread_id)
    customer_token = None if customer_id is None else _customer_id.set(customer_id)
    try:
        yield
    finally:
        if customer_token is not None:
            _customer_id.reset(customer_token)
        if thread_token is not None:
            _thread_id.reset(thread_token)
