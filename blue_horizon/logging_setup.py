"""Process-wide logging setup and the per-turn log context.

`configure_logging` installs one stderr handler on the root logger, whose
format carries the conversation's `thread_id` and `customer_id` on every
line. Those two values come from `log_context`, which binds them to the
current `contextvars` context. Every asyncio task LangGraph starts for a turn
copies that context, so a line logged inside a node or a tool is attributable
to its guest and conversation without passing either value down every call.
"""

from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    from blue_horizon.config import LoggingConfig

_LOG_FORMAT = (
    "%(asctime)s %(levelname)s %(name)s "
    "[thread_id=%(thread_id)s customer_id=%(customer_id)s] %(message)s"
)
# Rendered in place of a context value that is not bound, such as a line
# logged during startup, outside any turn.
_UNBOUND = "-"

_thread_id: ContextVar[str | None] = ContextVar("log_thread_id", default=None)
_customer_id: ContextVar[int | None] = ContextVar("log_customer_id", default=None)


def configure_logging(config: LoggingConfig) -> None:
    """Install the context-aware stderr handler on the root logger.

    Safe to call more than once: the handler is added only if the root logger
    does not already have one, while the levels are applied on every call.

    Args:
        config: Root log level and the third-party loggers held at
            ``WARNING``.

    """
    root = logging.getLogger()
    if not any(isinstance(h, _ContextStderrHandler) for h in root.handlers):
        handler = _ContextStderrHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        handler.addFilter(LogContextFilter())
        root.addHandler(handler)
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

    Attached to the handler rather than to a logger, so records from every
    logger that reaches the root, third-party ones included, get both fields
    and the format string never meets a record without them.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Copy the current log context onto `record`.

        Args:
            record: The record being handled.

        Returns:
            bool: Always True; this filter annotates records and never drops
            one.

        """
        customer_id = _customer_id.get()
        record.__dict__["thread_id"] = _thread_id.get() or _UNBOUND
        record.__dict__["customer_id"] = (
            _UNBOUND if customer_id is None else customer_id
        )
        return True


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
