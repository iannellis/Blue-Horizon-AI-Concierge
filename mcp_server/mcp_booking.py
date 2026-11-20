"""Provide an MCP server to search rooms in the Blue Horizon hotel."""

import asyncio
import logging
import sys
from collections.abc import AsyncGenerator, Coroutine
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import psycopg
from fastmcp import Context, FastMCP
from fastmcp.server.middleware.rate_limiting import RateLimitingMiddleware
from psycopg_pool import AsyncConnectionPool

# Prefect Horizon neither installs blue_horizon nor puts the repo root on the path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blue_horizon.agents.booking import (
    SEARCH_TOOL_DESCRIPTION,
    assert_pool_is_read_only,
    build_search_args_model,
    check_credentials,
    fetch_rooms_metadata,
    search_rooms_with_retry,
)
from blue_horizon.config import load_app_config
from blue_horizon.logging_setup import configure_logging, log_context, stop_log_shipping

# Run every coroutine on a selector loop: psycopg's async mode can't use the
# Proactor loop that Windows uses by default.
_LOOP = asyncio.SelectorEventLoop

_RO_LABEL = "read-only (bh_agent_ro)"

# The `error` text of a failed search. Written for a client that brings its own
# prompt: the booking agent's DATABASE_UNAVAILABLE marker means nothing to one.
# `error_kind` says whether the database was unreachable or the search hit a bug.
_SEARCH_FAILURE_MESSAGE = (
    "The room search failed on the server, not because of the search arguments. "
    "Do not change the arguments; try again later."
)

def _run_startup[T](coro: Coroutine[Any, Any, T]) -> T:
    """Run a startup coroutine to completion on its own event loop.

    `fastmcp run` and Prefect Horizon import this module from inside a running
    event loop, where `asyncio.run` raises, so the coroutine runs on a worker
    thread instead.

    Args:
        coro: The coroutine to run.

    Returns:
        T: The coroutine's result.

    Raises:
        Exception: Whatever the coroutine raises.

    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coro, loop_factory=_LOOP).result()

cfg = load_app_config()
sql_ro_url = cfg.pgsql_ro_db_url

configure_logging(
    cfg.logging,
    axiom_api_key=cfg.axiom_api_key,
    axiom_dataset=cfg.axiom_dataset,
    service_name="blue-horizon-mcp",
)

logger = logging.getLogger(__name__)

_run_startup(
    check_credentials(sql_ro_url, _RO_LABEL, timeout_s=cfg.booking.db.pool.timeout_s),
)

rooms_meta = _run_startup(fetch_rooms_metadata(sql_ro_url))

async def get_rooms_count(ro_db_url: str) -> int:
    """Get the number of rooms in the hotel."""
    async with (
            await psycopg.AsyncConnection.connect(ro_db_url) as conn,
            conn.cursor() as cur,
        ):
            await cur.execute("SET TRANSACTION READ ONLY")
            await cur.execute("SELECT COUNT(*) FROM rooms")
            room_count = await cur.fetchone()
    if room_count is None or None in room_count:
        msg = "rooms is empty"
        raise ValueError(msg)
    return room_count[0]

room_count = _run_startup(get_rooms_count(sql_ro_url))

SearchArgs = build_search_args_model(
    meta=rooms_meta,
    max_room_numbers=room_count,
    default_results=cfg.booking.agent.default_search_results,
    max_results=room_count,
)

@asynccontextmanager
async def lifespan(
    _server: FastMCP[Any],
) -> AsyncGenerator[dict[str, AsyncConnectionPool[Any]]]:
    """Startup and shutdown the connection pool when the server starts and stops."""
    pool_cfg = cfg.booking.db.pool
    try:
        async with AsyncConnectionPool(
            conninfo=sql_ro_url,
            min_size=pool_cfg.min_size,
            max_size=pool_cfg.max_size,
            timeout=pool_cfg.timeout_s,
            max_idle=pool_cfg.max_idle_s,
            reconnect_timeout=pool_cfg.reconnect_timeout_s,
            check=AsyncConnectionPool.check_connection,
            open=False,
        ) as pool:
            await pool.wait(timeout=10)
            await assert_pool_is_read_only(pool)
            logger.info("MCP server ready: %d rooms", room_count)
            yield {"pool": pool}
    finally:
        stop_log_shipping()

mcp = FastMCP("blue-horizon", lifespan=lifespan)
mcp.add_middleware(
    RateLimitingMiddleware(
        max_requests_per_second=cfg.mcp_server.rate_limit.max_requests_per_second,
        burst_capacity=cfg.mcp_server.rate_limit.burst_capacity,
        global_limit=True,
    ),
)

@mcp.tool(description=SEARCH_TOOL_DESCRIPTION)
async def search_rooms(
    args: SearchArgs, # pyright: ignore[reportInvalidTypeForm]
    ctx: Context,
) -> dict[str, Any]:
    """Search for room information in the Blue Horizon hotel."""
    db_cfg = cfg.booking.db
    with log_context(thread_id=f"mcp:{ctx.session_id}"):
        return await search_rooms_with_retry(
            ctx.lifespan_context["pool"],
            args,
            retry=db_cfg.retry,
            checkout_timeout_s=db_cfg.pool.timeout_s,
            failure_message=_SEARCH_FAILURE_MESSAGE,
        )

# Local runs only: Prefect Horizon imports `mcp` and ignores this block.
if __name__ == "__main__":
    mcp.run(transport="http", host="127.0.0.1", port=7860)
