"""Tests for the Redis client `InfoRagResources` builds.

No Redis is needed: building a client and its pool opens no connection.
"""

# ruff: noqa: S101
from __future__ import annotations

import asyncio

from redis.asyncio import BlockingConnectionPool

from blue_horizon.agents.information.resources import InfoRagResources
from blue_horizon.config import InfoRedisConfig

_MAX_CONNECTIONS = 7
_POOL_TIMEOUT_S = 3.5


def _redis_config() -> InfoRedisConfig:
    """Build a Redis config section with distinctive pool settings.

    Returns:
        InfoRedisConfig: The config section.

    """
    return InfoRedisConfig(
        connect_timeout_s=2.0,
        socket_timeout_s=4.0,
        health_check_interval_s=30,
        retry_max_retries=2,
        retry_backoff_base_s=0.1,
        retry_backoff_max_s=1.0,
        max_connections=_MAX_CONNECTIONS,
        pool_timeout_s=_POOL_TIMEOUT_S,
    )


class TestBuildRedisClient:
    """The client's pool is capped and blocks when the cap is reached."""

    def test_pool_is_blocking_and_capped_by_config(self) -> None:
        """A busy pool makes a command wait rather than open a connection."""
        client = InfoRagResources._build_redis_client(  # noqa: SLF001
            "redis://localhost:6379/0", _redis_config(),
        )
        pool = client.connection_pool
        try:
            assert isinstance(pool, BlockingConnectionPool)
            assert pool.max_connections == _MAX_CONNECTIONS
            assert pool.timeout == _POOL_TIMEOUT_S
            assert pool.connection_kwargs["decode_responses"] is True
        finally:
            asyncio.run(client.aclose())

    def test_client_owns_its_pool(self) -> None:
        """Closing the client also closes the pool, so shutdown leaks nothing."""
        client = InfoRagResources._build_redis_client(  # noqa: SLF001
            "redis://localhost:6379/0", _redis_config(),
        )
        assert client.auto_close_connection_pool is True
        asyncio.run(client.aclose())
