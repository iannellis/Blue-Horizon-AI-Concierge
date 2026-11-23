"""Tests for `InfoRagResources.startup_check`: Redis ping and catalog lookup.

No Redis is needed: the resources object is built without `__init__` and
given a mocked async client.
"""

# ruff: noqa: S101
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from redis.exceptions import AuthenticationError as RedisAuthenticationError
from redis.exceptions import ConnectionError as RedisConnectionError

from blue_horizon.agents.exceptions import ConfigurationError, OperationalError
from blue_horizon.agents.information.resources import InfoRagResources


def _resources_whose_ping_raises(exc: BaseException) -> InfoRagResources:
    """Build an `InfoRagResources` whose Redis ping raises `exc`.

    Args:
        exc: The exception the ping should raise.

    Returns:
        InfoRagResources: An instance with only `redis_async` set.

    """
    resources = InfoRagResources.__new__(InfoRagResources)
    client = MagicMock()
    client.ping = AsyncMock(side_effect=exc)
    resources.redis_async = client
    return resources


class TestStartupCheckRedisPing:
    """A rejected password is permanent; an unreachable Redis is transient."""

    def test_authentication_error_raises_configuration_error(self) -> None:
        """Wrong credentials are not reported as the system starting up."""
        resources = _resources_whose_ping_raises(
            RedisAuthenticationError("invalid username-password pair"),
        )
        with pytest.raises(ConfigurationError) as exc_info:
            asyncio.run(resources.startup_check())
        assert isinstance(exc_info.value.__cause__, RedisAuthenticationError)

    def test_connection_error_raises_operational_error(self) -> None:
        """An unreachable Redis stays transient, so startup keeps saying so."""
        resources = _resources_whose_ping_raises(
            RedisConnectionError("Error 10061 connecting to redis"),
        )
        with pytest.raises(OperationalError):
            asyncio.run(resources.startup_check())


class TestFetchCatalogCategories:
    """Overview answers depend on reading every category name from Redis."""

    def test_merges_dedupes_and_sorts_both_indexes(self) -> None:
        """Rows from both aggregations are merged; a row without a value is skipped."""
        resources = InfoRagResources.__new__(InfoRagResources)
        client = MagicMock()
        client.execute_command = AsyncMock(
            side_effect=[
                [2, ["category", "Spa Services"], ["category", "Room Service"]],
                [2, ["service_type", "Spa Services"], ["service_type"]],
            ],
        )
        resources.redis_async = client
        names = asyncio.run(resources._fetch_catalog_categories())  # noqa: SLF001
        assert names == ["Room Service", "Spa Services"]

    def test_aggregation_failure_raises_operational_error(self) -> None:
        """A failed lookup keeps startup retrying rather than serving a gap."""
        resources = InfoRagResources.__new__(InfoRagResources)
        client = MagicMock()
        client.execute_command = AsyncMock(side_effect=RedisConnectionError("down"))
        resources.redis_async = client
        with pytest.raises(OperationalError):
            asyncio.run(resources._fetch_catalog_categories())  # noqa: SLF001
