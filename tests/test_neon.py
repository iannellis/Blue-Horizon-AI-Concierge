"""Tests for blue_horizon/neon.py.

All async functions are exercised via asyncio.run() so no pytest-asyncio
dependency is required.
"""

# ruff: noqa: S101

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from blue_horizon.config import NeonConfig
from blue_horizon.neon import (
    _find_branch,
    _restore_branch,
    _wait_for_operation,
    _wait_for_operations,
    reset_branch,
)

_PROJECT_ID = "test-project"
_BRANCH_ID = "br-abc123"
_PARENT_ID = "br-parent"
_BRANCH_NAME = "Working"
_OPERATION_ID = "op-123"
_LOCK_RETRY_ATTEMPTS_TWO = 2
_LOCK_RETRY_ATTEMPTS_THREE = 3
_LOCK_RETRY_DELAY = 1.0
_OPERATION_POLL_INTERVAL = 0.0
_OPERATION_POLL_TIMEOUT = 30.0


# ---------------------------------------------------------------------------
# _find_branch
# ---------------------------------------------------------------------------


class TestFindBranch:
    """_find_branch resolves a branch name to its ID and parent ID."""

    def _make_client(self, branches: list[dict]) -> MagicMock:
        """Build a mock AsyncClient whose GET returns the given branches list.

        Args:
            branches: List of branch dicts to include in the API response.

        Returns:
            MagicMock configured to act as an authenticated AsyncClient.

        """
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"branches": branches}
        client = MagicMock()
        client.get = AsyncMock(return_value=mock_response)
        return client

    def test_returns_branch_id_and_parent_id(self) -> None:
        """The matching branch's ID and parent ID are returned as a tuple."""
        client = self._make_client([
            {"id": _BRANCH_ID, "name": _BRANCH_NAME, "parent_id": _PARENT_ID},
        ])
        branch_id, parent_id = asyncio.run(
            _find_branch(client, _PROJECT_ID, _BRANCH_NAME),
        )
        assert branch_id == _BRANCH_ID
        assert parent_id == _PARENT_ID

    def test_ignores_other_branches(self) -> None:
        """Only the branch whose name matches is returned."""
        client = self._make_client([
            {"id": "br-other", "name": "production", "parent_id": None},
            {"id": _BRANCH_ID, "name": _BRANCH_NAME, "parent_id": _PARENT_ID},
        ])
        branch_id, _ = asyncio.run(_find_branch(client, _PROJECT_ID, _BRANCH_NAME))
        assert branch_id == _BRANCH_ID

    def test_raises_when_branch_not_found(self) -> None:
        """RuntimeError is raised when no branch matches the given name."""
        client = self._make_client([
            {"id": "br-other", "name": "production", "parent_id": None},
        ])
        with pytest.raises(RuntimeError, match=_BRANCH_NAME):
            asyncio.run(_find_branch(client, _PROJECT_ID, _BRANCH_NAME))

    def test_raises_when_branch_has_no_parent(self) -> None:
        """RuntimeError is raised when the matched branch has no parent_id."""
        client = self._make_client([
            {"id": _BRANCH_ID, "name": _BRANCH_NAME, "parent_id": None},
        ])
        with pytest.raises(RuntimeError, match="no parent"):
            asyncio.run(_find_branch(client, _PROJECT_ID, _BRANCH_NAME))


# ---------------------------------------------------------------------------
# _restore_branch
# ---------------------------------------------------------------------------


class TestRestoreBranch:
    """_restore_branch calls the Neon restore endpoint with lock-retry logic."""

    def _make_client(
        self, status_codes: list[int], *, operations: list[dict] | None = None,
    ) -> MagicMock:
        """Build a mock AsyncClient whose POST returns responses in sequence.

        Args:
            status_codes: HTTP status codes to return on successive POST calls.
                A 423 response does not raise; a non-2xx non-423 response
                raises httpx.HTTPStatusError via raise_for_status.
            operations: The ``operations`` list every successful response's
                body reports. Defaults to a single already-finished operation
                so tests that don't care about polling don't hang.

        Returns:
            MagicMock configured to act as an authenticated AsyncClient.

        """
        if operations is None:
            operations = [{"id": _OPERATION_ID}]
        responses = []
        for code in status_codes:
            mock_response = MagicMock()
            mock_response.status_code = code
            mock_response.json.return_value = {"operations": operations}
            if code >= 400:  # noqa: PLR2004
                mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                    f"HTTP {code}",
                    request=MagicMock(),
                    response=mock_response,
                )
            else:
                mock_response.raise_for_status.return_value = None
            responses.append(mock_response)
        client = MagicMock()
        client.post = AsyncMock(side_effect=responses)
        finished_response = MagicMock()
        finished_response.raise_for_status.return_value = None
        finished_response.json.return_value = {"operation": {"status": "finished"}}
        client.get = AsyncMock(return_value=finished_response)
        return client

    def test_succeeds_on_first_attempt(self) -> None:
        """A 200 response on the first attempt returns without retrying."""
        client = self._make_client([200])
        asyncio.run(
            _restore_branch(
                client,
                _PROJECT_ID,
                _BRANCH_ID,
                _PARENT_ID,
                lock_retry_attempts=3,
                lock_retry_delay_s=0.0,
                operation_poll_interval_s=_OPERATION_POLL_INTERVAL,
                operation_poll_timeout_s=_OPERATION_POLL_TIMEOUT,
            ),
        )
        assert client.post.call_count == 1

    def test_retries_on_423_then_succeeds(self) -> None:
        """A 423 on the first attempt triggers a retry that succeeds on 200."""
        client = self._make_client([423, 200])
        with patch("blue_horizon.neon.asyncio.sleep", new_callable=AsyncMock):
            asyncio.run(
                _restore_branch(
                    client,
                    _PROJECT_ID,
                    _BRANCH_ID,
                    _PARENT_ID,
                    lock_retry_attempts=3,
                    lock_retry_delay_s=0.0,
                    operation_poll_interval_s=_OPERATION_POLL_INTERVAL,
                    operation_poll_timeout_s=_OPERATION_POLL_TIMEOUT,
                ),
            )
        assert client.post.call_count == _LOCK_RETRY_ATTEMPTS_TWO

    def test_raises_after_all_retries_exhausted(self) -> None:
        """HTTPStatusError is raised when every attempt returns 423."""
        client = self._make_client([423, 423])
        with (
            patch("blue_horizon.neon.asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(httpx.HTTPStatusError),
        ):
            asyncio.run(
                _restore_branch(
                    client,
                    _PROJECT_ID,
                    _BRANCH_ID,
                    _PARENT_ID,
                    lock_retry_attempts=_LOCK_RETRY_ATTEMPTS_TWO,
                    lock_retry_delay_s=0.0,
                    operation_poll_interval_s=_OPERATION_POLL_INTERVAL,
                    operation_poll_timeout_s=_OPERATION_POLL_TIMEOUT,
                ),
            )
        assert client.post.call_count == _LOCK_RETRY_ATTEMPTS_TWO

    def test_sleeps_between_retries(self) -> None:
        """asyncio.sleep is called with lock_retry_delay_s between 423 retries."""
        client = self._make_client([423, 200])
        sleep_mock = AsyncMock()
        with patch("blue_horizon.neon.asyncio.sleep", sleep_mock):
            asyncio.run(
                _restore_branch(
                    client,
                    _PROJECT_ID,
                    _BRANCH_ID,
                    _PARENT_ID,
                    lock_retry_attempts=3,
                    lock_retry_delay_s=5.0,
                    operation_poll_interval_s=_OPERATION_POLL_INTERVAL,
                    operation_poll_timeout_s=_OPERATION_POLL_TIMEOUT,
                ),
            )
        sleep_mock.assert_called_once_with(5.0)

    def test_waits_for_returned_operations_before_returning(self) -> None:
        """The restore doesn't return until the reported operation finishes."""
        client = self._make_client([200], operations=[{"id": _OPERATION_ID}])
        asyncio.run(
            _restore_branch(
                client,
                _PROJECT_ID,
                _BRANCH_ID,
                _PARENT_ID,
                lock_retry_attempts=3,
                lock_retry_delay_s=0.0,
                operation_poll_interval_s=_OPERATION_POLL_INTERVAL,
                operation_poll_timeout_s=_OPERATION_POLL_TIMEOUT,
            ),
        )
        client.get.assert_called_once_with(
            f"https://console.neon.tech/api/v2/projects/{_PROJECT_ID}"
            f"/operations/{_OPERATION_ID}",
        )


# ---------------------------------------------------------------------------
# _wait_for_operation / _wait_for_operations
# ---------------------------------------------------------------------------


class TestWaitForOperation:
    """_wait_for_operation polls a single operation until it is terminal."""

    def _make_client(self, statuses: list[str]) -> MagicMock:
        """Build a mock AsyncClient whose GET reports statuses in sequence.

        Args:
            statuses: Operation statuses to return on successive GET calls.

        Returns:
            MagicMock configured to act as an authenticated AsyncClient.

        """
        responses = []
        for status in statuses:
            mock_response = MagicMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {"operation": {"status": status}}
            responses.append(mock_response)
        client = MagicMock()
        client.get = AsyncMock(side_effect=responses)
        return client

    def test_returns_once_finished(self) -> None:
        """Polling stops as soon as the operation reports 'finished'."""
        client = self._make_client(["running", "finished"])
        with patch("blue_horizon.neon.asyncio.sleep", new_callable=AsyncMock):
            asyncio.run(
                _wait_for_operation(
                    client,
                    _PROJECT_ID,
                    _OPERATION_ID,
                    poll_interval_s=0.0,
                    timeout_s=_OPERATION_POLL_TIMEOUT,
                ),
            )
        assert client.get.call_count == _LOCK_RETRY_ATTEMPTS_TWO

    def test_raises_on_failed_status(self) -> None:
        """A 'failed' status raises RuntimeError instead of continuing to poll."""
        client = self._make_client(["failed"])
        with pytest.raises(RuntimeError, match="failed"):
            asyncio.run(
                _wait_for_operation(
                    client,
                    _PROJECT_ID,
                    _OPERATION_ID,
                    poll_interval_s=0.0,
                    timeout_s=_OPERATION_POLL_TIMEOUT,
                ),
            )

    def test_raises_on_timeout(self) -> None:
        """RuntimeError is raised once the deadline passes without finishing."""
        client = self._make_client(["running", "running", "running"])
        with pytest.raises(RuntimeError, match="Timed out"):
            asyncio.run(
                _wait_for_operation(
                    client,
                    _PROJECT_ID,
                    _OPERATION_ID,
                    poll_interval_s=0.0,
                    timeout_s=0.0,
                ),
            )


class TestWaitForOperations:
    """_wait_for_operations waits on every operation, skipping id-less ones."""

    def test_waits_for_each_operation_in_order(self) -> None:
        """Every operation with an id is passed to _wait_for_operation."""
        wait_mock = AsyncMock()
        client = MagicMock()
        with patch("blue_horizon.neon._wait_for_operation", wait_mock):
            asyncio.run(
                _wait_for_operations(
                    client,
                    _PROJECT_ID,
                    [{"id": "op-1"}, {}, {"id": "op-2"}],
                    poll_interval_s=0.0,
                    timeout_s=_OPERATION_POLL_TIMEOUT,
                ),
            )
        assert wait_mock.call_count == _LOCK_RETRY_ATTEMPTS_TWO
        polled_ids = [call.args[2] for call in wait_mock.call_args_list]
        assert polled_ids == ["op-1", "op-2"]


# ---------------------------------------------------------------------------
# reset_branch
# ---------------------------------------------------------------------------


class TestResetBranch:
    """reset_branch orchestrates find + restore against a real AsyncClient."""

    def test_invokes_find_and_restore_with_correct_args(self) -> None:
        """_find_branch and _restore_branch are called with the right arguments."""
        find_mock = AsyncMock(return_value=(_BRANCH_ID, _PARENT_ID))
        restore_mock = AsyncMock()
        with (
            patch("blue_horizon.neon._find_branch", find_mock),
            patch("blue_horizon.neon._restore_branch", restore_mock),
            patch("blue_horizon.neon.httpx.AsyncClient") as mock_client_cls,
        ):
            mock_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock(),
            )
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            neon_cfg = NeonConfig(
                project_id=_PROJECT_ID,
                branch_name=_BRANCH_NAME,
                lock_retry_attempts=_LOCK_RETRY_ATTEMPTS_THREE,
                lock_retry_delay_s=_LOCK_RETRY_DELAY,
                operation_poll_interval_s=_OPERATION_POLL_INTERVAL,
                operation_poll_timeout_s=_OPERATION_POLL_TIMEOUT,
            )
            asyncio.run(reset_branch(neon_cfg, api_key="api-key"))
        find_mock.assert_called_once()
        assert find_mock.call_args.args[1] == _PROJECT_ID
        assert find_mock.call_args.args[2] == _BRANCH_NAME
        restore_mock.assert_called_once()
        kwargs = restore_mock.call_args.kwargs
        assert kwargs["lock_retry_attempts"] == _LOCK_RETRY_ATTEMPTS_THREE
        assert kwargs["lock_retry_delay_s"] == _LOCK_RETRY_DELAY
        assert kwargs["operation_poll_interval_s"] == _OPERATION_POLL_INTERVAL
        assert kwargs["operation_poll_timeout_s"] == _OPERATION_POLL_TIMEOUT
