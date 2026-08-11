"""Tests for failure handling and cleanup in eval/stress/runner.py."""

# ruff: noqa: S101

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from eval.stress import runner


class _FakeOrchestration:
    """Test double for ``OrchestrationManager`` cleanup behavior."""

    def __init__(self) -> None:
        """Initialize cleanup tracking."""
        self.stop_called = False

    async def stop(self) -> None:
        """Record that cleanup was requested."""
        self.stop_called = True


class _FakePool:
    """Test double for the stress runner's async DB pool."""

    def __init__(self) -> None:
        """Initialize cleanup tracking."""
        self.close_called = False

    async def close(self) -> None:
        """Record that pool cleanup was requested."""
        self.close_called = True


def _make_cfg(tmp_path: Path) -> SimpleNamespace:
    """Build a minimal stress config for runner tests.

    Args:
        tmp_path: Temporary directory used for artifact paths.

    Returns:
        Minimal config namespace with the fields used by ``run_stress()``.

    """
    return SimpleNamespace(
        users=2,
        ops_per_user=3,
        max_concurrency=2,
        output_dir=tmp_path,
        log_dir=tmp_path,
        db_url="postgres://demo",
        ro_db_url="postgres://demo-ro",
        pool_max=2,
    )


def test_run_stress_writes_failure_artifacts_and_closes_resources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Failures still write artifacts and close orchestration and pool resources.

    Args:
        monkeypatch: Pytest fixture for replacing runner collaborators.
        tmp_path: Temporary directory used for artifact paths.

    """
    cfg = _make_cfg(tmp_path)
    fake_orchestration = _FakeOrchestration()
    fake_pool = _FakePool()
    captured: dict[str, Any] = {}

    def _fake_configure_logging(*_args: object, **_kwargs: object) -> None:
        """Replace runner logging setup during the test."""

    def _fake_override_db_url(_db_url: str, _ro_db_url: str) -> None:
        """Replace environment mutation during the test."""

    monkeypatch.setattr(runner, "_load_config", lambda: cfg)
    monkeypatch.setattr(runner, "_configure_logging", _fake_configure_logging)
    monkeypatch.setattr(runner, "_override_db_url", _fake_override_db_url)
    monkeypatch.setattr(
        runner,
        "load_stress_config",
        lambda: SimpleNamespace(
            orchestration=SimpleNamespace(ready_timeout_s=1.0),
            neon=SimpleNamespace(branch_name="demo", project_id="proj"),
            neon_api_key="secret",
        ),
    )

    async def _fake_start_orchestration(
        *,
        db_url: str,
        ro_db_url: str,
        ready_timeout_s: float,
    ) -> _FakeOrchestration:
        """Return a fake orchestration manager.

        Args:
            db_url: Read-write database URL forwarded by the runner.
            ro_db_url: Read-only database URL forwarded by the runner.
            ready_timeout_s: Ready timeout forwarded by the runner.

        Returns:
            Fake orchestration manager.

        """
        _ = db_url, ro_db_url, ready_timeout_s
        return fake_orchestration

    async def _fake_open_schema_pool(
        db_url: str,
        *,
        max_size: int,
    ) -> _FakePool:
        """Return a fake pool instance.

        Args:
            db_url: Database URL forwarded by the runner.
            max_size: Requested pool size.

        Returns:
            Fake pool instance.

        """
        _ = db_url, max_size
        return fake_pool

    async def _fake_init_branch_and_targets(
        pool: _FakePool,
        cfg_obj: SimpleNamespace,
        neon_cfg: SimpleNamespace,
        *,
        api_key: str | None,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        """Return deterministic targets for the failure-path test.

        Args:
            pool: Fake pool instance.
            cfg_obj: Stress config namespace.
            neon_cfg: Neon config namespace.
            api_key: Fake API key.

        Returns:
            Single target and no hot targets.

        """
        _ = pool, cfg_obj, neon_cfg, api_key
        targets = [
            {
                "room_number": 101,
                "check_in": "2025-01-01",
                "check_out": "2025-01-03",
            },
        ]
        return targets, []

    async def _fake_run_workload(
        orchestration: _FakeOrchestration,
        cfg_obj: SimpleNamespace,
        targets: list[dict[str, object]],
        hot_targets: list[dict[str, object]],
        *,
        run_id: str,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        """Raise a failure after setup succeeds.

        Args:
            orchestration: Fake orchestration manager.
            cfg_obj: Stress config namespace.
            targets: Target list.
            hot_targets: Hot target list.
            run_id: Stress run identifier.

        Raises:
            RuntimeError: Always raised for the test.

        """
        _ = orchestration, cfg_obj, targets, hot_targets, run_id
        msg = "boom"
        raise RuntimeError(msg)

    def _fake_build_summary(*_args: object, **_kwargs: object) -> dict[str, object]:
        """Return a base summary that the runner should mutate on failure.

        Returns:
            Base summary payload.

        """
        return {"status": "PASS"}

    def _fake_write_artifacts(
        cfg_obj: SimpleNamespace,
        *,
        run_id: str,
        op_logs: list[dict[str, object]],
        user_logs: list[dict[str, object]],
        summary: dict[str, object],
    ) -> None:
        """Capture artifact arguments for assertions.

        Args:
            cfg_obj: Stress config namespace.
            run_id: Stress run identifier.
            op_logs: Operation logs collected so far.
            user_logs: User logs collected so far.
            summary: Final summary payload.

        """
        captured["cfg"] = cfg_obj
        captured["run_id"] = run_id
        captured["op_logs"] = op_logs
        captured["user_logs"] = user_logs
        captured["summary"] = summary

    monkeypatch.setattr(runner, "_start_orchestration", _fake_start_orchestration)
    monkeypatch.setattr(runner, "open_schema_pool", _fake_open_schema_pool)
    monkeypatch.setattr(
        runner,
        "_init_branch_and_targets",
        _fake_init_branch_and_targets,
    )
    monkeypatch.setattr(runner, "_run_workload", _fake_run_workload)
    monkeypatch.setattr(runner, "_build_summary", _fake_build_summary)
    monkeypatch.setattr(runner, "_write_artifacts", _fake_write_artifacts)

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(runner.run_stress())

    assert fake_orchestration.stop_called is True
    assert fake_pool.close_called is True
    assert captured["summary"]["status"] == "FAIL"
    assert captured["summary"]["error"] == "RuntimeError: boom"
    assert captured["run_id"].startswith("stress_")
