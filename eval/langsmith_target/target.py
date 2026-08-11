"""LangSmith evaluation target and shared orchestration manager.

Provides ``run_example()`` (the async target for LangSmith evaluate/aevaluate)
and ``ensure_orchestration_ready()`` (the shared singleton initializer).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from blue_horizon.agents.orchestration import OrchestrationManager
from eval.langsmith_target._callback import EvalCaptureCallback
from eval.langsmith_target._confirm import auto_confirm_pending_proposal
from eval.langsmith_target._orchestration_utils import wait_for_orchestration_ready
from eval.langsmith_target._text_utils import _extract_assistant_text_from_result

if TYPE_CHECKING:
    from eval.config import EvalConfig

logger = logging.getLogger(__name__)

_ORCHESTRATION_LOCK = asyncio.Lock()
_ORCHESTRATION: OrchestrationManager | None = None

_ROUTE_KEY = "route"  # key from orchestration.py

# Only the first 25 customers are seeded (see booking_pgsql.py::_CUSTOMER_COUNT).
_SEEDED_CUSTOMER_COUNT = 25

# Case tag that opts a case out of post-turn auto-confirmation, so a
# `propose_*` call is left pending rather than always committed -- the only
# way to represent abandonment or supersession without a browser.
_NO_AUTO_CONFIRM_TAG = "no_auto_confirm"


def _customer_id_for_case(case_id: str) -> int:
    """Deterministically assign one of the 25 seeded guests to a case.

    Deterministic (not random) so a re-run of the same case exercises the
    same guest, and stable across processes without any shared counter.

    Args:
        case_id: Dataset case identifier.

    Returns:
        int: A customer_id in `[1, _SEEDED_CUSTOMER_COUNT]`.

    """
    digest = hashlib.sha256(case_id.encode("utf-8")).hexdigest()
    return (int(digest, 16) % _SEEDED_CUSTOMER_COUNT) + 1


async def run_example(
    example: dict[str, Any],
    *,
    cfg: EvalConfig,
) -> dict[str, Any]:
    """Run a single LangSmith example through the orchestration pipeline.

    Each case is deterministically assigned one of the 25 seeded guests (see
    `_customer_id_for_case`), and after every turn any proposal left pending
    is auto-confirmed -- the harness's stand-in for a human clicking Confirm
    -- unless the case carries the `"no_auto_confirm"` tag, which leaves a
    turn's proposal pending so a case can model abandonment or supersession
    instead of a completed write.

    Args:
        example: LangSmith dataset example input containing case_id, turns, and tags.
        cfg: Evaluation configuration for orchestration setup.

    Returns:
        Dict containing turn outputs and case tags.

    Raises:
        KeyError: If required keys are missing from the example.

    """
    case_id = str(example["case_id"])
    turns = example["turns"]
    case_tags_raw = example.get("tags") or []
    case_tags = list(case_tags_raw)
    auto_confirm = _NO_AUTO_CONFIRM_TAG not in case_tags

    orchestration_mgr = await ensure_orchestration_ready(cfg)

    thread_id = uuid4().hex
    customer_id = _customer_id_for_case(case_id)

    turn_outputs: list[dict[str, Any]] = []

    for turn_index, turn in enumerate(turns):
        user_text = str(turn["user"])
        callback = EvalCaptureCallback()
        tags = ["eval", f"case:{case_id}", f"turn:{turn_index}", *case_tags]
        metadata = {
            "case_id": case_id,
            "turn_index": turn_index,
        }

        _t0 = asyncio.get_running_loop().time()
        result = await orchestration_mgr.ainvoke(
            thread_id=thread_id,
            user_text=user_text,
            customer_id=customer_id,
            callbacks=[callback],
            tags=tags,
            metadata=metadata,
        )
        if auto_confirm:
            # The harness's stand-in for a human clicking Confirm -- see
            # eval.langsmith_target._confirm for why this cannot be skipped
            # in favor of reading proposals.py state directly.
            await auto_confirm_pending_proposal(
                orchestration_mgr,
                thread_id=thread_id,
                customer_id=customer_id,
                callback=callback,
            )
        latency_ms = (asyncio.get_running_loop().time() - _t0) * 1000.0

        assistant_text = _extract_assistant_text_from_result(result)

        route_pred = callback.route_pred
        if route_pred is None:
            fallback_route = result.get(_ROUTE_KEY)
            if isinstance(fallback_route, str):
                route_pred = fallback_route

        turn_outputs.append(
            {
                "assistant_text": assistant_text,
                "route_pred": route_pred,
                "tool_summary": callback.tool_summary,
                "contexts_used": callback.contexts_used,
                "latency_ms": latency_ms,
                "confirm_receipt_text": callback.confirm_receipt_text,
            },
        )

    return {
        "turn_outputs": turn_outputs,
        "case_tags": case_tags,
    }


async def ensure_orchestration_ready(cfg: EvalConfig) -> OrchestrationManager:
    """Ensure the shared orchestration manager is initialized and ready.

    The manager is constructed with ``pgsql_rw_eval_db_url`` and
    ``pgsql_ro_eval_db_url`` from the eval config so that the booking agent's
    write pool and evaluator pool connections use the same read-write
    database that ``run_sql`` reads from via the read-only role. When either
    override is not set, the corresponding value is ``None`` and the agent
    falls back to ``PGSQL_RW_DB_URL`` / ``PGSQL_RO_DB_URL``.

    Args:
        cfg: Evaluation configuration with orchestration settings and DB URLs.

    Returns:
        The shared OrchestrationManager instance.

    Raises:
        RuntimeError: If the orchestration manager does not become ready in time.

    """
    global _ORCHESTRATION  # noqa: PLW0603
    if _ORCHESTRATION is not None and _ORCHESTRATION.is_ready:
        return _ORCHESTRATION

    async with _ORCHESTRATION_LOCK:
        if _ORCHESTRATION is None:
            _ORCHESTRATION = OrchestrationManager(
                pgsql_rw_db_url=cfg.pgsql_rw_eval_db_url,
                pgsql_ro_db_url=cfg.pgsql_ro_eval_db_url,
            )
        if _ORCHESTRATION.is_ready:
            return _ORCHESTRATION
        await _ORCHESTRATION.start()

        timeout_s = cfg.orchestration.ready_timeout_s
        try:
            await wait_for_orchestration_ready(
                _ORCHESTRATION,
                timeout_s=timeout_s,
                manager_name="OrchestrationManager",
            )
        except TimeoutError as exc:
            msg = "OrchestrationManager did not become ready within timeout."
            raise RuntimeError(msg) from exc

    if _ORCHESTRATION is None:
        msg = "OrchestrationManager was not initialized."
        raise RuntimeError(msg)
    return _ORCHESTRATION
