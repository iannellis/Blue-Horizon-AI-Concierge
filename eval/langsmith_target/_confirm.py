"""Auto-confirm helper: the eval/stress harnesses' stand-in for a Confirm click.

Neither harness drives a browser, so neither can click the UI's Confirm
button. This module performs the same write the button would, through the
same `ProposalStore.confirm()` code path `POST /v1/booking/confirm` uses --
never a shortcut that bypasses it -- and injects the outcome into the turn's
`EvalCaptureCallback`, since `confirm()` is not a LangChain-tracked runnable
and `on_tool_end` never observes it on its own.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from blue_horizon.agents.booking import proposals as proposals_module
from blue_horizon.agents.booking import write_ops
from blue_horizon.agents.booking.receipts import receipt_message, serialize_write_result

if TYPE_CHECKING:
    from blue_horizon.agents.orchestration import OrchestrationManager
    from eval.langsmith_target._callback import EvalCaptureCallback


async def auto_confirm_pending_proposal(
    orchestration: OrchestrationManager,
    *,
    thread_id: str,
    customer_id: int,
    callback: EvalCaptureCallback,
) -> None:
    """Confirm any proposal left pending after a turn, then record the outcome.

    A no-op when the turn made no `propose_*` call. On success, also appends
    an app-authored receipt to the thread's state via
    `OrchestrationManager.append_assistant_message`, exactly as
    `POST /v1/booking/confirm` does, so the agent's next turn knows what
    happened -- the model is never the one reporting it, in eval either.

    Args:
        orchestration: Shared orchestration manager reached directly by the
            harness, with no FastAPI app involved.
        thread_id: Conversation thread whose pending proposal, if any, should
            be confirmed.
        customer_id: Guest the harness is impersonating for this case.
        callback: The turn's capture callback, mutated in place with the
            confirm outcome.

    """
    resources = orchestration.get_booking_resources()
    proposal = resources.proposals.get_pending_for_thread(thread_id)
    if proposal is None:
        return

    try:
        outcome = await resources.proposals.confirm(
            proposal_id=proposal.proposal_id,
            customer_id=customer_id,
            write_pool=resources.get_write_pool(),
        )
    except (proposals_module.ProposalError, write_ops.BookingWriteError) as exc:
        callback.capture_confirm_error(action=proposal.action, error=str(exc))
        return

    receipt_text = receipt_message(outcome)
    callback.capture_confirm_result(
        action=outcome.proposal.action,
        already_confirmed=outcome.already_confirmed,
        result=serialize_write_result(outcome.result),
        receipt_text=receipt_text,
    )
    await orchestration.append_assistant_message(
        thread_id=thread_id, text=receipt_text,
    )
