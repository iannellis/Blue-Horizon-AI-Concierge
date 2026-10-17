"""App-authored receipt and refusal text, and JSON serialization of confirms.

Both `blue_horizon.api.app` (the real `/v1/booking/confirm` endpoint) and the
eval/stress harnesses' auto-confirm helper -- their stand-in for a human
clicking Confirm -- share this module, so a guest and a test run are always
scored against exactly the same wording and fields.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from blue_horizon.agents.booking import write_ops

if TYPE_CHECKING:
    from blue_horizon.agents.booking import proposals as proposals_module


def serialize_write_result(result: proposals_module.WriteResult) -> dict[str, Any]:
    """Convert a `write_ops` commit/cancel/modify result into JSON-safe fields.

    Args:
        result: Result returned by confirming a proposal.

    Returns:
        dict[str, Any]: Fields specific to the result's type.

    """
    if isinstance(result, write_ops.CommitResult):
        return {
            "booking_id": result.booking_id,
            "confirmation_number": result.confirmation_number,
            "rooms": [write_ops.serialize_priced_stay(r) for r in result.rooms],
            "total_amount": write_ops.fmt_money(result.total_amount),
        }
    if isinstance(result, write_ops.CancelResult):
        return {
            "booking_id": result.booking_id,
            "refunded_amount": write_ops.fmt_money(result.refunded_amount),
            "fully_cancelled": result.fully_cancelled,
        }
    return {
        "booking_id": result.booking_id,
        "rooms": [write_ops.serialize_priced_stay(r) for r in result.rooms],
        "total_amount": write_ops.fmt_money(result.total_amount),
    }


def receipt_message(outcome: proposals_module.ConfirmOutcome) -> str:
    """Build the app-authored confirmation receipt for a committed proposal.

    This -- not the model -- is what tells the guest what happened, per the
    rule that a success claim must come with a receipt attached.

    Args:
        outcome: Result of confirming a proposal.

    Returns:
        str: App-authored receipt text.

    """
    result = outcome.result
    if isinstance(result, write_ops.CommitResult):
        return (
            f"Booking confirmed. Confirmation number {result.confirmation_number}. "
            f"Total charged: ${write_ops.fmt_money(result.total_amount)}."
        )
    if isinstance(result, write_ops.CancelResult):
        return (
            f"Cancellation confirmed. ${write_ops.fmt_money(result.refunded_amount)} "
            "refunded."
        )
    return (
        "Modification confirmed. "
        f"New total: ${write_ops.fmt_money(result.total_amount)}."
    )


def refusal_message(exc: write_ops.BookingWriteError) -> str:
    """Build the app-authored note for a confirm the write refused.

    A refused confirm retires its proposal, so without this note the thread's
    last word is the model's own "please review and confirm", and on the
    guest's next turn the model tells them to press a Confirm button that no
    longer exists. Deliberately silent on whether anything changed: after a
    lost acknowledgement, an earlier attempt on the same proposal may have
    landed.

    Args:
        exc: The refusal `ProposalStore.confirm()` raised.

    Returns:
        str: App-authored text stating that the request was not confirmed
        and is closed.

    """
    return (
        f"Not confirmed: {exc} This request is closed and can no longer be "
        "confirmed."
    )
