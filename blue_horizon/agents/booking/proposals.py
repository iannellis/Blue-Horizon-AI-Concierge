"""App-owned booking proposal store: the human-in-the-loop mechanism.

The LLM never commits a booking, cancellation, or modification directly. It
calls a `propose_*` tool, which prices the request in Python and stores the
result here as a `Proposal`. The application renders a confirmation dialog
from `Proposal.summary` -- resolved, priced line items computed by Python,
never LLM text -- and only `ProposalStore.confirm()`, triggered by the user's
Confirm button, calls into `write_ops` to actually mutate the database.

A proposal reserves nothing: `room_availability` is untouched until commit.
The TTL below exists only to bound the store's size, not to release held
inventory, since none is held.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Literal, cast

from blue_horizon.agents.booking import write_ops

if TYPE_CHECKING:
    from psycopg_pool import AsyncConnectionPool

logger = logging.getLogger(__name__)

ProposalAction = Literal["book", "cancel", "modify"]

# Opaque payload forwarded to the matching `write_ops` function on confirm.
type CancelDetails = tuple[int, list[write_ops.CancelRoomInstruction] | None]
type ModifyDetails = tuple[int, list[write_ops.ModifyRoomInstruction]]
type ProposalDetails = list[write_ops.RoomRequest] | CancelDetails | ModifyDetails

# Result of a confirmed proposal, one variant per `ProposalAction`.
type WriteResult = (
    write_ops.CommitResult | write_ops.CancelResult | write_ops.ModifyResult
)


class ProposalError(Exception):
    """Base class for proposal-store errors that map to user-facing refusals."""


class ProposalNotFoundError(ProposalError):
    """Raised when a proposal id is unknown or has expired."""


class ProposalOwnershipError(ProposalError):
    """Raised when a proposal is confirmed/dismissed by the wrong guest."""


@dataclass(frozen=True, slots=True)
class Proposal:
    """One pending action awaiting human confirmation.

    Attributes:
        proposal_id: Opaque identifier handed to the client.
        thread_id: Conversation thread this proposal belongs to.
        customer_id: Guest this proposal is for.
        action: Which kind of write this proposal would perform.
        summary: Action-shaped, JSON-serializable line items for the
            confirmation dialog. Built strictly from resolved database state,
            never from LLM text. Always carries a ``"total"`` key: a
            fixed-2-decimal string of the headline amount (charge for
            `book`, refund for `cancel`, new total for `modify`), checked
            against the commit's own computed total as a defense-in-depth
            assertion.
        details: Opaque payload forwarded to the matching `write_ops`
            function on confirm (a `RoomRequest` list for `book`, or the
            `booking_id` + instructions for `cancel`/`modify`).
        created_at: Creation timestamp, used for TTL expiry.

    """

    proposal_id: str
    thread_id: str
    customer_id: int
    action: ProposalAction
    summary: dict[str, Any]
    details: ProposalDetails
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True, slots=True)
class ConfirmOutcome:
    """Result of a successful `ProposalStore.confirm()` call.

    Attributes:
        proposal: The proposal that was confirmed.
        result: The `write_ops` result (`CommitResult`, `CancelResult`, or
            `ModifyResult`) for `proposal.action`.
        already_confirmed: True if this call replayed a cached result from an
            earlier confirm of the same `proposal_id`, rather than performing
            a new write.

    """

    proposal: Proposal
    result: WriteResult
    already_confirmed: bool


class ProposalStore:
    """In-process, TTL-bounded store of pending and recently-confirmed proposals.

    Not a reservation system: `create()` never touches the database. Backed
    by a plain dict, matching the `MemorySaver` checkpointer's process-lifetime
    model -- both are cleared together by `blue_horizon.agents.orchestration
    .resources.OrchestrationResources` on a database reset.
    """

    __slots__ = ("_by_thread", "_proposals", "_results", "_ttl_s")

    def __init__(self, *, ttl_s: float) -> None:
        """Initialize an empty store.

        Args:
            ttl_s: Seconds after which an unconfirmed proposal is purged.
                Confirmed proposals are exempt, so a delayed duplicate
                confirm can still replay its cached result.

        """
        self._proposals: dict[str, Proposal] = {}
        self._by_thread: dict[str, str] = {}
        self._results: dict[str, WriteResult] = {}
        self._ttl_s = ttl_s

    def create(
        self,
        *,
        thread_id: str,
        customer_id: int,
        action: ProposalAction,
        summary: dict[str, Any],
        details: ProposalDetails,
    ) -> Proposal:
        """Store a new proposal, superseding any pending proposal on this thread.

        Args:
            thread_id: Conversation thread this proposal belongs to.
            customer_id: Guest this proposal is for.
            action: Which kind of write this proposal would perform.
            summary: Action-shaped line items for the confirmation dialog.
            details: Opaque payload forwarded to `write_ops` on confirm.

        Returns:
            Proposal: The newly stored proposal.

        """
        self._purge_expired()
        self.invalidate_thread(thread_id)

        proposal = Proposal(
            proposal_id=uuid.uuid4().hex,
            thread_id=thread_id,
            customer_id=customer_id,
            action=action,
            summary=summary,
            details=details,
        )
        self._proposals[proposal.proposal_id] = proposal
        self._by_thread[thread_id] = proposal.proposal_id
        return proposal

    def get_pending_for_thread(self, thread_id: str) -> Proposal | None:
        """Return the pending proposal for a thread, if any and not expired.

        Args:
            thread_id: Conversation thread to look up.

        Returns:
            Proposal | None: The pending proposal, or ``None``.

        """
        self._purge_expired()
        proposal_id = self._by_thread.get(thread_id)
        return self._proposals.get(proposal_id) if proposal_id else None

    def invalidate_thread(self, thread_id: str) -> None:
        """Invalidate any pending proposal for a thread without confirming it.

        Called whenever a new proposal supersedes an old one, and whenever a
        new user message arrives -- a dialog must never be confirmable after
        the conversation has moved past it.

        Args:
            thread_id: Conversation thread to invalidate.

        """
        proposal_id = self._by_thread.pop(thread_id, None)
        if proposal_id is not None:
            self._proposals.pop(proposal_id, None)

    def dismiss(self, *, proposal_id: str, customer_id: int) -> Proposal:
        """Mark a pending proposal as dismissed, without writing anything.

        Args:
            proposal_id: Proposal to dismiss.
            customer_id: Guest requesting the dismissal; must own the proposal.

        Returns:
            Proposal: The dismissed proposal.

        Raises:
            ProposalNotFoundError: If the proposal is unknown, expired, or
                already used.
            ProposalOwnershipError: If `customer_id` does not own the proposal.

        """
        self._purge_expired()
        proposal = self._get_pending_and_validate(proposal_id, customer_id)
        self._retire(proposal)
        return proposal

    async def confirm(
        self,
        *,
        proposal_id: str,
        customer_id: int,
        write_pool: AsyncConnectionPool[Any],
    ) -> ConfirmOutcome:
        """Confirm a pending proposal, committing it through `write_ops`.

        Single-use: a second confirm of the same `proposal_id` is a no-op
        that replays the cached result rather than writing again, so a
        double-click cannot double-book.

        Args:
            proposal_id: Proposal to confirm.
            customer_id: Guest requesting confirmation; must own the proposal.
            write_pool: Read-write booking database pool (`bh_agent_rw`).

        Returns:
            ConfirmOutcome: The write result and whether it was replayed.

        Raises:
            ProposalNotFoundError: If the proposal is unknown or expired.
            ProposalOwnershipError: If `customer_id` does not own the proposal.
            write_ops.BookingWriteError: If the underlying write fails (for
                example, the proposed nights were taken in the meantime).

        """
        self._purge_expired()

        cached = self._results.get(proposal_id)
        if cached is not None:
            proposal = self._proposals[proposal_id]
            if proposal.customer_id != customer_id:
                raise ProposalOwnershipError(_OWNERSHIP_MSG)
            return ConfirmOutcome(
                proposal=proposal, result=cached, already_confirmed=True,
            )

        proposal = self._get_pending_and_validate(proposal_id, customer_id)
        self._retire(proposal)

        result = await _commit_proposal(write_pool, proposal)
        self._results[proposal_id] = result
        return ConfirmOutcome(proposal=proposal, result=result, already_confirmed=False)

    def _get_pending_and_validate(self, proposal_id: str, customer_id: int) -> Proposal:
        """Look up a still-pending proposal and verify ownership.

        Args:
            proposal_id: Proposal to look up.
            customer_id: Expected owner.

        Returns:
            Proposal: The pending proposal.

        Raises:
            ProposalNotFoundError: If no pending proposal has this id.
            ProposalOwnershipError: If `customer_id` does not own it.

        """
        proposal = self._proposals.get(proposal_id)
        still_pending = (
            proposal is not None
            and self._by_thread.get(proposal.thread_id) == proposal_id
        )
        if not still_pending:
            raise ProposalNotFoundError(_EXPIRED_MSG)
        proposal = cast("Proposal", proposal)
        if proposal.customer_id != customer_id:
            raise ProposalOwnershipError(_OWNERSHIP_MSG)
        return proposal

    def _retire(self, proposal: Proposal) -> None:
        """Remove a proposal from the pending-by-thread index.

        The record itself is kept in `_proposals` so a subsequent confirm can
        still resolve `already_confirmed` results, and so `dismiss` calls to
        stale ids raise `ProposalNotFoundError` instead of silently no-op'ing.

        Args:
            proposal: Proposal to retire.

        """
        if self._by_thread.get(proposal.thread_id) == proposal.proposal_id:
            del self._by_thread[proposal.thread_id]

    def _purge_expired(self) -> None:
        """Drop pending proposals whose TTL has elapsed.

        Confirmed proposals (present in `_results`) are exempt so a delayed
        duplicate confirm can still replay its cached result.
        """
        cutoff = datetime.now(UTC) - timedelta(seconds=self._ttl_s)
        expired = [
            proposal_id
            for proposal_id, proposal in self._proposals.items()
            if proposal_id not in self._results and proposal.created_at < cutoff
        ]
        for proposal_id in expired:
            proposal = self._proposals.pop(proposal_id)
            if self._by_thread.get(proposal.thread_id) == proposal_id:
                del self._by_thread[proposal.thread_id]


_EXPIRED_MSG = "That request has expired, shall I check those dates again?"
_OWNERSHIP_MSG = "That request does not belong to this guest."


async def _commit_proposal(
    write_pool: AsyncConnectionPool[Any],
    proposal: Proposal,
) -> WriteResult:
    """Dispatch a confirmed proposal to its matching `write_ops` function.

    Args:
        write_pool: Read-write booking database pool (`bh_agent_rw`).
        proposal: The proposal being confirmed.

    Returns:
        The `write_ops` result for `proposal.action`.

    Raises:
        write_ops.BookingWriteError: If the write fails.
        AssertionError: If the computed total unexpectedly diverges from the
            total shown in the confirmation dialog. Prices are fixed and
            never written by any operation, so this can only fire on a bug
            in this codebase -- exactly the case where a guest would
            otherwise be charged a number the dialog never showed.

    """
    result: WriteResult
    if proposal.action == "book":
        result = await write_ops.commit_booking(
            write_pool,
            customer_id=proposal.customer_id,
            rooms=proposal.details,
        )
    elif proposal.action == "cancel":
        booking_id, instructions = proposal.details
        result = await write_ops.cancel_booking(
            write_pool,
            customer_id=proposal.customer_id,
            booking_id=booking_id,
            rooms=instructions,
        )
    else:
        booking_id, changes = proposal.details
        result = await write_ops.modify_booking(
            write_pool,
            customer_id=proposal.customer_id,
            booking_id=booking_id,
            changes=changes,
        )

    committed_total = (
        getattr(result, "total_amount", None)
        if proposal.action != "cancel"
        else getattr(result, "refunded_amount", None)
    )
    proposed_total = Decimal(proposal.summary["total"])
    if committed_total != proposed_total:
        msg = (
            f"Internal pricing mismatch on proposal {proposal.proposal_id}: "
            f"dialog showed {proposed_total}, commit computed {committed_total}."
        )
        raise AssertionError(msg)

    return result
