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
import time
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
        logger.info("Proposal created: %s", _describe(proposal))
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
        logger.info("Proposal dismissed: %s", _describe(proposal))
        return proposal

    async def confirm(
        self,
        *,
        proposal_id: str,
        customer_id: int,
        write_pool: AsyncConnectionPool[Any],
    ) -> ConfirmOutcome:
        """Confirm a pending proposal, committing it through `write_ops`.

        Single-use on the success path: a second confirm of the same
        `proposal_id` after it has been committed is a no-op that replays
        the cached result rather than writing again, so a double-click
        cannot double-book. That guarantee does not extend to a confirm that
        raised `write_ops.BookingUnavailableError`: the proposal is left
        pending rather than retired (see below), specifically so the guest
        can press Confirm again. Retention is still bounded -- by `ttl_s`
        and by `invalidate_thread`, which `manager.ainvoke*` calls at the
        start of every turn -- so a retained-pending proposal cannot outlive
        the guest's next message.

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
                Retires the proposal: this is a deterministic refusal, and a
                retry would only re-evaluate the same now-known outcome.
            write_ops.PricingMismatchError: If the write computed a different
                total from the one the confirmation dialog showed. A
                `BookingWriteError` subclass, raised before the write
                commits, so nothing was written and the proposal is retired
                as above. Logged as an error, since only a bug can cause it.
            write_ops.BookingUnavailableError: If the database could not be
                reached at all, and either the action was not `"book"` or
                the reconciliation read below found nothing. Leaves the
                proposal pending instead of retiring it, so a subsequent
                confirm is a real retry rather than a 404.

        """
        self._purge_expired()

        cached = self._results.get(proposal_id)
        if cached is not None:
            proposal = self._proposals[proposal_id]
            _check_owner(proposal, customer_id)
            logger.info(
                "Proposal confirm replayed: %s %s",
                _describe(proposal),
                _describe_result(cached),
            )
            return ConfirmOutcome(
                proposal=proposal, result=cached, already_confirmed=True,
            )

        proposal = self._get_pending_and_validate(proposal_id, customer_id)

        # A retry lands here too, since the proposal above was left pending
        # on the previous attempt's BookingUnavailableError. It calls
        # write_ops.commit_booking (or cancel_/modify_) again, which
        # re-prices every night under FOR UPDATE from scratch -- the same
        # lock that makes double-booking structurally impossible in the
        # first place (invariant 6). Do not "optimise" this by skipping the
        # re-price on a retry: that lock is what makes retaining the
        # proposal across a failed attempt safe at all.
        started = time.perf_counter()
        try:
            result = await _commit_proposal(write_pool, proposal)
        except write_ops.PricingMismatchError as exc:
            self._retire(proposal)
            # Nothing was written, but only a bug can cause this, so it is
            # logged as an error with its traceback, not as a refusal.
            timing = _write_timing(proposal, started)
            logger.exception(
                "Proposal refused on a pricing mismatch: %s computed_total=%s "
                "duration_ms=%s",
                _describe(proposal),
                exc.computed_total,
                timing["duration_ms"],
                extra=timing,
            )
            raise
        except write_ops.BookingWriteError as exc:
            self._retire(proposal)
            timing = _write_timing(proposal, started)
            logger.info(
                "Proposal refused: %s reason=%r duration_ms=%s",
                _describe(proposal),
                str(exc),
                timing["duration_ms"],
                extra=timing,
            )
            raise
        except write_ops.BookingUnavailableError:
            timing = _write_timing(proposal, started)
            logger.warning(
                "Proposal commit could not reach the database: %s duration_ms=%s",
                _describe(proposal),
                timing["duration_ms"],
                extra=timing,
            )
            # The commit itself could not tell whether it landed: the
            # connection may have died after COMMIT but before the ack
            # returned. Settle it with a read instead of guessing, but only
            # for "book" -- find_booking_for_rooms relies on
            # booking_rooms_no_overlap's uniqueness, a property "cancel" and
            # "modify" don't share, and their own state checks (already
            # cancelled, unknown booking_room_id) already describe what
            # happened without needing this. If the reconciliation read
            # itself fails, let that exception propagate as-is: the proposal
            # stays pending either way, and turning one failure into a
            # second, different-shaped one would tell the guest something
            # this code never actually established.
            if proposal.action == "book":
                reconciled = await write_ops.find_booking_for_rooms(
                    write_pool,
                    customer_id=proposal.customer_id,
                    rooms=cast("list[write_ops.RoomRequest]", proposal.details),
                )
                if reconciled is not None:
                    self._retire(proposal)
                    self._results[proposal_id] = reconciled
                    timing = _write_timing(proposal, started)
                    logger.info(
                        "Proposal confirmed after a lost commit ack: %s %s "
                        "duration_ms=%s",
                        _describe(proposal),
                        _describe_result(reconciled),
                        timing["duration_ms"],
                        extra=timing,
                    )
                    return ConfirmOutcome(
                        proposal=proposal, result=reconciled, already_confirmed=False,
                    )
            raise

        self._retire(proposal)
        self._results[proposal_id] = result
        timing = _write_timing(proposal, started)
        logger.info(
            "Proposal confirmed: %s %s duration_ms=%s",
            _describe(proposal),
            _describe_result(result),
            timing["duration_ms"],
            extra=timing,
        )
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
            logger.info(
                "Proposal not pending (unknown, expired, or used): "
                "proposal_id=%s customer_id=%s",
                proposal_id,
                customer_id,
            )
            raise ProposalNotFoundError(_EXPIRED_MSG)
        proposal = cast("Proposal", proposal)
        _check_owner(proposal, customer_id)
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


def _describe(proposal: Proposal) -> str:
    """Render a proposal's identifying fields for a log line.

    Every proposal lifecycle line carries these, so each line stands on its
    own in an audit without being joined to the line that created the
    proposal.

    Args:
        proposal: The proposal to describe.

    Returns:
        str: ``key=value`` pairs for the proposal id, action, thread, guest,
        and the total shown in the confirmation dialog.

    """
    return (
        f"proposal_id={proposal.proposal_id} action={proposal.action} "
        f"thread_id={proposal.thread_id} customer_id={proposal.customer_id} "
        f"total={proposal.summary['total']}"
    )


def _check_owner(proposal: Proposal, customer_id: int) -> None:
    """Refuse, and log, an action on a proposal by a guest who does not own it.

    Args:
        proposal: The proposal being acted on.
        customer_id: The guest requesting the action.

    Raises:
        ProposalOwnershipError: If `customer_id` does not own `proposal`.

    """
    if proposal.customer_id == customer_id:
        return
    logger.warning(
        "Proposal refused to a different guest: customer_id=%s attempted %s",
        customer_id,
        _describe(proposal),
    )
    raise ProposalOwnershipError(_OWNERSHIP_MSG)


def _describe_result(result: WriteResult) -> str:
    """Render a write result's identifying fields for a log line.

    Args:
        result: The `write_ops` result of a confirmed proposal.

    Returns:
        str: ``booking_id=...``, plus ``confirmation_number=...`` for a new
        booking.

    """
    fields = f"booking_id={result.booking_id}"
    confirmation_number = getattr(result, "confirmation_number", None)
    if confirmation_number is not None:
        fields += f" confirmation_number={confirmation_number}"
    return fields


def _write_timing(proposal: Proposal, started: float) -> dict[str, object]:
    """Measure a confirm's write, as a log record's extra fields.

    Attached to the record, so a shipped line carries the action and duration
    as attributes that can be grouped and aggregated.

    Args:
        proposal: The proposal being confirmed.
        started: `time.perf_counter` reading when the write began.

    Returns:
        dict[str, object]: ``action``, and ``duration_ms``, whole milliseconds
        elapsed, including any reconciliation read.

    """
    return {
        "action": proposal.action,
        "duration_ms": round((time.perf_counter() - started) * 1000),
    }


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
        write_ops.PricingMismatchError: If the total the write computes
            differs from the total shown in the confirmation dialog. The
            write checks this inside its transaction, so nothing is written.

    """
    # Checked by the write before it commits, not here after it returns: by
    # then the guest would already have been charged a number the dialog
    # never showed.
    expected_total = Decimal(proposal.summary["total"])
    # `Proposal.details` is an opaque, `action`-shaped payload (see the class
    # docstring): its concrete shape is guaranteed by `ProposalStore.create()`'s
    # caller, not by the type system, so each branch casts it to the shape
    # `action` promises.
    if proposal.action == "book":
        return await write_ops.commit_booking(
            write_pool,
            customer_id=proposal.customer_id,
            rooms=cast("list[write_ops.RoomRequest]", proposal.details),
            expected_total=expected_total,
        )
    if proposal.action == "cancel":
        booking_id, instructions = cast("CancelDetails", proposal.details)
        return await write_ops.cancel_booking(
            write_pool,
            customer_id=proposal.customer_id,
            booking_id=booking_id,
            rooms=instructions,
            expected_total=expected_total,
        )
    booking_id, changes = cast("ModifyDetails", proposal.details)
    return await write_ops.modify_booking(
        write_pool,
        customer_id=proposal.customer_id,
        booking_id=booking_id,
        changes=changes,
        expected_total=expected_total,
    )
