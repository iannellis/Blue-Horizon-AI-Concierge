"""Booking SQL agent package.

Re-exports the public API so callers can use the same import paths as before:

    from blue_horizon.agents.booking import BookingSqlResources, build_booking_agent
"""

from blue_horizon.agents.booking.config import (
    load_booking_config,
    render_system_prompt,
)
from blue_horizon.agents.booking.db_utils import ENUM_TYPES, fetch_rooms_metadata
from blue_horizon.agents.booking.factory import build_booking_agent
from blue_horizon.agents.booking.guardrails import validate_sql
from blue_horizon.agents.booking.proposals import (
    Proposal,
    ProposalError,
    ProposalNotFoundError,
    ProposalOwnershipError,
    ProposalStore,
)
from blue_horizon.agents.booking.resources import BookingSqlResources
from blue_horizon.agents.booking.write_ops import BookingWriteError

__all__ = [
    "ENUM_TYPES",
    "BookingSqlResources",
    "BookingWriteError",
    "Proposal",
    "ProposalError",
    "ProposalNotFoundError",
    "ProposalOwnershipError",
    "ProposalStore",
    "build_booking_agent",
    "fetch_rooms_metadata",
    "load_booking_config",
    "render_system_prompt",
    "validate_sql",
]
