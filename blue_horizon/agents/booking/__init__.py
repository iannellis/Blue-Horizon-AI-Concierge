"""Booking agent package.

Re-exports the public API so callers can use the same import paths as before:

    from blue_horizon.agents.booking import BookingSqlResources, build_booking_agent
"""

from blue_horizon.agents.booking.config import (
    load_booking_config,
    render_system_prompt,
)
from blue_horizon.agents.booking.db_utils import (
    ENUM_TYPES,
    assert_pool_is_read_only,
    check_credentials,
    fetch_rooms_metadata,
    is_transient_conn_error,
)
from blue_horizon.agents.booking.factory import (
    build_booking_agent,
    database_unavailable_this_turn,
)
from blue_horizon.agents.booking.proposals import (
    Proposal,
    ProposalError,
    ProposalNotFoundError,
    ProposalOwnershipError,
    ProposalStore,
)
from blue_horizon.agents.booking.resources import BookingSqlResources
from blue_horizon.agents.booking.search import (
    SEARCH_TOOL_DESCRIPTION,
    RoomsMetadata,
    SearchArgsBase,
    build_search_args_model,
    run_room_search,
)
from blue_horizon.agents.booking.search_runner import (
    SqlErrorKind,
    search_rooms_with_retry,
)
from blue_horizon.agents.booking.write_ops import BookingWriteError

__all__ = [
    "ENUM_TYPES",
    "SEARCH_TOOL_DESCRIPTION",
    "BookingSqlResources",
    "BookingWriteError",
    "Proposal",
    "ProposalError",
    "ProposalNotFoundError",
    "ProposalOwnershipError",
    "ProposalStore",
    "RoomsMetadata",
    "SearchArgsBase",
    "SqlErrorKind",
    "assert_pool_is_read_only",
    "build_booking_agent",
    "build_search_args_model",
    "check_credentials",
    "database_unavailable_this_turn",
    "fetch_rooms_metadata",
    "is_transient_conn_error",
    "load_booking_config",
    "render_system_prompt",
    "run_room_search",
    "search_rooms_with_retry",
]
