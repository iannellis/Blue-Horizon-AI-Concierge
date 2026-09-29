"""Hand-built search metadata shared by the booking unit tests.

None of these values are read from a database. They mirror the shape of the
real data (an availability window that crosses New Year, floors 1 to 20) so
that edge cases can be tested without a connection. Whether the real data has
this window is checked separately, in `test_search_db.py`.
"""

from __future__ import annotations

import datetime as dt

from blue_horizon.agents.booking.search import RoomsMetadata

FIRST_NIGHT = dt.date(2025, 1, 4)
LAST_NIGHT = dt.date(2026, 1, 3)
MAX_ROOM_NUMBERS = 10
DEFAULT_RESULTS = 4
MAX_RESULTS = 15


def make_rooms_metadata() -> RoomsMetadata:
    """Build a `RoomsMetadata` with made-up vocabularies.

    Returns:
        Metadata with a 2025-01-04 to 2026-01-03 window and floors 1 to 20.

    """
    return RoomsMetadata(
        enum_values={
            "room_type": ["Standard", "Suite", "Presidential Suite"],
            "room_bed_type": ["Queen", "King"],
        },
        amenities=("Balcony", "Mini Bar", "Wi-Fi"),
        view_types=("City View", "Ocean View"),
        min_floor=1,
        max_floor=20,
        first_night=FIRST_NIGHT,
        last_night=LAST_NIGHT,
    )
