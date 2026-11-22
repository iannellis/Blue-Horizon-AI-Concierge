# Using the MCP Server

The hotel's room search is also available as an MCP server, so any MCP client (Claude,
an IDE assistant, your own agent) can search Blue Horizon's rooms directly. It is
deployed on Prefect Horizon at:

```text
https://blue-horizon.fastmcp.app/mcp
```

**The MCP server only searches rooms.** It cannot book, cancel, or modify a reservation,
cannot see any guest's reservations, and does not answer questions about the hotel's
amenities or services. For any of those, use the [demo](using-the-demo.md).

## Connecting

The server uses the Streamable HTTP transport and needs no sign-in or API key.

**Claude (claude.ai or the desktop app):** open *Settings*, then *Connectors*, choose
*Add custom connector*, and paste the URL above.

**Claude Code:**

```bash
claude mcp add --transport http blue-horizon https://blue-horizon.fastmcp.app/mcp
```

**Other clients** that take a JSON configuration usually accept an entry like this:

```json
{
  "mcpServers": {
    "blue-horizon": {
      "url": "https://blue-horizon.fastmcp.app/mcp"
    }
  }
}
```

Once connected, ask in plain language, for example *find me an ocean-view suite at Blue
Horizon for March 14 to 17, 2025*. Naming the hotel helps your client pick this tool
rather than a web search.

## What to expect

The server exposes one tool, `search_rooms`. It takes the same filters as the
concierge's own search: dates, room type, bed type, view, floor, guests, size,
accessibility, nightly price, specific room numbers, and amenities, sorted by stay
price, floor, size, or room number. See
[What you can search for](using-the-demo.md#what-you-can-search-for) for the values.

- **The room calendar covers January 4, 2025 to January 3, 2026.** The tool tells your
  client to treat those dates as open whatever today's date is, and to pick the year
  that puts your dates in that window when you give none.
- **Results are structured data, not prose.** Each search returns `matching_count`,
  the number of rooms that match, and a list of rooms with their number, floor, type,
  bed type, occupancy, size, views, amenities, and accessibility. With dates, each room
  also carries the number of nights and the stay's `total_price`. Your client's model
  writes the answer from that data.
- **A search returns four rooms by default.** Your client can ask for more, up to every
  room in the hotel; `matching_count` always counts every match.
- **Invalid filters are rejected with a message naming the problem,** such as a date
  outside the calendar or a price filter without dates, so your client can correct the
  search.
- **The server is rate limited,** at 2 requests per second with bursts of up to 20,
  shared by everyone using it. A request over the limit gets an error; wait a moment and
  try again.
- **A database failure returns an error saying to try again later,** not an empty
  result. The first search after a quiet stretch can take a few extra seconds while the
  database wakes up.
- **Availability reflects the live demo.** Rooms booked by demo visitors show as
  unavailable until they are cancelled or the nightly reset clears them.

The server connects to the database with a read-only role that can see only rooms and
their availability, so no guest data can reach your client. The
[Booking Agent](architecture/booking-agent.md#mcp-server) page explains how it is built.
