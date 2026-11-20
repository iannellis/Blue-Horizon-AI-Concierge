"""Temporary MCP server that echoes the proxy headers a request arrives with.

Deploy it on Prefect Horizon with the entrypoint ``mcp_server/echo_headers.py:mcp``,
call ``echo_headers`` from a client, and compare the result with your public IP
(for example from https://ifconfig.me). That shows which ``X-Forwarded-For``
entry is the real client, so the rate limiter knows which one to trust. Delete
this file afterward.
"""

from fastmcp import FastMCP
from fastmcp.server.dependencies import get_http_headers, get_http_request

# Horizon runs on AWS Lambda, which puts the caller's address in the request context.
_PROXY_HEADERS = (
    "x-forwarded-for",
    "x-real-ip",
    "forwarded",
    "x-forwarded-proto",
    "x-amzn-request-context",
    "user-agent",
    "x-anthropic-client",
)

mcp = FastMCP("echo-headers")


@mcp.tool
def echo_headers() -> dict[str, object]:
    """Return the proxy headers and socket address of the current request.

    Returns:
        dict[str, object]: ``proxy_headers`` (the proxy headers that were set),
        ``x_forwarded_for_entries`` (that header split into its entries, left to
        right), ``socket_client`` (the address the connection came from), and
        ``all_header_names`` (every header name, with credentials stripped).

    """
    headers = get_http_headers()
    client = get_http_request().client
    entries = headers.get("x-forwarded-for", "").split(",")
    return {
        "proxy_headers": {n: headers[n] for n in _PROXY_HEADERS if n in headers},
        "x_forwarded_for_entries": [e.strip() for e in entries if e.strip()],
        "socket_client": f"{client.host}:{client.port}" if client else None,
        "all_header_names": sorted(headers),
    }
