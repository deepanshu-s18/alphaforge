"""Real MCP protocol tests: official stdio client -> all 4 servers.

These exercise the actual FastMCP wire protocol (stdio transport, JSON-RPC),
not just in-process function calls. Skipped automatically when the `mcp`
package is absent (it is in dev extras, so CI runs them).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

mcp_client = pytest.importorskip("mcp.client.session", reason="mcp package not installed")
from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SERVERS = {
    "statistics": "alphaforge.mcp_servers.statistics.server",
    "market_data": "alphaforge.mcp_servers.market_data.server",
    "backtest": "alphaforge.mcp_servers.backtest.server",
    "retrieval": "alphaforge.mcp_servers.retrieval.server",
}


async def _call(module: str, tool: str | None, arguments: dict | None):
    """Start a server over stdio, optionally call one tool, return (tools, result)."""
    from mcp.types import TextContent

    params = StdioServerParameters(command=sys.executable, args=["-m", module],
                                   cwd=str(REPO))
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            if tool is None:
                return [t.name for t in tools.tools], None
            res = await session.call_tool(tool, arguments or {})
            text = "".join(c.text for c in res.content if isinstance(c, TextContent))
            return [t.name for t in tools.tools], text


def _run(coro):
    import asyncio

    return asyncio.run(asyncio.wait_for(coro, timeout=60))


def test_statistics_roundtrip():
    import json as _json

    tools, text = _run(_call(SERVERS["statistics"], "apply_bonferroni",
                             {"p_values": [0.001, 0.5], "alpha": 0.05}))
    assert "apply_bonferroni" in tools and "apply_bh_fdr" in tools
    # FastMCP emits each list element as its own content block: "truefalse"
    dec, flags, idx = _json.JSONDecoder(), [], 0
    while idx < len(text):
        val, idx = dec.raw_decode(text, idx)
        flags.append(val)
    assert flags == [True, False]


def test_market_data_list_universe():
    tools, text = _run(_call(SERVERS["market_data"], "list_universe", {}))
    assert "list_universe" in tools
    assert "XLK" in text and "megacap_tech" in text


def test_retrieval_semantic_search():
    tools, text = _run(_call(SERVERS["retrieval"], "semantic_search_tool",
                             {"query": "post-earnings announcement drift", "k": 2}))
    assert "semantic_search_tool" in tools
    assert "drift" in text.lower()


def test_backtest_lists_tools():
    tools, _ = _run(_call(SERVERS["backtest"], None, None))
    assert "run_backtest_tool" in tools


def test_all_servers_initialize():
    """Each server must complete the MCP initialize handshake."""
    for name, module in SERVERS.items():
        tools, _ = _run(_call(module, None, None))
        assert tools, f"{name} server returned no tools"
