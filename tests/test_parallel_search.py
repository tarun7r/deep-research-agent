"""Exercise the configured LangChain caller and real MCP HTTP transport offline."""

import asyncio
import json
from unittest.mock import patch

import httpx
import pytest
from mcp import ClientSession

from src.config import config
from src.exceptions import RateLimitError, SearchError
from src.utils import tools
from src.utils.parallel_search import MCP_URL, USER_AGENT, ParallelSearchProvider
from src.utils.web_utils import DuckDuckGoProvider, TavilyProvider, WebSearchTool


@pytest.fixture
def mcp_http(monkeypatch):
    observed = []
    state = {"status": 200, "error": False, "text_only": False}

    def handle(request):
        observed.append(request)
        if state["status"] != 200:
            return httpx.Response(state["status"], headers={"Retry-After": "120"})
        if request.method == "GET":
            return httpx.Response(405)
        if request.method == "DELETE":
            return httpx.Response(200)
        data = json.loads(request.content)
        if "id" not in data:
            return httpx.Response(202)
        if data["method"] == "initialize":
            result = {"protocolVersion": "2025-11-25", "capabilities": {},
                      "serverInfo": {"name": "fixture", "version": "1"}}
        elif data["method"] == "tools/list":
            result = {"tools": [{"name": "web_search", "inputSchema": {"type": "object"}}]}
        else:
            payload = {"results": [
                {"url": "https://docs.python.org/3/library/asyncio.html",
                 "title": None, "excerpts": ["asyncio documentation", "Concurrent tasks"]},
                {"url": "https://example.org", "title": "Second", "excerpts": []},
            ]}
            result = {"content": [{"type": "text", "text": json.dumps(payload)}],
                      "isError": state["error"]}
            if not state["text_only"]:
                result["structuredContent"] = payload
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": data["id"], "result": result})

    real_client = httpx.AsyncClient

    def client(**kwargs):
        return real_client(transport=httpx.MockTransport(handle), **kwargs)

    monkeypatch.setattr("src.utils.parallel_search.httpx.AsyncClient", client)
    return observed, state


def test_configured_tool_and_headers(monkeypatch, mcp_http):
    observed, _ = mcp_http
    monkeypatch.setattr(config, "search_provider", "parallel")
    providers = tools._build_search_providers()
    assert isinstance(providers[0], ParallelSearchProvider)
    monkeypatch.setattr(tools, "_search_impl", WebSearchTool(providers=providers))
    results = asyncio.run(tools.web_search.ainvoke({"query": "Python asyncio", "max_results": 1}))
    assert results == [{"query": "Python asyncio", "title": "",
                        "url": "https://docs.python.org/3/library/asyncio.html",
                        "snippet": "asyncio documentation\n\nConcurrent tasks"}]
    assert observed
    assert all(str(r.url) == MCP_URL for r in observed)
    assert all(r.headers["User-Agent"] == USER_AGENT for r in observed)
    assert all("Authorization" not in r.headers for r in observed)
    calls = [json.loads(r.content) for r in observed if r.method == "POST"]
    assert next(c for c in calls if c["method"] == "tools/call")["params"] == {
        "name": "web_search", "arguments": {"objective": "Python asyncio", "search_queries": ["Python asyncio"]}}


def test_text_fallback(mcp_http):
    mcp_http[1]["text_only"] = True
    results = asyncio.run(ParallelSearchProvider().search("query", 1))
    assert len(results) == 1 and "Concurrent tasks" in results[0].snippet


@pytest.mark.parametrize("status,error_type", [(429, RateLimitError), (503, SearchError)])
def test_http_failure(mcp_http, status, error_type):
    mcp_http[1]["status"] = status
    provider = ParallelSearchProvider()
    with pytest.raises(error_type) as caught:
        asyncio.run(provider.search("query"))
    assert provider.circuit_breaker._failures == 1
    if status == 429:
        assert caught.value.retry_after == 120


def test_tool_error(mcp_http):
    mcp_http[1]["error"] = True
    with pytest.raises(SearchError):
        asyncio.run(ParallelSearchProvider().search("query"))


def test_defaults_and_explicit_tavily(monkeypatch):
    monkeypatch.setattr(config, "search_provider", "duckduckgo")
    assert isinstance(tools._build_search_providers()[0], DuckDuckGoProvider)
    monkeypatch.setattr(config, "search_provider", "tavily")
    with patch("src.utils.tools.TavilyProvider", spec=TavilyProvider) as provider:
        assert tools._build_search_providers() == [provider.return_value]


def test_cancellation(monkeypatch, mcp_http):
    async def cancel(*args, **kwargs):
        raise asyncio.CancelledError()
    monkeypatch.setattr(ClientSession, "initialize", cancel)
    # Cancellation propagates through the real transport, rather than becoming a search error.
    async def run():
        provider = ParallelSearchProvider()
        with pytest.raises(asyncio.CancelledError):
            await provider.search("query")
        assert provider.circuit_breaker._failures == 0
    asyncio.run(run())


def test_timeout(monkeypatch, mcp_http):
    async def slow(*args, **kwargs):
        await asyncio.sleep(10)
    monkeypatch.setattr(ClientSession, "initialize", slow)
    provider = ParallelSearchProvider(timeout=0.01)
    with pytest.raises(SearchError):
        asyncio.run(provider.search("query"))
    assert provider.circuit_breaker._failures == 1
