"""Anonymous Parallel Search MCP provider using Streamable HTTP."""

import asyncio
import json
from typing import List, Optional

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from src.exceptions import CircuitOpenError, RateLimitError, SearchError
from src.state import SearchResult
from src.utils.web_utils import CircuitBreaker, SearchProvider

MCP_URL = "https://search.parallel.ai/mcp"
USER_AGENT = "deep-research-agent/0.1.0 (https://github.com/tarun7r/deep-research-agent)"


class ParallelSearchProvider(SearchProvider):
    """Opt-in, keyless search; the existing content extractor stays unchanged."""

    def __init__(self, max_results: int = 5, timeout: float = 30.0):
        self.max_results = max_results
        self.timeout = timeout
        self.circuit_breaker = CircuitBreaker(
            name="parallel", failure_threshold=3, reset_timeout=60.0
        )

    @property
    def name(self) -> str:
        return "parallel"

    async def search(self, query: str, max_results: Optional[int] = None) -> List[SearchResult]:
        if not self.circuit_breaker.can_execute():
            raise CircuitOpenError("parallel", self.circuit_breaker.get_retry_after())
        count = self.max_results if max_results is None else max_results
        if count <= 0:
            return []

        # HTTP errors can be wrapped in an SDK task group's ExceptionGroup.
        rate_limit = None

        async def check_response(response: httpx.Response) -> None:
            nonlocal rate_limit
            if response.status_code == 429:
                retry = response.headers.get("Retry-After", "60")
                rate_limit = RateLimitError(
                    "Parallel Search MCP rate limit", service="parallel",
                    retry_after=int(retry) if retry.isdigit() else 60,
                )

        try:
            async with asyncio.timeout(self.timeout):
                async with httpx.AsyncClient(
                    headers={"User-Agent": USER_AGENT}, timeout=self.timeout,
                    event_hooks={"response": [check_response]},
                ) as client:
                    async with streamable_http_client(MCP_URL, http_client=client) as (read, write, _):
                        async with ClientSession(read, write) as session:
                            await session.initialize()
                            response = await session.call_tool(
                                "web_search", {"objective": query, "search_queries": [query]}
                            )
                            if response.isError:
                                raise SearchError("Parallel Search MCP tool returned an error")
                            payload = response.structuredContent
                            if payload is None:
                                text = next(
                                    (item.text for item in response.content if item.type == "text"), ""
                                )
                                payload = json.loads(text)
                            results = [
                                SearchResult(
                                    query=query, title=item.get("title") or "",
                                    url=item["url"], snippet="\n\n".join(item.get("excerpts", [])),
                                )
                                for item in payload["results"][:count]
                            ]
            self.circuit_breaker.record_success()
            return results
        except Exception as error:
            self.circuit_breaker.record_failure()
            if rate_limit is not None:
                raise rate_limit from error
            raise SearchError("Parallel Search MCP search failed", details=str(error)) from error
