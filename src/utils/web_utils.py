"""Web search and content extraction utilities with async support and circuit breaker."""

import asyncio
import re
import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from urllib.parse import urlparse
import logging
from enum import Enum
from abc import ABC, abstractmethod

import httpx
from bs4 import BeautifulSoup
from ddgs import DDGS

from src.state import SearchResult
from src.exceptions import (
    SearchError, 
    RateLimitError, 
    ContentExtractionError,
    CircuitOpenError
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Circuit Breaker Implementation
# =============================================================================

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    """Circuit breaker for external service calls."""
    
    name: str
    failure_threshold: int = 5
    reset_timeout: float = 30.0
    half_open_max_calls: int = 1
    
    _failures: int = field(default=0, init=False)
    _successes: int = field(default=0, init=False)
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _last_failure_time: Optional[float] = field(default=None, init=False)
    _half_open_calls: int = field(default=0, init=False)
    
    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            if self._last_failure_time and (time.time() - self._last_failure_time) >= self.reset_timeout:
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
                logger.info(f"Circuit '{self.name}' transitioning to HALF_OPEN")
        return self._state
    
    def can_execute(self) -> bool:
        state = self.state
        if state == CircuitState.CLOSED:
            return True
        if state == CircuitState.HALF_OPEN:
            return self._half_open_calls < self.half_open_max_calls
        return False
    
    def record_success(self) -> None:
        if self._state == CircuitState.HALF_OPEN:
            self._successes += 1
            if self._successes >= self.half_open_max_calls:
                self._state = CircuitState.CLOSED
                self._failures = 0
                self._successes = 0
                logger.info(f"Circuit '{self.name}' closed after successful recovery")
        else:
            self._failures = 0
    
    def record_failure(self) -> None:
        self._failures += 1
        self._last_failure_time = time.time()
        
        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            logger.warning(f"Circuit '{self.name}' reopened after half-open failure")
        elif self._failures >= self.failure_threshold:
            self._state = CircuitState.OPEN
            logger.warning(f"Circuit '{self.name}' opened after {self._failures} failures")
    
    def get_retry_after(self) -> float:
        if self._last_failure_time:
            elapsed = time.time() - self._last_failure_time
            return max(0, self.reset_timeout - elapsed)
        return self.reset_timeout


# =============================================================================
# HTTP Client Manager (Connection Pooling)
# =============================================================================

class HTTPClientManager:
    """Manages a shared httpx AsyncClient with connection pooling."""
    
    _instance: Optional['HTTPClientManager'] = None
    
    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None
        self._lock = asyncio.Lock()
    
    @classmethod
    def get_instance(cls) -> 'HTTPClientManager':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    async def get_client(self) -> httpx.AsyncClient:
        async with self._lock:
            if self._client is None or self._client.is_closed:
                self._client = httpx.AsyncClient(
                    limits=httpx.Limits(
                        max_connections=50,
                        max_keepalive_connections=20,
                        keepalive_expiry=30.0
                    ),
                    timeout=httpx.Timeout(15.0, connect=5.0),
                    follow_redirects=True,
                    http2=True,
                    headers={
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                        'Accept-Language': 'en-US,en;q=0.9',
                    }
                )
            return self._client
    
    async def close(self) -> None:
        async with self._lock:
            if self._client and not self._client.is_closed:
                await self._client.aclose()
                self._client = None


# =============================================================================
# URL Validation
# =============================================================================

BLOCKED_HOSTS = {'localhost', '127.0.0.1', '0.0.0.0', '::1', '[::1]'}
BLOCKED_SCHEMES = {'file', 'ftp', 'data', 'javascript'}


def is_valid_url(url: str) -> bool:
    """Check if a URL is valid and safe to access."""
    try:
        result = urlparse(url)
        if not all([result.scheme, result.netloc]):
            return False
        if result.scheme.lower() not in {'http', 'https'}:
            return False
        hostname = result.hostname or ''
        if hostname.lower() in BLOCKED_HOSTS:
            return False
        return True
    except Exception:
        return False


# =============================================================================
# Search Provider Abstraction
# =============================================================================

class SearchProvider(ABC):
    """Abstract base class for search providers."""
    
    @abstractmethod
    async def search(self, query: str, max_results: int) -> List[SearchResult]:
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass


class DuckDuckGoProvider(SearchProvider):
    """DuckDuckGo search provider with rate limiting and circuit breaker."""
    
    def __init__(self, max_results: int = 5):
        self.max_results = max_results
        self.last_search_time = 0.0
        self.min_delay = 2.0
        self.circuit_breaker = CircuitBreaker(
            name="duckduckgo",
            failure_threshold=3,
            reset_timeout=60.0
        )
    
    @property
    def name(self) -> str:
        return "duckduckgo"
    
    async def search(self, query: str, max_results: Optional[int] = None) -> List[SearchResult]:
        if not self.circuit_breaker.can_execute():
            retry_after = self.circuit_breaker.get_retry_after()
            raise CircuitOpenError("duckduckgo", retry_after)
        
        results_count = max_results or self.max_results
        
        try:
            elapsed = time.time() - self.last_search_time
            if elapsed < self.min_delay:
                wait_time = self.min_delay - elapsed
                logger.debug(f"Rate limiting: waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
            
            logger.info(f"Searching DuckDuckGo for: {query}")
            
            results = await self._execute_search(query, results_count)
            
            self.last_search_time = time.time()
            self.circuit_breaker.record_success()
            
            logger.info(f"Found {len(results)} results for: {query}")
            return results
            
        except CircuitOpenError:
            raise
        except Exception as e:
            self.circuit_breaker.record_failure()
            self.last_search_time = time.time()
            
            error_str = str(e).lower()
            if "ratelimit" in error_str or "202" in error_str:
                raise RateLimitError(
                    message=f"DuckDuckGo rate limit: {str(e)}",
                    retry_after=60,
                    service="duckduckgo"
                )
            
            raise SearchError(f"Search failed for '{query}'", details=str(e))
    
    async def _execute_search(self, query: str, max_results: int) -> List[SearchResult]:
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                search_results = await asyncio.to_thread(
                    self._sync_search, query, max_results
                )
                return search_results
            except Exception as e:
                error_str = str(e).lower()
                if ("ratelimit" in error_str or "202" in error_str) and attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5
                    logger.warning(f"Rate limit hit, waiting {wait_time}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(wait_time)
                else:
                    raise
        
        return []
    
    def _sync_search(self, query: str, max_results: int) -> List[SearchResult]:
        results = []
        ddgs = DDGS()
        search_results = list(ddgs.text(query, max_results=max_results))
        
        for result in search_results:
            results.append(SearchResult(
                query=query,
                title=result.get("title", ""),
                url=result.get("href", ""),
                snippet=result.get("body", "")
            ))
        
        return results


class TavilyProvider(SearchProvider):
    """Tavily search provider with circuit breaker integration."""

    def __init__(self, api_key: str, max_results: int = 5):
        from tavily import TavilyClient
        self.max_results = max_results
        self.client = TavilyClient(api_key=api_key)
        self.circuit_breaker = CircuitBreaker(
            name="tavily",
            failure_threshold=3,
            reset_timeout=60.0
        )

    @property
    def name(self) -> str:
        return "tavily"

    async def search(self, query: str, max_results: Optional[int] = None) -> List[SearchResult]:
        if not self.circuit_breaker.can_execute():
            retry_after = self.circuit_breaker.get_retry_after()
            raise CircuitOpenError("tavily", retry_after)

        results_count = max_results or self.max_results

        try:
            logger.info(f"Searching Tavily for: {query}")

            response = await asyncio.to_thread(
                self.client.search,
                query=query,
                max_results=results_count,
                search_depth="basic",
            )

            results = []
            for item in response.get("results", []):
                results.append(SearchResult(
                    query=query,
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("content", ""),
                ))

            self.circuit_breaker.record_success()
            logger.info(f"Found {len(results)} results for: {query}")
            return results

        except CircuitOpenError:
            raise
        except Exception as e:
            self.circuit_breaker.record_failure()
            error_str = str(e).lower()
            if "rate" in error_str or "limit" in error_str or "429" in error_str:
                raise RateLimitError(
                    message=f"Tavily rate limit: {str(e)}",
                    retry_after=60,
                    service="tavily"
                )
            raise SearchError(f"Tavily search failed for '{query}'", details=str(e))


class WebSearchTool:
    """Web search tool with provider abstraction and fallback support."""
    
    def __init__(self, max_results: int = 5, providers: Optional[List[SearchProvider]] = None):
        self.max_results = max_results
        self.providers = providers or [DuckDuckGoProvider(max_results)]
    
    def search(self, query: str) -> List[SearchResult]:
        """Synchronous search - runs async search in event loop."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.search_async(query))
    
    async def search_async(self, query: str) -> List[SearchResult]:
        """Async search with provider fallback."""
        last_error: Optional[Exception] = None
        
        for provider in self.providers:
            try:
                return await provider.search(query, self.max_results)
            except CircuitOpenError as e:
                logger.warning(f"Provider {provider.name} circuit open, trying next")
                last_error = e
                continue
            except RateLimitError as e:
                logger.warning(f"Provider {provider.name} rate limited: {e}")
                last_error = e
                continue
            except SearchError as e:
                logger.error(f"Provider {provider.name} error: {e}")
                last_error = e
                continue
        
        if last_error:
            logger.error(f"All search providers failed. Last error: {last_error}")
        return []


# =============================================================================
# Content Extractor (True Async with httpx)
# =============================================================================

class ContentExtractor:
    """Extract and clean content from web pages using httpx."""
    
    CONTENT_SELECTORS = [
        'article',
        'main',
        '[role="main"]',
        '.post-content',
        '.article-content',
        '.entry-content',
        '.content',
        '#content',
        '.post',
        '.article'
    ]
    
    REMOVE_SELECTORS = [
        'script', 'style', 'nav', 'footer', 'header', 'aside',
        '.sidebar', '.navigation', '.menu', '.comments', '.comment',
        '.advertisement', '.ad', '.ads', '.social-share', '.related-posts',
        '[role="navigation"]', '[role="complementary"]'
    ]
    
    def __init__(self, timeout: int = 15, max_content_length: int = 8000):
        self.timeout = timeout
        self.max_content_length = max_content_length
        self.client_manager = HTTPClientManager.get_instance()
        self.circuit_breaker = CircuitBreaker(
            name="content_extraction",
            failure_threshold=10,
            reset_timeout=30.0
        )
    
    def extract_content(self, url: str) -> Optional[str]:
        """Synchronous content extraction - runs async in event loop."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.extract_content_async(url))
    
    async def extract_content_async(self, url: str) -> Optional[str]:
        """Async content extraction with httpx."""
        if not is_valid_url(url):
            logger.warning(f"Invalid URL: {url}")
            return None
        
        if not self.circuit_breaker.can_execute():
            logger.warning(f"Circuit breaker open for content extraction")
            return None
        
        try:
            logger.info(f"Extracting content from: {url}")
            
            client = await self.client_manager.get_client()
            response = await client.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '').lower()
            if not any(ct in content_type for ct in ['text/html', 'application/xhtml']):
                logger.warning(f"Unsupported content type: {content_type}")
                return None
            
            html_content = response.text
            extracted = self._parse_html(html_content)
            
            self.circuit_breaker.record_success()
            
            if extracted:
                logger.info(f"Extracted {len(extracted)} characters from {url}")
            
            return extracted
            
        except httpx.HTTPStatusError as e:
            self.circuit_breaker.record_failure()
            raise ContentExtractionError(
                f"HTTP error: {e.response.status_code}",
                url=url,
                status_code=e.response.status_code
            )
        except httpx.TimeoutException:
            self.circuit_breaker.record_failure()
            logger.warning(f"Timeout extracting content from {url}")
            return None
        except Exception as e:
            self.circuit_breaker.record_failure()
            logger.warning(f"Failed to extract content from {url}: {str(e)}")
            return None
    
    def _parse_html(self, html: str) -> Optional[str]:
        """Parse HTML and extract main content."""
        soup = BeautifulSoup(html, 'html.parser')
        
        for selector in self.REMOVE_SELECTORS:
            for element in soup.select(selector):
                element.decompose()
        
        main_content = None
        for selector in self.CONTENT_SELECTORS:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        if not main_content:
            main_content = soup.body
        
        if main_content:
            text = main_content.get_text(separator='\n', strip=True)
            text = re.sub(r'\n\s*\n+', '\n\n', text)
            text = re.sub(r' +', ' ', text)
            text = text[:self.max_content_length] if len(text) > self.max_content_length else text
            return text
        
        return None
    
    async def enhance_search_results_async(
        self, 
        results: List[SearchResult],
        max_concurrent: int = 5
    ) -> List[SearchResult]:
        """Enhance search results with full content extraction (async with semaphore)."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def enhance_one(result: SearchResult) -> SearchResult:
            async with semaphore:
                if not result.content:
                    try:
                        content = await self.extract_content_async(result.url)
                        if content:
                            result.content = content
                    except ContentExtractionError as e:
                        logger.warning(f"Failed to enhance {result.url}: {e}")
                    except Exception as e:
                        logger.warning(f"Unexpected error enhancing {result.url}: {e}")
                return result
        
        try:
            tasks = [enhance_one(result) for result in results]
            return list(await asyncio.gather(*tasks, return_exceptions=False))
        except Exception as e:
            logger.error(f"Error enhancing results: {e}")
            return results


# =============================================================================
# Utility Functions
# =============================================================================

async def cleanup_http_client() -> None:
    """Cleanup the shared HTTP client. Call on application shutdown."""
    client_manager = HTTPClientManager.get_instance()
    await client_manager.close()
