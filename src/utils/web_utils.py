"""Web search and content extraction utilities."""

import asyncio
import re
import time
from typing import List, Optional
from ddgs import DDGS
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import logging

from src.state import SearchResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def is_valid_url(url: str) -> bool:
    """Check if a URL is valid.
    
    Args:
        url: URL string to validate
        
    Returns:
        bool: True if URL is valid, False otherwise
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False


class WebSearchTool:
    """DuckDuckGo web search tool with rate limiting."""
    
    def __init__(self, max_results: int = 5):
        self.max_results = max_results
        self.last_search_time = 0
        self.min_delay = 2.0  # Minimum 2 seconds between searches
        
    def search(self, query: str) -> List[SearchResult]:
        """Perform a web search using DuckDuckGo with rate limiting.
        
        Args:
            query: Search query string
            
        Returns:
            List[SearchResult]: List of search results
        """
        try:
            # Rate limiting: wait if needed
            elapsed = time.time() - self.last_search_time
            if elapsed < self.min_delay:
                wait_time = self.min_delay - elapsed
                logger.info(f"Rate limiting: waiting {wait_time:.1f}s")
                time.sleep(wait_time)
            
            logger.info(f"Searching for: {query}")
            results = []
            
            # Use DDGS with retry logic for rate limits
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    ddgs = DDGS()
                    search_results = list(ddgs.text(
                        query,
                        max_results=self.max_results
                    ))
                    
                    for result in search_results:
                        results.append(SearchResult(
                            query=query,
                            title=result.get("title", ""),
                            url=result.get("href", ""),
                            snippet=result.get("body", "")
                        ))
                    
                    self.last_search_time = time.time()
                    logger.info(f"Found {len(results)} results for: {query}")
                    return results
                    
                except Exception as retry_error:
                    error_str = str(retry_error).lower()
                    if ("ratelimit" in error_str or "202" in error_str) and attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 5  # 5, 10, 15 seconds
                        logger.warning(f"Rate limit hit, waiting {wait_time}s before retry {attempt + 2}/{max_retries}")
                        time.sleep(wait_time)
                    else:
                        raise
            
            return results
            
        except Exception as e:
            logger.error(f"Search error for '{query}': {str(e)}")
            self.last_search_time = time.time()
            return []
    
    async def search_async(self, query: str) -> List[SearchResult]:
        """Async version of search.
        
        Args:
            query: Search query string
            
        Returns:
            List[SearchResult]: List of search results
        """
        return await asyncio.to_thread(self.search, query)


class ContentExtractor:
    """Extract and clean content from web pages."""
    
    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def extract_content(self, url: str) -> Optional[str]:
        """Extract main content from a URL.
        
        Args:
            url: URL to extract content from
            
        Returns:
            Optional[str]: Extracted content or None if extraction fails
        """
        try:
            logger.info(f"Extracting content from: {url}")
            
            response = requests.get(
                url,
                headers=self.headers,
                timeout=self.timeout,
                allow_redirects=True
            )
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                element.decompose()
            
            # Try to find main content
            main_content = None
            for selector in ['article', 'main', '[role="main"]', '.content', '#content']:
                main_content = soup.select_one(selector)
                if main_content:
                    break
            
            if not main_content:
                main_content = soup.body
            
            if main_content:
                text = main_content.get_text(separator='\n', strip=True)
                # Clean up excessive whitespace
                text = re.sub(r'\n\s*\n', '\n\n', text)
                text = re.sub(r' +', ' ', text)
                
                # Limit to reasonable length (first 5000 chars)
                text = text[:5000] if len(text) > 5000 else text
                
                logger.info(f"Extracted {len(text)} characters from {url}")
                return text
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to extract content from {url}: {str(e)}")
            return None
    
    async def extract_content_async(self, url: str) -> Optional[str]:
        """Async version of content extraction.
        
        Args:
            url: URL to extract content from
            
        Returns:
            Optional[str]: Extracted content or None if extraction fails
        """
        return await asyncio.to_thread(self.extract_content, url)
    
    async def enhance_search_results_async(
        self, 
        results: List[SearchResult],
        max_concurrent: int = 3
    ) -> List[SearchResult]:
        """Enhance search results with full content extraction (async).
        
        Args:
            results: List of search results to enhance
            max_concurrent: Maximum concurrent extraction tasks
            
        Returns:
            List[SearchResult]: Enhanced search results with content
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def enhance_one(result: SearchResult) -> SearchResult:
            async with semaphore:
                if not result.content:
                    try:
                        content = await self.extract_content_async(result.url)
                        if content:
                            result.content = content
                    except Exception as e:
                        logger.warning(f"Failed to enhance {result.url}: {str(e)}")
                return result
        
        try:
            tasks = [enhance_one(result) for result in results]
            return await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Error enhancing results: {str(e)}")
            return results

