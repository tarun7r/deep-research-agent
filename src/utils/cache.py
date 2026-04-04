"""Caching layer for research results to avoid redundant searches."""

import json
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import logging
from urllib.parse import urlsplit, urlunsplit

logger = logging.getLogger(__name__)


class ResearchCache:
    """Simple file-based cache for research results."""
    
    def __init__(self, cache_dir: Path = Path(".cache/research")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "cache.json"
        self.cache_ttl_days = 7  # Cache expires after 7 days
        
        # Load existing cache
        self._cache: Dict[str, Dict[str, Any]] = self._load_cache()
    
    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        """Load cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
                    # Filter expired entries
                    now = datetime.now()
                    valid_cache = {}
                    for key, value in cache.items():
                        cached_time = datetime.fromisoformat(value.get('timestamp', '2000-01-01'))
                        if (now - cached_time).days < self.cache_ttl_days:
                            valid_cache[key] = value
                    return valid_cache
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                return {}
        return {}
    
    def _save_cache(self):
        """Save cache to disk."""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self._cache, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def _get_key(self, topic: str) -> str:
        """Generate cache key from topic."""
        # Normalize topic (lowercase, strip whitespace)
        normalized = topic.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def get(self, topic: str) -> Optional[Dict[str, Any]]:
        """Get cached research result for a topic."""
        key = self._get_key(topic)
        if key in self._cache:
            logger.info(f"Cache hit for topic: {topic}")
            return self._cache[key].get('data')
        logger.info(f"Cache miss for topic: {topic}")
        return None
    
    def set(self, topic: str, data: Dict[str, Any]):
        """Cache research result for a topic."""
        key = self._get_key(topic)
        self._cache[key] = {
            'topic': topic,
            'data': data,
            'timestamp': datetime.now().isoformat()
        }
        self._save_cache()
        logger.info(f"Cached research result for topic: {topic}")
    
    def clear(self):
        """Clear all cached entries."""
        self._cache = {}
        self._save_cache()
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'total_entries': len(self._cache),
            'cache_dir': str(self.cache_dir),
            'cache_file': str(self.cache_file)
        }


class ToolCache:
    """Tool-level cache for web search results and extracted webpage content.

    This complements (not replaces) the topic-level `ResearchCache` used for full
    workflow outputs. It improves speed and reduces rate-limiting by caching:
    - search(query, provider, max_results) -> list[dict]
    - extract(url) -> str | None
    """

    def __init__(
        self,
        cache_dir: Path = Path(".cache/research"),
        ttl_days: int = 7,
        max_entries: int = 5000,
    ):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "tool_cache.json"
        self.ttl = timedelta(days=ttl_days)
        self.max_entries = max_entries
        self._cache: Dict[str, Dict[str, Any]] = self._load_cache()

    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        if not self.cache_file.exists():
            return {"search": {}, "content": {}}

        try:
            raw = json.loads(self.cache_file.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return {"search": {}, "content": {}}

            data = {
                "search": raw.get("search", {}) if isinstance(raw.get("search", {}), dict) else {},
                "content": raw.get("content", {}) if isinstance(raw.get("content", {}), dict) else {},
            }
            self._purge_expired_inplace(data)
            return data
        except Exception as e:
            logger.warning(f"Failed to load tool cache: {e}")
            return {"search": {}, "content": {}}

    def _save_cache(self) -> None:
        try:
            self.cache_file.write_text(json.dumps(self._cache, indent=2, default=str), encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to save tool cache: {e}")

    def _purge_expired_inplace(self, data: Dict[str, Dict[str, Any]]) -> None:
        now = datetime.now()
        for section in ("search", "content"):
            section_map = data.get(section, {})
            if not isinstance(section_map, dict):
                data[section] = {}
                continue

            expired_keys = []
            for key, value in section_map.items():
                try:
                    ts = datetime.fromisoformat(value.get("timestamp", ""))
                except Exception:
                    expired_keys.append(key)
                    continue

                if now - ts >= self.ttl:
                    expired_keys.append(key)

            for key in expired_keys:
                section_map.pop(key, None)

    def _maybe_evict(self) -> None:
        """Evict oldest items if cache grows too large (best-effort)."""
        total = len(self._cache.get("search", {})) + len(self._cache.get("content", {}))
        if total <= self.max_entries:
            return

        def collect_items(section: str):
            items = []
            for k, v in self._cache.get(section, {}).items():
                ts_str = (v or {}).get("timestamp", "")
                try:
                    ts = datetime.fromisoformat(ts_str)
                except Exception:
                    ts = datetime.fromtimestamp(0)
                items.append((ts, section, k))
            return items

        items = collect_items("search") + collect_items("content")
        items.sort(key=lambda x: x[0])  # oldest first
        to_remove = max(0, total - self.max_entries)
        for _, section, key in items[:to_remove]:
            self._cache.get(section, {}).pop(key, None)

    @staticmethod
    def _normalize_query(query: str) -> str:
        return " ".join((query or "").strip().lower().split())

    @staticmethod
    def _normalize_url(url: str) -> str:
        """Normalize URL for caching: remove fragment, normalize scheme/host."""
        try:
            parts = urlsplit((url or "").strip())
            scheme = (parts.scheme or "").lower()
            netloc = (parts.netloc or "").lower()
            path = parts.path or ""
            query = parts.query or ""
            # Drop fragment
            return urlunsplit((scheme, netloc, path, query, ""))
        except Exception:
            return (url or "").strip()

    def _search_key(self, provider: str, query: str, max_results: int) -> str:
        normalized = self._normalize_query(query)
        provider_norm = (provider or "").strip().lower() or "unknown"
        return hashlib.md5(f"{provider_norm}|{max_results}|{normalized}".encode("utf-8")).hexdigest()

    def _content_key(self, url: str) -> str:
        normalized = self._normalize_url(url)
        return hashlib.md5(normalized.encode("utf-8")).hexdigest()

    def get_search(self, provider: str, query: str, max_results: int) -> Optional[list]:
        key = self._search_key(provider, query, max_results)
        entry = self._cache.get("search", {}).get(key)
        if not entry:
            return None
        return entry.get("data")

    def set_search(self, provider: str, query: str, max_results: int, data: list) -> None:
        key = self._search_key(provider, query, max_results)
        self._cache.setdefault("search", {})[key] = {
            "provider": provider,
            "query": query,
            "max_results": max_results,
            "timestamp": datetime.now().isoformat(),
            "data": data,
        }
        self._maybe_evict()
        self._save_cache()

    def get_content(self, url: str) -> Optional[str]:
        key = self._content_key(url)
        entry = self._cache.get("content", {}).get(key)
        if not entry:
            return None
        return entry.get("data")

    def set_content(self, url: str, content: Optional[str]) -> None:
        key = self._content_key(url)
        self._cache.setdefault("content", {})[key] = {
            "url": url,
            "timestamp": datetime.now().isoformat(),
            "data": content,
        }
        self._maybe_evict()
        self._save_cache()

