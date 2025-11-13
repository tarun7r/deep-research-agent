"""Caching layer for research results to avoid redundant searches."""

import json
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import logging

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

