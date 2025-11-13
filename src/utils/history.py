"""Research history tracking and persistence."""

import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ResearchHistory:
    """Track and manage research history."""
    
    def __init__(self, history_file: Path = Path(".cache/research_history.json")):
        self.history_file = history_file
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        self._history: List[Dict] = self._load_history()
    
    def _load_history(self) -> List[Dict]:
        """Load history from disk."""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load history: {e}")
                return []
        return []
    
    def _save_history(self):
        """Save history to disk."""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self._history, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save history: {e}")
    
    def add_research(
        self,
        topic: str,
        output_file: Optional[Path] = None,
        quality_score: Optional[Dict] = None,
        metadata: Optional[Dict] = None
    ):
        """Add a research entry to history."""
        entry = {
            'topic': topic,
            'timestamp': datetime.now().isoformat(),
            'output_file': str(output_file) if output_file else None,
            'quality_score': quality_score,
            'metadata': metadata or {}
        }
        
        # Add to beginning of list (most recent first)
        self._history.insert(0, entry)
        
        # Keep only last 100 entries
        if len(self._history) > 100:
            self._history = self._history[:100]
        
        self._save_history()
        logger.info(f"Added research to history: {topic}")
    
    def get_recent(self, limit: int = 10) -> List[Dict]:
        """Get recent research entries."""
        return self._history[:limit]
    
    def search_history(self, query: str) -> List[Dict]:
        """Search history by topic."""
        query_lower = query.lower()
        return [
            entry for entry in self._history
            if query_lower in entry.get('topic', '').lower()
        ]
    
    def get_by_topic(self, topic: str) -> Optional[Dict]:
        """Get most recent research for a topic."""
        for entry in self._history:
            if entry.get('topic', '').lower() == topic.lower():
                return entry
        return None
    
    def clear_history(self):
        """Clear all history."""
        self._history = []
        self._save_history()
        logger.info("History cleared")
    
    def get_stats(self) -> Dict:
        """Get history statistics."""
        if not self._history:
            return {
                'total_researches': 0,
                'oldest': None,
                'newest': None
            }
        
        timestamps = [datetime.fromisoformat(e['timestamp']) for e in self._history if 'timestamp' in e]
        
        return {
            'total_researches': len(self._history),
            'oldest': min(timestamps).isoformat() if timestamps else None,
            'newest': max(timestamps).isoformat() if timestamps else None
        }

