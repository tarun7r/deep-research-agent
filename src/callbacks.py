"""Callback system for real-time progress updates in the research workflow."""

import asyncio
from typing import Callable, Optional, Dict, Any, List
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ResearchStage(Enum):
    """Research workflow stages."""
    INITIALIZING = "initializing"
    PLANNING = "planning"
    SEARCHING = "searching"
    EXTRACTING = "extracting"
    SYNTHESIZING = "synthesizing"
    WRITING = "writing"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class ProgressUpdate:
    """A progress update event."""
    stage: ResearchStage
    message: str
    details: Optional[str] = None
    progress_pct: Optional[float] = None  # 0-100
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class ProgressCallback:
    """Manages progress callbacks for research workflow."""
    
    _instance: Optional['ProgressCallback'] = None
    _callbacks: List[Callable[[ProgressUpdate], None]] = []
    _async_callbacks: List[Callable[[ProgressUpdate], Any]] = []
    _updates: List[ProgressUpdate] = []
    _current_stage: ResearchStage = ResearchStage.INITIALIZING
    
    def __new__(cls):
        """Singleton pattern to ensure one global callback manager."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._callbacks = []
            cls._instance._async_callbacks = []
            cls._instance._updates = []
            cls._instance._current_stage = ResearchStage.INITIALIZING
        return cls._instance
    
    def reset(self):
        """Reset state for a new research session."""
        self._updates = []
        self._current_stage = ResearchStage.INITIALIZING
    
    def register(self, callback: Callable[[ProgressUpdate], None]):
        """Register a synchronous callback function."""
        if callback not in self._callbacks:
            self._callbacks.append(callback)
    
    def register_async(self, callback: Callable[[ProgressUpdate], Any]):
        """Register an async callback function."""
        if callback not in self._async_callbacks:
            self._async_callbacks.append(callback)
    
    def unregister(self, callback: Callable):
        """Unregister a callback function."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
        if callback in self._async_callbacks:
            self._async_callbacks.remove(callback)
    
    def clear_callbacks(self):
        """Clear all registered callbacks."""
        self._callbacks = []
        self._async_callbacks = []
    
    async def emit(self, update: ProgressUpdate):
        """Emit a progress update to all registered callbacks."""
        self._current_stage = update.stage
        self._updates.append(update)
        
        # Log the update
        logger.info(f"[{update.stage.value}] {update.message}" + 
                   (f" - {update.details}" if update.details else ""))
        
        # Call sync callbacks
        for callback in self._callbacks:
            try:
                callback(update)
            except Exception as e:
                logger.error(f"Error in sync callback: {e}")
        
        # Call async callbacks
        for callback in self._async_callbacks:
            try:
                await callback(update)
            except Exception as e:
                logger.error(f"Error in async callback: {e}")
    
    @property
    def current_stage(self) -> ResearchStage:
        return self._current_stage
    
    @property
    def updates(self) -> List[ProgressUpdate]:
        return self._updates.copy()


# Global progress callback instance
progress_callback = ProgressCallback()


# Convenience functions for emitting progress
async def emit_progress(
    stage: ResearchStage,
    message: str,
    details: Optional[str] = None,
    progress_pct: Optional[float] = None,
    **metadata
):
    """Emit a progress update."""
    update = ProgressUpdate(
        stage=stage,
        message=message,
        details=details,
        progress_pct=progress_pct,
        metadata=metadata
    )
    await progress_callback.emit(update)


async def emit_planning_start(topic: str):
    """Emit planning stage start."""
    await emit_progress(
        ResearchStage.PLANNING,
        "Creating research plan",
        f"Topic: {topic}",
        progress_pct=5
    )


async def emit_planning_complete(num_queries: int, num_sections: int):
    """Emit planning stage completion."""
    await emit_progress(
        ResearchStage.PLANNING,
        "Research plan created",
        f"{num_queries} search queries, {num_sections} report sections planned",
        progress_pct=15
    )


async def emit_search_start(query: str, query_num: int, total_queries: int):
    """Emit search start."""
    base_progress = 15
    search_progress_range = 35  # 15% to 50%
    progress = base_progress + (query_num / total_queries) * search_progress_range
    
    await emit_progress(
        ResearchStage.SEARCHING,
        f"Searching ({query_num}/{total_queries})",
        f"Query: {query[:60]}..." if len(query) > 60 else f"Query: {query}",
        progress_pct=progress
    )


async def emit_search_results(num_results: int, query_num: int, total_queries: int):
    """Emit search results found."""
    base_progress = 15
    search_progress_range = 35
    progress = base_progress + ((query_num + 0.5) / total_queries) * search_progress_range
    
    await emit_progress(
        ResearchStage.SEARCHING,
        f"Found {num_results} results",
        f"Query {query_num}/{total_queries} complete",
        progress_pct=progress
    )


async def emit_extraction_start(url: str, current: int, total: int):
    """Emit content extraction start."""
    base_progress = 50
    extract_progress_range = 15  # 50% to 65%
    progress = base_progress + (current / total) * extract_progress_range
    
    # Extract domain from URL for cleaner display
    try:
        from urllib.parse import urlparse
        domain = urlparse(url).netloc
    except:
        domain = url[:40]
    
    await emit_progress(
        ResearchStage.EXTRACTING,
        f"Extracting content ({current}/{total})",
        f"Source: {domain}",
        progress_pct=progress
    )


async def emit_extraction_complete(num_extracted: int, total_chars: int):
    """Emit extraction completion."""
    await emit_progress(
        ResearchStage.EXTRACTING,
        f"Content extraction complete",
        f"{num_extracted} pages, {total_chars:,} characters extracted",
        progress_pct=65
    )


async def emit_synthesis_start(num_sources: int):
    """Emit synthesis stage start."""
    await emit_progress(
        ResearchStage.SYNTHESIZING,
        "Analyzing sources",
        f"Synthesizing {num_sources} sources into key findings",
        progress_pct=68
    )


async def emit_synthesis_progress(message: str):
    """Emit synthesis progress."""
    await emit_progress(
        ResearchStage.SYNTHESIZING,
        message,
        progress_pct=72
    )


async def emit_synthesis_complete(num_findings: int):
    """Emit synthesis completion."""
    await emit_progress(
        ResearchStage.SYNTHESIZING,
        "Synthesis complete",
        f"Extracted {num_findings} key findings",
        progress_pct=78
    )


async def emit_writing_start(num_sections: int):
    """Emit writing stage start."""
    await emit_progress(
        ResearchStage.WRITING,
        "Writing report",
        f"Generating {num_sections} sections",
        progress_pct=80
    )


async def emit_writing_section(section_title: str, section_num: int, total_sections: int):
    """Emit section writing progress."""
    base_progress = 80
    writing_progress_range = 18  # 80% to 98%
    progress = base_progress + (section_num / total_sections) * writing_progress_range
    
    await emit_progress(
        ResearchStage.WRITING,
        f"Writing section ({section_num}/{total_sections})",
        f"Section: {section_title[:50]}..." if len(section_title) > 50 else f"Section: {section_title}",
        progress_pct=progress
    )


async def emit_writing_complete(report_length: int):
    """Emit writing completion."""
    await emit_progress(
        ResearchStage.WRITING,
        "Report writing complete",
        f"Generated {report_length:,} character report",
        progress_pct=98
    )


async def emit_complete(topic: str, sources: int, findings: int):
    """Emit research completion."""
    await emit_progress(
        ResearchStage.COMPLETE,
        "Research complete!",
        f"{sources} sources analyzed, {findings} insights extracted",
        progress_pct=100
    )


async def emit_error(error_message: str):
    """Emit error."""
    await emit_progress(
        ResearchStage.ERROR,
        "Error occurred",
        error_message,
        progress_pct=None
    )

