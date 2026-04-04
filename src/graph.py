"""LangGraph workflow for deep research with checkpointing and enhanced routing.

Nodes return dict updates that LangGraph automatically merges into state.
This is the recommended pattern per LangGraph documentation.
"""

import uuid
import sqlite3
from typing import Optional, Dict, Any
from pathlib import Path
from contextlib import contextmanager

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from src.state import ResearchState
from src.agents import ResearchPlanner, ResearchSearcher, ResearchSynthesizer, ReportWriter
from src.deep_research import DeepResearchSearcher
from src.agents import get_llm
from src.utils.cache import ResearchCache
from src.config import config
from src.exceptions import DeepResearchError
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Checkpoint Management
# =============================================================================

def get_checkpoint_path() -> Path:
    """Get the path for SQLite checkpoint storage."""
    cache_dir = Path(".cache/checkpoints")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "research_checkpoints.db"


def create_memory_checkpointer() -> MemorySaver:
    """Create an in-memory checkpointer (doesn't persist across restarts)."""
    return MemorySaver()


@contextmanager
def create_sqlite_checkpointer():
    """Create a SQLite checkpointer as a context manager for workflow persistence.
    
    Usage:
        with create_sqlite_checkpointer() as checkpointer:
            graph = create_research_graph(checkpointer=checkpointer)
            result = await graph.ainvoke(...)
    """
    checkpoint_path = get_checkpoint_path()
    with SqliteSaver.from_conn_string(str(checkpoint_path)) as checkpointer:
        logger.info(f"SQLite checkpointer initialized: {checkpoint_path}")
        yield checkpointer


# =============================================================================
# Graph Construction
# =============================================================================

def create_research_graph(checkpointer=None):
    """Create the research workflow graph with enhanced routing and error handling.
    
    Args:
        checkpointer: Optional checkpointer (MemorySaver or SqliteSaver) for persistence
        
    Returns:
        Compiled LangGraph workflow
    """
    
    planner = ResearchPlanner()
    searcher = ResearchSearcher()
    deep_searcher = DeepResearchSearcher(llm=get_llm(temperature=0.3, model_override=config.summarization_model))
    synthesizer = ResearchSynthesizer()
    writer = ReportWriter(citation_style=config.citation_style)
    
    workflow = StateGraph(ResearchState)
    
    workflow.add_node("plan", planner.plan)
    workflow.add_node("search", searcher.search)
    workflow.add_node("deep_search", deep_searcher.deep_search)
    workflow.add_node("synthesize", synthesizer.synthesize)
    workflow.add_node("write_report", writer.write_report)
    
    workflow.add_edge(START, "plan")
    
    def should_continue_after_plan(state: ResearchState) -> str:
        """Validate planning output and route appropriately."""
        if state.error:
            logger.error(f"Planning failed: {state.error}")
            return END
        
        if not state.plan or not state.plan.search_queries:
            logger.error("No search queries generated in plan")
            return END
            
        logger.info(f"Plan validated: {len(state.plan.search_queries)} queries")
        return "deep_search" if config.deep_research else "search"
    
    def should_continue_after_search(state: ResearchState) -> str:
        """Validate search results and route appropriately."""
        if state.error:
            logger.error(f"Search failed: {state.error}")
            return END
        
        if not state.search_results:
            logger.warning("No search results found")
            return END
        
        if len(state.search_results) < 2:
            logger.warning(f"Insufficient search results: {len(state.search_results)}")
            return END
            
        logger.info(f"Search validated: {len(state.search_results)} results")
        return "synthesize"
    
    def should_continue_after_synthesize(state: ResearchState) -> str:
        """Validate synthesis output and route appropriately."""
        if state.error:
            logger.error(f"Synthesis failed: {state.error}")
            return END
        
        if not state.key_findings:
            logger.warning("No key findings extracted")
            return END
        
        logger.info(f"Synthesis validated: {len(state.key_findings)} findings")
        return "write_report"
    
    def should_continue_after_report(state: ResearchState) -> str:
        """Validate final report and complete workflow."""
        if state.error:
            logger.error(f"Report generation failed: {state.error}")
        elif not state.final_report:
            logger.error("No report generated")
        else:
            logger.info("Report generation complete")
            
        return END
    
    workflow.add_conditional_edges(
        "plan",
        should_continue_after_plan,
        {"search": "search", "deep_search": "deep_search", END: END}
    )

    workflow.add_conditional_edges(
        "deep_search",
        should_continue_after_search,
        {"synthesize": "synthesize", END: END}
    )
    
    workflow.add_conditional_edges(
        "search",
        should_continue_after_search,
        {"synthesize": "synthesize", END: END}
    )
    
    workflow.add_conditional_edges(
        "synthesize",
        should_continue_after_synthesize,
        {"write_report": "write_report", END: END}
    )
    
    workflow.add_conditional_edges(
        "write_report",
        should_continue_after_report,
        {END: END}
    )
    
    return workflow.compile(checkpointer=checkpointer)


# =============================================================================
# Research Execution
# =============================================================================

async def run_research(
    topic: str, 
    verbose: bool = True, 
    use_cache: bool = True,
    use_checkpoints: bool = True,
    thread_id: Optional[str] = None
) -> Dict[str, Any]:
    """Run the research workflow for a given topic.
    
    Args:
        topic: Research topic
        verbose: Enable verbose logging
        use_cache: Use cached results if available
        use_checkpoints: Enable checkpoint persistence for crash recovery
        thread_id: Optional thread ID for checkpoint tracking (auto-generated if not provided)
    
    Returns:
        Complete accumulated state as a dict
    """
    logger.info(f"Starting research on: {topic}")
    
    cache = ResearchCache()
    if use_cache:
        cached_result = cache.get(topic)
        if cached_result:
            logger.info("Using cached research result")
            return cached_result
    
    initial_state = ResearchState(research_topic=topic)
    
    run_config: Dict[str, Any] = {}
    
    if use_checkpoints:
        checkpointer = create_memory_checkpointer()
        tid = thread_id or f"research-{uuid.uuid4().hex[:8]}"
        run_config["configurable"] = {"thread_id": tid}
        logger.info(f"Using thread_id: {tid} for checkpoint tracking")
    else:
        checkpointer = None
    
    graph = create_research_graph(checkpointer=checkpointer)
    
    try:
        final_state = await graph.ainvoke(initial_state, config=run_config if run_config else None)
    except Exception as e:
        logger.error(f"Research workflow failed: {e}")
        if run_config.get("configurable", {}).get("thread_id"):
            logger.info(f"Thread ID was: {run_config['configurable']['thread_id']}")
        raise
    
    if use_cache and not final_state.get("error"):
        cache.set(topic, final_state)
    
    if verbose:
        logger.info("Workflow completed")
        if final_state.get("final_report"):
            logger.info(f"Report generated: {len(final_state['final_report'])} characters")
    
    return final_state


async def run_research_with_persistence(
    topic: str, 
    verbose: bool = True, 
    use_cache: bool = True,
    thread_id: Optional[str] = None
) -> Dict[str, Any]:
    """Run the research workflow with SQLite persistence for crash recovery.
    
    This version persists checkpoints to disk, allowing resumption across restarts.
    
    Args:
        topic: Research topic
        verbose: Enable verbose logging
        use_cache: Use cached results if available
        thread_id: Optional thread ID for checkpoint tracking (auto-generated if not provided)
    
    Returns:
        Complete accumulated state as a dict
    """
    logger.info(f"Starting research on: {topic}")
    
    cache = ResearchCache()
    if use_cache:
        cached_result = cache.get(topic)
        if cached_result:
            logger.info("Using cached research result")
            return cached_result
    
    initial_state = ResearchState(research_topic=topic)
    
    tid = thread_id or f"research-{uuid.uuid4().hex[:8]}"
    run_config = {"configurable": {"thread_id": tid}}
    logger.info(f"Using thread_id: {tid} for persistent checkpoint tracking")
    
    with create_sqlite_checkpointer() as checkpointer:
        graph = create_research_graph(checkpointer=checkpointer)
        
        try:
            final_state = await graph.ainvoke(initial_state, config=run_config)
        except Exception as e:
            logger.error(f"Research workflow failed: {e}")
            logger.info(f"Workflow state saved to disk. Resume with thread_id: {tid}")
            raise
    
    if use_cache and not final_state.get("error"):
        cache.set(topic, final_state)
    
    if verbose:
        logger.info("Workflow completed")
        if final_state.get("final_report"):
            logger.info(f"Report generated: {len(final_state['final_report'])} characters")
    
    return final_state


async def resume_research(
    thread_id: str,
    additional_input: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Resume a previously interrupted research workflow from SQLite checkpoint.
    
    Args:
        thread_id: The thread ID from the interrupted workflow
        additional_input: Optional additional input to provide
        
    Returns:
        Complete accumulated state as a dict
    """
    logger.info(f"Resuming research with thread_id: {thread_id}")
    
    run_config = {"configurable": {"thread_id": thread_id}}
    
    with create_sqlite_checkpointer() as checkpointer:
        graph = create_research_graph(checkpointer=checkpointer)
        
        state = await graph.aget_state(run_config)
        if not state or not state.values:
            raise DeepResearchError(f"No checkpoint found for thread_id: {thread_id}")
        
        logger.info(f"Found checkpoint at stage: {state.values.get('current_stage', 'unknown')}")
        
        input_state = additional_input if additional_input else None
        final_state = await graph.ainvoke(input_state, config=run_config)
    
    return final_state


async def get_workflow_state(thread_id: str) -> Optional[Dict[str, Any]]:
    """Get the current state of a workflow by thread ID.
    
    Args:
        thread_id: The thread ID to look up
        
    Returns:
        Current state dict or None if not found
    """
    run_config = {"configurable": {"thread_id": thread_id}}
    
    with create_sqlite_checkpointer() as checkpointer:
        graph = create_research_graph(checkpointer=checkpointer)
        
        state = await graph.aget_state(run_config)
        if state and state.values:
            return dict(state.values)
    
    return None


def list_research_threads() -> list:
    """List all available research thread IDs from checkpoints."""
    checkpoint_path = get_checkpoint_path()
    if not checkpoint_path.exists():
        return []
    
    try:
        conn = sqlite3.connect(str(checkpoint_path))
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_ts DESC")
        threads = [row[0] for row in cursor.fetchall()]
        conn.close()
        return threads
    except Exception as e:
        logger.warning(f"Failed to list threads: {e}")
        return []
