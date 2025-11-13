"""State management for the Deep Research Agent."""

from typing import Annotated, List, Dict, Optional, Literal
from pydantic import BaseModel, Field
from langgraph.graph import MessagesState
from langchain_core.messages import BaseMessage


class SearchQuery(BaseModel):
    """A search query with metadata."""
    query: str = Field(description="The search query text")
    purpose: str = Field(description="Why this query is being made")
    completed: bool = Field(default=False)


class SearchResult(BaseModel):
    """A search result with content."""
    query: str = Field(description="The original query")
    title: str = Field(description="Result title")
    url: str = Field(description="Result URL")
    snippet: str = Field(description="Result snippet/summary")
    content: Optional[str] = Field(default=None, description="Full scraped content if available")


class ReportSection(BaseModel):
    """A section of the research report."""
    title: str = Field(description="Section title")
    content: str = Field(description="Section content in markdown")
    sources: List[str] = Field(default_factory=list, description="Source URLs used")


class ResearchPlan(BaseModel):
    """Research plan with queries and outline."""
    topic: str = Field(description="The research topic")
    objectives: List[str] = Field(description="Research objectives")
    search_queries: List[SearchQuery] = Field(description="Search queries to execute")
    report_outline: List[str] = Field(description="Outline of report sections")


class ResearchState(BaseModel):
    """State for the research workflow."""
    
    # User input
    research_topic: str = Field(description="The topic to research")
    
    # Planning phase
    plan: Optional[ResearchPlan] = Field(default=None, description="Research plan")
    
    # Search phase
    search_results: List[SearchResult] = Field(
        default_factory=list,
        description="All search results collected"
    )
    
    # Synthesis phase
    key_findings: List[str] = Field(
        default_factory=list,
        description="Key findings extracted from search results"
    )
    
    # Report generation phase
    report_sections: List[ReportSection] = Field(
        default_factory=list,
        description="Generated report sections"
    )
    
    final_report: Optional[str] = Field(
        default=None,
        description="Complete final report in markdown"
    )
    
    # Workflow control
    current_stage: Literal[
        "planning", "searching", "synthesizing", "reporting", "complete"
    ] = Field(default="planning")
    
    error: Optional[str] = Field(default=None, description="Error message if any")
    
    # Metadata
    iterations: int = Field(default=0, description="Number of iterations")
    
    # Quality and metrics
    quality_score: Optional[Dict] = Field(default=None, description="Report quality metrics")
    credibility_scores: List[Dict] = Field(default_factory=list, description="Source credibility scores")
    
    # LLM tracking
    llm_calls: int = Field(default=0, description="Total number of LLM API calls")
    total_input_tokens: int = Field(default=0, description="Total input tokens used")
    total_output_tokens: int = Field(default=0, description="Total output tokens generated")
    llm_call_details: List[Dict] = Field(default_factory=list, description="Details of each LLM call")
    
    class Config:
        arbitrary_types_allowed = True

