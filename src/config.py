"""Configuration management for the Deep Research Agent."""

import os
from typing import Optional
from pathlib import Path
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


class ResearchConfig(BaseModel):
    """Configuration for the research agent."""

    # API Keys
    google_api_key: str = Field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY", ""),
        description="Google/Gemini API key"
    )
    
    # Model Configuration
    model_name: str = Field(
        default="gemini-2.5-flash",
        description="Gemini model to use for research and generation"
    )
    
    summarization_model: str = Field(
        default="gemini-2.5-flash",
        description="Model for summarizing search results (faster/cheaper)"
    )
    
    # Search Configuration
    max_search_queries: int = Field(
        default=int(os.getenv("MAX_SEARCH_QUERIES", "3")),
        description="Maximum number of search queries to generate"
    )
    
    max_search_results_per_query: int = Field(
        default=int(os.getenv("MAX_SEARCH_RESULTS_PER_QUERY", "3")),
        description="Maximum results to fetch per search query"
    )
    
    max_parallel_searches: int = Field(
        default=int(os.getenv("MAX_PARALLEL_SEARCHES", "3")),
        description="Maximum number of parallel search operations"
    )
    
    # Credibility Configuration
    min_credibility_score: int = Field(
        default=int(os.getenv("MIN_CREDIBILITY_SCORE", "40")),
        description="Minimum credibility score (0-100) to filter low-quality sources"
    )
    
    # Report Configuration
    max_report_sections: int = Field(
        default=int(os.getenv("MAX_REPORT_SECTIONS", "8")),
        description="Maximum number of sections in the final report"
    )
    
    min_section_words: int = Field(
        default=200,
        description="Minimum words per section"
    )
    
    # Citation Configuration
    citation_style: str = Field(
        default=os.getenv("CITATION_STYLE", "apa"),
        description="Citation style (apa, mla, chicago, ieee)"
    )
    
    # LangSmith Configuration
    langsmith_tracing: bool = Field(
        default=os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true",
        description="Enable LangSmith tracing"
    )
    
    langsmith_project: str = Field(
        default=os.getenv("LANGCHAIN_PROJECT", "deep-research-agent"),
        description="LangSmith project name"
    )
    
    def validate_config(self) -> bool:
        """Validate that required configuration is present."""
        if not self.google_api_key:
            raise ValueError(
                "GOOGLE_API_KEY is required. Get one from https://makersuite.google.com/app/apikey"
            )
        return True


# Global configuration instance
config = ResearchConfig()

