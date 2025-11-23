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

    # Model Provider Configuration
    model_provider: str = Field(
        default=os.getenv("MODEL_PROVIDER", "gemini"),
        description="Model provider: 'gemini', 'ollama', 'openai', or 'llamacpp'"
    )
    
    # API Keys
    google_api_key: str = Field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY", ""),
        description="Google/Gemini API key (required if using Gemini)"
    )
    
    openai_api_key: str = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", ""),
        description="OpenAI API key (required if using OpenAI)"
    )
    
    # Ollama Configuration
    ollama_base_url: str = Field(
        default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        description="Ollama server URL"
    )
    
    # llama.cpp Server Configuration
    llamacpp_base_url: str = Field(
        default=os.getenv("LLAMACPP_BASE_URL", "http://localhost:8080"),
        description="llama.cpp server URL (OpenAI-compatible API)"
    )
    
    # Model Configuration
    model_name: str = Field(
        default=os.getenv("MODEL_NAME", "gemini-2.5-flash"),
        description="Model to use for research and generation"
    )
    
    summarization_model: str = Field(
        default=os.getenv("SUMMARIZATION_MODEL", "gemini-2.5-flash"),
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
        if self.model_provider == "gemini":
            if not self.google_api_key:
                raise ValueError(
                    "GEMINI_API_KEY is required when using Gemini. Get one from https://makersuite.google.com/app/apikey"
                )
        elif self.model_provider == "ollama":
            # Validate Ollama is accessible
            try:
                import requests
                response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
                if response.status_code != 200:
                    raise ValueError(f"Ollama server not accessible at {self.ollama_base_url}")
            except requests.exceptions.RequestException as e:
                raise ValueError(f"Cannot connect to Ollama server at {self.ollama_base_url}: {e}")
        elif self.model_provider == "openai":
            if not self.openai_api_key:
                raise ValueError(
                    "OPENAI_API_KEY is required when using OpenAI. Get one from https://platform.openai.com/api-keys"
                )
        elif self.model_provider == "llamacpp":
            # Validate llama.cpp server is accessible
            try:
                import requests
                response = requests.get(f"{self.llamacpp_base_url}/health", timeout=5)
                if response.status_code not in [200, 404]:  # 404 is ok, means server is running but no health endpoint
                    raise ValueError(f"llama.cpp server not accessible at {self.llamacpp_base_url}")
            except requests.exceptions.RequestException as e:
                raise ValueError(f"Cannot connect to llama.cpp server at {self.llamacpp_base_url}: {e}")
        else:
            raise ValueError(f"Invalid MODEL_PROVIDER: {self.model_provider}. Must be 'gemini', 'ollama', 'openai', or 'llamacpp'")
        
        return True


# Global configuration instance
config = ResearchConfig()

# Log configuration for debugging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info(f"Configuration loaded - MAX_SEARCH_QUERIES: {config.max_search_queries}, "
           f"MAX_SEARCH_RESULTS_PER_QUERY: {config.max_search_results_per_query}, "
           f"MAX_REPORT_SECTIONS: {config.max_report_sections}")

