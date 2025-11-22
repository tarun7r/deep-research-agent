"""LLM-invokable tools for research agents."""

from typing import List, Optional, Dict
from langchain_core.tools import tool
import logging
import json

from src.utils.web_utils import WebSearchTool as WebSearchImpl, ContentExtractor as ContentExtractorImpl
from src.state import SearchResult
from src.utils.citations import CitationFormatter
from src.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Initialize tool implementations with config values
_search_impl = WebSearchImpl(max_results=config.max_search_results_per_query)
_extractor_impl = ContentExtractorImpl(timeout=10)
_citation_formatter = CitationFormatter()


@tool
async def web_search(query: str, max_results: int = None) -> List[dict]:
    """Search the web for information using DuckDuckGo search engine.
    
    This tool allows you to search the internet for current information, articles, 
    research papers, news, and any publicly available content. Use this when you need 
    to gather information, verify facts, find sources, or research any topic.
    
    Best practices:
    - Use specific, targeted queries for better results
    - Include key terms and relevant keywords
    - For academic topics, include terms like "research", "study", or "paper"
    - For news, include year or recent timeframes
    - Avoid overly broad queries - be specific
    
    Args:
        query: The search query string. Should be clear and specific.
               Examples: "climate change effects 2024", "machine learning best practices",
               "renewable energy trends research"
        max_results: Maximum number of search results to return (default: from config, max: 10)
        
    Returns:
        List of dictionaries, each containing:
        - query (str): The original search query used
        - title (str): Title of the web page or article
        - url (str): Full URL to the source
        - snippet (str): Preview text/summary from the page (100-200 chars)
        
    Example:
        results = await web_search("artificial intelligence applications healthcare")
        # Returns list of relevant articles about AI in healthcare
    """
    try:
        # Use config value if not specified
        if max_results is None:
            max_results = config.max_search_results_per_query
            
        # Update max_results if different
        if _search_impl.max_results != max_results:
            _search_impl.max_results = max_results
            
        results = await _search_impl.search_async(query)
        
        # Convert SearchResult objects to dicts for LLM consumption
        return [
            {
                "query": r.query,
                "title": r.title,
                "url": r.url,
                "snippet": r.snippet
            }
            for r in results
        ]
    except Exception as e:
        logger.error(f"Web search tool error: {str(e)}")
        return []


@tool
async def extract_webpage_content(url: str) -> Optional[str]:
    """Extract and retrieve the main text content from any webpage.
    
    This tool fetches a webpage and intelligently extracts the main article or content,
    automatically removing navigation menus, advertisements, sidebars, headers, footers,
    and other non-essential elements. Use this when you need to read the full content
    of an article, blog post, research paper, or documentation page that you found
    through web search.
    
    Best practices:
    - Use this after finding interesting URLs through web_search
    - Good for articles, blog posts, documentation, research papers
    - Works best with content-rich pages (not good for video sites, social media)
    - Returns None if the page cannot be accessed or parsed
    
    Args:
        url: The complete URL of the webpage to extract content from.
             Must be a valid HTTP/HTTPS URL.
             Examples: "https://example.com/article", "https://blog.site.com/post-title"
        
    Returns:
        str or None: Extracted main text content from the page (truncated to 5000 characters
        for efficiency). Returns None if the page cannot be accessed, requires authentication,
        or if extraction fails.
        
    Example:
        content = await extract_webpage_content("https://example.com/ai-research-2024")
        if content:
            # Process the extracted article content
            print(f"Extracted {len(content)} characters")
        else:
            print("Failed to extract content from URL")
    """
    try:
        content = await _extractor_impl.extract_content_async(url)
        return content
    except Exception as e:
        logger.error(f"Content extraction tool error: {str(e)}")
        return None


@tool
def analyze_research_topic(topic: str) -> Dict[str, List[str]]:
    """Analyze a research topic to identify key aspects and dimensions to explore.
    
    This tool helps break down complex research topics into manageable components,
    identifying different aspects, perspectives, and angles that should be covered.
    
    Args:
        topic: The research topic to analyze
        
    Returns:
        Dictionary with:
        - aspects: Key aspects/dimensions of the topic
        - perspectives: Different viewpoints to consider
        - questions: Important questions to answer
        
    Example:
        result = analyze_research_topic("AI in healthcare")
        # Returns: {"aspects": ["applications", "benefits", "challenges"],
        #           "perspectives": ["patients", "doctors", "policy"],
        #           "questions": ["How effective?", "What risks?"]}
    """
    # This is a structured thinking tool for the planning agent
    # Returns structured breakdown to help with planning
    logger.info(f"Analyzing topic: {topic}")
    
    # Basic heuristic analysis
    aspects = []
    perspectives = []
    questions = []
    
    # Extract key concepts
    words = topic.lower().split()
    if "ai" in words or "artificial" in words or "intelligence" in words:
        aspects.extend(["applications", "technology", "impact"])
        perspectives.extend(["technical", "ethical", "societal"])
    
    if "healthcare" in words or "medical" in words or "health" in words:
        aspects.extend(["patient care", "diagnosis", "treatment"])
        perspectives.extend(["patients", "doctors", "researchers"])
    
    # Default structure
    if not aspects:
        aspects = ["overview", "current state", "future trends", "implications"]
    if not perspectives:
        perspectives = ["technical", "practical", "societal"]
    
    questions = [
        f"What is the current state of {topic}?",
        f"What are the key benefits and challenges?",
        f"What does the future hold for {topic}?"
    ]
    
    return {
        "aspects": aspects[:5],
        "perspectives": perspectives[:4],
        "questions": questions[:5]
    }


@tool
def extract_insights_from_text(text: str, focus: str = "key findings") -> List[str]:
    """Extract key insights from a text based on a specific focus.
    
    This tool helps synthesize information by extracting relevant insights,
    findings, or patterns from text content.
    
    Args:
        text: The text to analyze (e.g., search results, article content)
        focus: What to focus on (e.g., "key findings", "trends", "challenges")
        
    Returns:
        List of extracted insights
        
    Example:
        insights = extract_insights_from_text(article_text, "key benefits")
        # Returns: ["Benefit 1...", "Benefit 2..."]
    """
    logger.info(f"Extracting insights with focus: {focus}")
    
    # Simple extraction: split by sentences and filter
    insights = []
    sentences = text.split('. ')
    
    focus_keywords = focus.lower().split()
    for sentence in sentences[:20]:  # Limit to first 20 sentences
        sentence_lower = sentence.lower()
        # Check if sentence contains focus keywords
        if any(keyword in sentence_lower for keyword in focus_keywords):
            if len(sentence) > 20 and len(sentence) < 300:
                insights.append(sentence.strip() + '.')
    
    return insights[:10] if insights else ["No specific insights found for this focus."]


@tool
def format_citation(url: str, title: str = "", style: str = "apa") -> str:
    """Format a citation in a specific academic style.
    
    This tool formats citations for academic writing in various styles
    including APA, MLA, Chicago, and IEEE.
    
    Args:
        url: The URL of the source
        title: The title of the source (optional)
        style: Citation style ("apa", "mla", "chicago", "ieee")
        
    Returns:
        Formatted citation string
        
    Example:
        citation = format_citation(
            "https://example.com/article",
            "AI in Healthcare",
            "apa"
        )
    """
    logger.info(f"Formatting citation in {style} style")
    
    try:
        # Use the appropriate formatting method based on style
        if style.lower() == "apa":
            return _citation_formatter.format_apa(url, title)
        elif style.lower() == "mla":
            return _citation_formatter.format_mla(url, title)
        elif style.lower() == "chicago":
            return _citation_formatter.format_chicago(url, title)
        elif style.lower() == "ieee":
            return _citation_formatter.format_ieee(url, title)
        else:
            # Default to APA
            return _citation_formatter.format_apa(url, title)
    except Exception as e:
        logger.error(f"Citation formatting error: {e}")
        # Fallback to simple format
        if title:
            return f"{title}. Retrieved from {url}"
        return url


@tool
def validate_section_quality(section_text: str, min_words: int = 150) -> Dict[str, any]:
    """Validate the quality of a report section.
    
    This tool checks if a section meets quality standards including length,
    citation presence, and structure.
    
    Args:
        section_text: The section text to validate
        min_words: Minimum word count required (default: 150)
        
    Returns:
        Dictionary with:
        - is_valid: Boolean indicating if section meets quality standards
        - word_count: Actual word count
        - has_citations: Whether section includes citations
        - issues: List of quality issues found
        - suggestions: List of improvement suggestions
        
    Example:
        quality = validate_section_quality(section_text, min_words=200)
        if not quality["is_valid"]:
            print(quality["issues"])
    """
    logger.info("Validating section quality")
    
    word_count = len(section_text.split())
    has_citations = '[' in section_text and ']' in section_text
    has_headers = '#' in section_text
    
    issues = []
    suggestions = []
    
    if word_count < min_words:
        issues.append(f"Section too short: {word_count} words (minimum: {min_words})")
        suggestions.append("Add more detail and supporting information")
    
    if not has_citations:
        issues.append("No citations found")
        suggestions.append("Add inline citations [1], [2] to support claims")
    
    if not has_headers and word_count > 300:
        suggestions.append("Consider adding subheadings for better structure")
    
    is_valid = len(issues) == 0
    
    return {
        "is_valid": is_valid,
        "word_count": word_count,
        "has_citations": has_citations,
        "issues": issues,
        "suggestions": suggestions
    }


# Tool lists for different agents
research_search_tools = [
    web_search,
    extract_webpage_content
]

synthesis_tools = [
    extract_insights_from_text
]

writing_tools = [
    format_citation,
    validate_section_quality
]

planning_tools = [
    analyze_research_topic
]

# All tools combined
all_research_tools = [
    web_search,
    extract_webpage_content,
    analyze_research_topic,
    extract_insights_from_text,
    format_citation,
    validate_section_quality
]


def get_research_tools(agent_type: str = "search") -> List:
    """Get research tools for a specific agent type.
    
    Args:
        agent_type: Type of agent ("search", "synthesis", "writing", "planning", "all")
        
    Returns:
        List of LangChain tool objects for that agent
    """
    tools_map = {
        "search": research_search_tools,
        "synthesis": synthesis_tools,
        "writing": writing_tools,
        "planning": planning_tools,
        "all": all_research_tools
    }
    return tools_map.get(agent_type, research_search_tools)
