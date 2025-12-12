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
    """Search the web for authoritative information using DuckDuckGo search engine.
    
    This tool executes web searches to find current, accurate information from diverse sources
    including academic papers, official documentation, news articles, and expert analyses.
    
    ## When to Use
    - Gathering factual information on any topic
    - Finding authoritative sources (academic, government, official docs)
    - Researching current events or recent developments
    - Verifying claims or finding supporting evidence
    - Discovering expert opinions and analyses
    
    ## Query Optimization Strategies
    
    ### For Maximum Accuracy:
    - Add "official" or "documentation" for technical topics
    - Include "research" or "study" for scientific topics
    - Add year (e.g., "2024") for time-sensitive information
    - Use site-specific queries: "site:edu" or "site:gov" for authoritative sources
    
    ### Query Formulation Best Practices:
    - Be specific: "Python async await tutorial" > "Python programming"
    - Use technical terms: "WebSocket protocol implementation" > "real-time web"
    - Include context: "Azure Speech SDK streaming architecture" > "Azure speech"
    - Combine concepts: "machine learning healthcare diagnosis 2024"
    
    ### Query Types for Comprehensive Research:
    - Definitional: "what is [topic]", "[topic] explained"
    - Technical: "[topic] architecture", "how [topic] works"
    - Comparative: "[topic] vs [alternative]", "[topic] comparison"
    - Practical: "[topic] best practices", "[topic] tutorial"
    - Current: "[topic] 2024", "latest [topic]"
    
    Args:
        query: A well-crafted search query string. Be specific and include relevant
               qualifiers. Maximum ~10 words for best results.
               
               Good examples:
               - "WebSocket vs HTTP streaming performance comparison"
               - "Azure cognitive services speech SDK documentation"
               - "transformer architecture deep learning explained 2024"
               - "site:arxiv.org large language models efficiency"
               
               Avoid:
               - Single words: "AI", "cloud", "programming"
               - Overly long queries (>15 words)
               - Ambiguous terms without context
               
        max_results: Maximum results to return (default: from config). Higher values
                     give more sources but may include less relevant results.
        
    Returns:
        List of dictionaries, each containing:
        - query (str): The search query used
        - title (str): Page title (indicates content focus)
        - url (str): Full URL (check domain for credibility)
        - snippet (str): Preview text (~150 chars, helps assess relevance)
        
    Tips:
        - Check URL domains: .edu, .gov, .org often indicate credibility
        - Review snippets before extracting full content
        - If results are poor, try rephrasing with different terms
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
    """Extract the main textual content from a webpage, removing boilerplate and noise.
    
    This tool fetches a webpage and uses intelligent content extraction to isolate the
    main article body, removing navigation, ads, sidebars, footers, and other non-content
    elements. Essential for getting the full context beyond search snippets.
    
    ## When to Use
    - After web_search identifies promising sources
    - To get full article text beyond the snippet preview
    - For in-depth analysis of specific sources
    - When you need to verify claims with full context
    - To extract technical details, examples, or data tables
    
    ## Source Prioritization Guide
    
    ### Extract First (High Value):
    - Official documentation pages (docs.*, developer.*)
    - Academic papers and research (arxiv.org, ieee.org, nature.com)
    - Government and institutional reports (.gov, .edu)
    - Detailed technical blog posts with code/examples
    - Industry whitepapers and case studies
    
    ### Extract If Needed (Medium Value):
    - News articles from reputable sources
    - Well-written tutorial and how-to guides
    - Expert commentary and analysis pieces
    - Wikipedia articles (good overviews)
    
    ### Usually Skip (Low Value):
    - Social media pages (limited extraction success)
    - Video-primary sites (YouTube, Vimeo) - no transcript extraction
    - Login-protected content
    - Heavily JavaScript-rendered single-page apps
    - Image galleries or portfolio sites
    
    ## What Gets Extracted
    - Main article/post body text
    - Headings and subheadings
    - Lists and bullet points
    - Code blocks and technical content
    - Tables (as text)
    
    ## What Gets Removed
    - Navigation menus and headers
    - Sidebar content and widgets
    - Footer links and copyright notices
    - Advertisements and promotions
    - Comment sections
    - Related article suggestions
    
    Args:
        url: Complete, valid HTTP/HTTPS URL to extract content from.
             Must be publicly accessible (no auth required).
             
             Good candidates:
             - "https://docs.microsoft.com/azure/cognitive-services/speech"
             - "https://arxiv.org/abs/2301.xxxxx"
             - "https://www.nature.com/articles/article-id"
             - "https://techblog.example.com/detailed-guide"
             
             Poor candidates:
             - "https://twitter.com/..." (social media)
             - "https://youtube.com/..." (video content)
             - URLs requiring login
        
    Returns:
        str: Extracted main text content (up to 5000 characters for efficiency).
             Content is cleaned and formatted with preserved paragraph breaks.
             
        None: If extraction fails due to:
             - Network/access errors (timeouts, 403/404)
             - Login/authentication requirements
             - JavaScript-heavy pages with no static content
             - Non-text content (PDFs, images, videos)
             
    Usage Pattern:
        1. Run web_search to find relevant URLs
        2. Review titles and domains for credibility
        3. Extract content from top 3-5 most promising sources
        4. Cross-reference extracted content for verification
    """
    try:
        content = await _extractor_impl.extract_content_async(url)
        return content
    except Exception as e:
        logger.error(f"Content extraction tool error: {str(e)}")
        return None


@tool
def analyze_research_topic(topic: str) -> Dict[str, List[str]]:
    """Decompose a research topic into structured dimensions for comprehensive coverage.
    
    This tool performs preliminary topic analysis to identify the key aspects,
    stakeholder perspectives, and essential questions that should be addressed
    in a thorough research investigation.
    
    ## Purpose
    Ensures research planning covers all important dimensions of a topic rather
    than focusing too narrowly on one aspect.
    
    ## Analysis Framework
    
    ### Aspects (What to cover)
    The fundamental dimensions or components of the topic:
    - Technical/Functional aspects (how it works)
    - Historical context (evolution, origins)
    - Current state (adoption, implementations)
    - Future outlook (trends, predictions)
    - Practical implications (real-world impact)
    
    ### Perspectives (Whose viewpoint)
    Different stakeholder or analytical lenses:
    - Technical perspective (engineers, developers)
    - Business perspective (costs, ROI, strategy)
    - User perspective (experience, benefits)
    - Ethical perspective (risks, implications)
    - Policy perspective (regulations, standards)
    
    ### Questions (What to answer)
    Core questions that comprehensive research should address:
    - Definitional: What is it?
    - Mechanistic: How does it work?
    - Evaluative: What are the pros/cons?
    - Comparative: How does it compare to alternatives?
    - Prospective: What's the future outlook?
    
    ## When to Use
    - At the start of research planning
    - When unsure how to structure research approach
    - To ensure comprehensive topic coverage
    - To generate diverse search query ideas
    
    Args:
        topic: The research topic or question to analyze.
               Can be a simple topic ("machine learning")
               or a complex question ("How does Azure Speech SDK streaming work?")
               
    Returns:
        Dictionary containing:
        
        - aspects (List[str]): Key dimensions to investigate
          Typically 3-5 core aspects of the topic
          
        - perspectives (List[str]): Stakeholder/analytical viewpoints
          Different angles from which to examine the topic
          
        - questions (List[str]): Essential questions to answer
          Core questions that research should address
          
    Usage:
        Use the returned analysis to:
        1. Generate diverse search queries covering all aspects
        2. Structure report outline to address all perspectives  
        3. Verify final report answers all essential questions
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
    """Extract specific, targeted insights from text content based on a defined focus area.
    
    This tool performs focused extraction of relevant information from raw text,
    helping to isolate specific types of insights like findings, trends, challenges,
    benefits, technical details, or statistics.
    
    ## When to Use
    - Extracting specific categories of information from article content
    - Isolating technical details or specifications
    - Finding statistics, numbers, or quantitative data
    - Identifying challenges, limitations, or criticisms
    - Pulling out benefits, advantages, or positive outcomes
    - Discovering trends, patterns, or predictions
    
    ## Effective Focus Parameters
    
    ### For Technical Research:
    - "technical specifications" - Extract specs, requirements, parameters
    - "implementation details" - How it works, architecture, components
    - "performance metrics" - Speed, accuracy, benchmarks, comparisons
    - "limitations" - Constraints, edge cases, known issues
    
    ### For Analysis:
    - "key findings" - Main conclusions and discoveries (default)
    - "trends" - Patterns, trajectories, emerging developments
    - "challenges" - Problems, obstacles, difficulties
    - "benefits" - Advantages, positive outcomes, value propositions
    - "comparisons" - How things differ, trade-offs, alternatives
    
    ### For Practical Use:
    - "best practices" - Recommended approaches, guidelines
    - "use cases" - Applications, examples, scenarios
    - "requirements" - Prerequisites, dependencies, conditions
    - "steps" - Procedures, processes, workflows
    
    Args:
        text: The text content to analyze. Can be:
              - Extracted webpage content
              - Search result snippets
              - Combined content from multiple sources
              Longer texts (>1000 chars) yield better results.
              
        focus: The type of insight to extract. Be specific for better results.
               Default: "key findings"
               
               Examples:
               - "technical architecture"
               - "performance benchmarks"
               - "security considerations"
               - "cost implications"
               - "user benefits"
        
    Returns:
        List[str]: Extracted insights matching the focus area.
                   Each insight is a complete, standalone statement.
                   Returns ["No specific insights found..."] if none match.
                   
    Tips:
        - Use specific focus terms for targeted extraction
        - Combine with multiple focus areas for comprehensive analysis
        - Review extracted insights for accuracy before including in reports
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
    """Format a source citation in a standardized academic style.
    
    This tool generates properly formatted citations for the References section
    of research reports. Supports major academic citation styles used in
    scholarly writing.
    
    ## Supported Citation Styles
    
    ### APA (American Psychological Association) - Default
    - Common in: Social sciences, psychology, education, business
    - Format: Author. (Year). Title. Retrieved from URL
    - Example: Smith, J. (2024). Machine Learning Basics. Retrieved from https://...
    
    ### MLA (Modern Language Association)
    - Common in: Humanities, literature, arts
    - Format: Author. "Title." Date. Web. Access Date. <URL>
    - Example: Smith, John. "Machine Learning Basics." Web. 15 Dec. 2024. <https://...>
    
    ### Chicago
    - Common in: History, some humanities, publishing
    - Format: Author. "Title." Accessed Date. URL.
    - Example: Smith, John. "Machine Learning Basics." Accessed December 15, 2024. https://...
    
    ### IEEE (Institute of Electrical and Electronics Engineers)
    - Common in: Engineering, computer science, technical fields
    - Format: Author, "Title," URL, accessed Date.
    - Example: J. Smith, "Machine Learning Basics," https://..., accessed December 15, 2024.
    
    ## When to Use
    - Building the References section of a report
    - Need consistent citation formatting
    - Converting URL + title into proper academic format
    
    Args:
        url: Complete URL of the source (required)
             Must be a valid HTTP/HTTPS URL
             
        title: Title of the article/page (recommended)
               Improves citation quality significantly
               If empty, citation will be URL-only
               
        style: Citation format to use (case-insensitive)
               Options: "apa" (default), "mla", "chicago", "ieee"
               Use the style appropriate for your field/audience
        
    Returns:
        str: Formatted citation string ready for inclusion in References
             Includes current date as access date where required by style
             
    Tips:
        - Always provide title when available for better citations
        - Use consistent style throughout a single report
        - APA is a safe default for most research contexts
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
    """Validate a report section against quality standards before finalizing.
    
    This tool performs comprehensive quality checks on written sections to ensure
    they meet minimum standards for length, citation usage, structure, and
    overall readability. Use BEFORE submitting final section content.
    
    ## Quality Dimensions Checked
    
    ### 1. Length Requirements
    - Minimum word count enforcement
    - Flags sections that are too short for meaningful coverage
    
    ### 2. Citation Analysis
    - Presence of inline citations [1], [2], etc.
    - Academic writing requires citations for factual claims
    
    ### 3. Structural Elements
    - Use of markdown headings for organization
    - Appropriate for sections over 300 words
    
    ## When to Use
    - After drafting any report section
    - Before returning final section content
    - To identify areas needing improvement
    - To ensure minimum quality thresholds are met
    
    ## Interpreting Results
    
    ### is_valid = True
    - Section meets all minimum requirements
    - Safe to include in final report
    
    ### is_valid = False
    - One or more critical issues found
    - Review 'issues' list for specific problems
    - Follow 'suggestions' for improvements
    - Revise section before submitting
    
    Args:
        section_text: The complete section content to validate.
                      Should be the full markdown text you plan to submit.
                      
        min_words: Minimum acceptable word count. Default: 150
                   For comprehensive sections, use 200-300
                   For brief overviews, 100-150 may suffice
        
    Returns:
        Dictionary containing:
        
        - is_valid (bool): True if ALL quality checks pass
        
        - word_count (int): Actual word count of the section
          Compare against min_words to see the gap
          
        - has_citations (bool): True if [n] citation format detected
          FALSE = Major issue for factual content
          
        - issues (List[str]): Specific problems found
          Empty list = no issues
          Examples:
            - "Section too short: 89 words (minimum: 150)"
            - "No citations found"
          
        - suggestions (List[str]): Actionable improvement recommendations
          Examples:
            - "Add more detail and supporting information"
            - "Add inline citations [1], [2] to support claims"
            - "Consider adding subheadings for better structure"
            
    Usage Pattern:
        1. Write your section content
        2. Call validate_section_quality(your_content, min_words=200)
        3. If is_valid is False, revise based on issues/suggestions
        4. Repeat until is_valid is True
        5. Submit the validated section
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
