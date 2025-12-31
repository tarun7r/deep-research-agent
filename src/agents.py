"""Agent nodes for the research workflow."""

import asyncio
from typing import List
import logging

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.agents import create_agent

from src.state import ResearchState, ResearchPlan, SearchQuery, ReportSection
from src.utils.tools import get_research_tools
from src.config import config
from src.utils.credibility import CredibilityScorer
from src.utils.citations import CitationFormatter
from src.llm_tracker import estimate_tokens
from src.callbacks import (
    emit_planning_start, emit_planning_complete,
    emit_search_start, emit_search_results, 
    emit_extraction_start, emit_extraction_complete,
    emit_synthesis_start, emit_synthesis_progress, emit_synthesis_complete,
    emit_writing_start, emit_writing_section, emit_writing_complete,
    emit_error
)
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_llm(temperature: float = 0.7, model_override: str = None):
    """Get LLM instance based on configuration.
    
    Args:
        temperature: Temperature for the LLM
        model_override: Optional model name to override config.model_name
        
    Returns:
        LLM instance (ChatOllama, ChatGoogleGenerativeAI, or ChatOpenAI)
    """
    model_name = model_override or config.model_name
    
    if config.model_provider == "ollama":
        logger.info(f"Using Ollama model: {model_name}")
        return ChatOllama(
            model=model_name,
            base_url=config.ollama_base_url,
            temperature=temperature,
            num_ctx=8192,  # Context window
        )
    elif config.model_provider == "openai":
        logger.info(f"Using OpenAI model: {model_name}")
        return ChatOpenAI(
            model=model_name,
            api_key=config.openai_api_key,
            temperature=temperature
        )
    elif config.model_provider == "llamacpp":
        logger.info(f"Using llama.cpp server model: {model_name}")
        # llama.cpp server exposes OpenAI-compatible API
        return ChatOpenAI(
            model=model_name,
            base_url=f"{config.llamacpp_base_url}/v1",  # OpenAI-compatible endpoint
            api_key="not-needed",  # llama.cpp doesn't require API key
            temperature=temperature
        )
    else:  # gemini
        logger.info(f"Using Gemini model: {model_name}")
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=config.google_api_key,
            temperature=temperature
        )


class ResearchPlanner:
    """Autonomous agent responsible for planning research strategy."""
    
    def __init__(self):
        self.llm = get_llm(temperature=0.7)
        # Note: Planning agent uses LLM directly with structured output for reliability
        # Tool calling works better for search/extraction tasks
        self.max_retries = 3
        
    async def plan(self, state: ResearchState) -> dict:
        """Create a research plan with structured LLM output.
        
        Returns dict with updates that LangGraph will merge into state.
        """
        logger.info(f"Planning research for: {state.research_topic}")
        
        # Emit progress update
        await emit_planning_start(state.research_topic)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert research strategist and information architect. Your role is to create comprehensive, methodical research plans that maximize accuracy and depth of coverage.

## Your Core Responsibilities

### 1. Define SMART Research Objectives (3-5 objectives)
Create objectives that are:
- **Specific**: Target concrete aspects of the topic, not vague generalities
- **Measurable**: Can be verified as addressed in the final report
- **Achievable**: Realistically answerable through web research
- **Relevant**: Directly address the user's query and implied needs
- **Time-aware**: Consider current state, recent developments, and future outlook

### 2. Design Strategic Search Queries (up to {max_queries} queries)

**Query Diversity Matrix** - Ensure coverage across:
- **Definitional queries**: "What is [topic]" / "[topic] explained"
- **Mechanism queries**: "How does [topic] work" / "[topic] architecture"
- **Comparison queries**: "[topic] vs alternatives" / "[topic] comparison"
- **Expert/authoritative queries**: "[topic] research paper" / "[topic] official documentation"
- **Practical queries**: "[topic] best practices" / "[topic] implementation guide"
- **Trend queries**: "[topic] 2024" / "latest [topic] developments"
- **Problem/solution queries**: "[topic] challenges" / "[topic] limitations"

**Query Quality Guidelines**:
- Use specific technical terms when appropriate
- Include year markers for time-sensitive topics (e.g., "2024", "latest")
- Add domain qualifiers for targeted results (e.g., "academic", "enterprise", "tutorial")
- Avoid overly broad single-word queries
- Consider alternative phrasings and synonyms

### 3. Structure the Report Outline (up to {max_sections} sections)

Create a logical flow that:
- Starts with context/background (helps readers understand the landscape)
- Progresses from fundamentals to advanced topics
- Groups related concepts together
- Ends with practical implications, conclusions, or future outlook
- Includes a dedicated section for technical details if applicable

**Recommended Section Types**:
- Executive Summary / Overview
- Background & Context  
- Core Concepts / How It Works
- Key Features / Components / Architecture
- Benefits & Advantages
- Challenges & Limitations
- Use Cases / Applications
- Comparison with Alternatives (if relevant)
- Best Practices / Implementation Guidelines
- Future Outlook / Trends
- Conclusion & Recommendations

## Output Quality Standards
- Every search query must have a clear, distinct purpose
- No redundant or overlapping queries
- Report sections should comprehensively cover all objectives
- Consider the user's apparent expertise level when designing the plan"""),
            ("human", """Research Topic: {topic}

Analyze this topic carefully. Consider:
1. What is the user really trying to understand?
2. What are the key dimensions of this topic?
3. What authoritative sources would have the best information?
4. What technical depth is appropriate?

Create a detailed research plan in JSON format:
{{
    "topic": "the research topic (refined if needed for clarity)",
    "objectives": [
        "Specific, measurable objective 1",
        "Specific, measurable objective 2",
        ...
    ],
    "search_queries": [
        {{"query": "well-crafted search query 1", "purpose": "specific reason this query helps achieve objectives"}},
        {{"query": "well-crafted search query 2", "purpose": "specific reason this query helps achieve objectives"}},
        ...
    ],
    "report_outline": [
        "Section 1: Logical starting point",
        "Section 2: Building on Section 1",
        ...
    ]
}}

Ensure each query targets different aspects and the outline tells a coherent story.""")
        ])
        
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                chain = prompt | self.llm | JsonOutputParser()
                
                # Estimate input tokens
                input_text = f"{state.research_topic} {config.max_search_queries} {config.max_report_sections}"
                input_tokens = estimate_tokens(input_text)
                
                result = await chain.ainvoke({
                    "topic": state.research_topic,
                    "max_queries": config.max_search_queries,
                    "max_sections": config.max_report_sections
                })
                
                # Track LLM call
                duration = time.time() - start_time
                output_tokens = estimate_tokens(str(result))
                call_detail = {
                    'agent': 'ResearchPlanner',
                    'operation': 'plan',
                    'model': config.model_name,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'duration': round(duration, 2),
                    'attempt': attempt + 1
                }
                
                # Validate result structure
                if not all(key in result for key in ["topic", "objectives", "search_queries", "report_outline"]):
                    raise ValueError("Invalid plan structure returned")
                
                if not result["search_queries"]:
                    raise ValueError("No search queries generated")
                
                # Convert to ResearchPlan
                plan_data = result
                
                # Validate result structure
                if not all(key in plan_data for key in ["topic", "objectives", "search_queries", "report_outline"]):
                    raise ValueError("Invalid plan structure returned")
                
                if not plan_data["search_queries"]:
                    raise ValueError("No search queries generated")
                
                # Convert to ResearchPlan with HARD LIMITS enforced
                plan = ResearchPlan(
                    topic=plan_data["topic"],
                    objectives=plan_data["objectives"][:5],  # Max 5 objectives
                    search_queries=[
                        SearchQuery(query=sq["query"], purpose=sq["purpose"])
                        for sq in plan_data["search_queries"][:config.max_search_queries]
                    ],
                    report_outline=plan_data["report_outline"][:config.max_report_sections]
                )
                
                logger.info(f"Created plan with {len(plan.search_queries)} queries (enforced max: {config.max_search_queries})")
                logger.info(f"Report outline has {len(plan.report_outline)} sections (enforced max: {config.max_report_sections})")
                
                # Emit progress update
                await emit_planning_complete(len(plan.search_queries), len(plan.report_outline))
                
                # Return dict updates - LangGraph merges into state
                return {
                    "plan": plan,
                    "current_stage": "searching",
                    "iterations": state.iterations + 1,
                    "llm_calls": state.llm_calls + 1,
                    "total_input_tokens": state.total_input_tokens + input_tokens,
                    "total_output_tokens": state.total_output_tokens + output_tokens,
                    "llm_call_details": state.llm_call_details + [call_detail]
                }
                
            except Exception as e:
                logger.warning(f"Planning attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.max_retries - 1:
                    logger.error(f"Planning failed after {self.max_retries} attempts")
                    return {
                        "error": f"Planning failed: {str(e)}",
                        "iterations": state.iterations + 1
                    }
                else:
                    await asyncio.sleep(2 ** attempt)
        
        # Fallback if all retries exhausted
        return {
            "error": "Planning failed: Maximum retries exceeded",
            "iterations": state.iterations + 1
        }


class ResearchSearcher:
    """Autonomous agent responsible for executing research searches."""
    
    def __init__(self):
        self.llm = get_llm(temperature=0.3)
        self.tools = get_research_tools(agent_type="search")
        self.credibility_scorer = CredibilityScorer()
        self.max_retries = 3
        
    async def search(self, state: ResearchState) -> dict:
        """Autonomously execute research searches using tools.
        
        The agent will decide which searches to perform, when to extract content,
        and how to gather comprehensive information.
        
        Returns dict with search results that LangGraph will merge into state.
        """
        if not state.plan:
            await emit_error("No research plan available")
            return {"error": "No research plan available"}
        
        logger.info(f"Autonomous agent researching: {len(state.plan.search_queries)} planned queries")
        
        # Emit progress for each planned query
        total_queries = len(state.plan.search_queries)
        for i, query in enumerate(state.plan.search_queries, 1):
            await emit_search_start(query.query, i, total_queries)
        
        # Create system prompt for autonomous agent with config-based limits
        max_searches = config.max_search_queries
        max_results_per_search = config.max_search_results_per_query
        expected_total_results = max_searches * max_results_per_search
        
        system_prompt = f"""You are an elite research investigator with expertise in finding accurate, authoritative information. Your mission is to gather comprehensive, verified data from the most credible sources available.

## Your Available Tools
1. **web_search(query, max_results)**: Search the web for information
2. **extract_webpage_content(url)**: Extract full article content from a URL

## Research Protocol

### Phase 1: Strategic Searching
Execute the planned search queries systematically:
- Limit to **{max_searches} searches maximum**
- Each search returns up to **{max_results_per_search} results**
- If initial queries yield poor results, adapt with refined queries

### Phase 2: Source Evaluation & Content Extraction
For each search result, quickly assess source quality:

**HIGH-PRIORITY Sources (extract immediately):**
- Government sites (.gov, .gov.uk, .europa.eu)
- Academic institutions (.edu, .ac.uk, university domains)
- Peer-reviewed journals (nature.com, sciencedirect.com, ieee.org)
- Official documentation (docs.*, official product sites)
- Established news organizations (reuters.com, bbc.com, nytimes.com)
- Industry-recognized publications

**MEDIUM-PRIORITY Sources (extract if needed):**
- Well-known tech publications (techcrunch.com, wired.com, arstechnica.com)
- Reputable blogs with author credentials
- Company blogs from established organizations
- Wikipedia (good for overview, verify claims elsewhere)

**LOW-PRIORITY Sources (use cautiously):**
- Personal blogs without credentials
- User-generated content sites
- Sites with excessive ads or clickbait titles
- Sources without clear authorship
- Outdated content (check publication dates)

### Phase 3: Content Gathering
- Extract full content from the **top {expected_total_results} most promising URLs**
- Prioritize sources that directly address the research objectives
- Look for primary sources (original research, official docs) over secondary summaries
- Note publication dates - prefer recent content for evolving topics

## Quality Checkpoints
Before concluding, verify you have:
[x] Multiple sources confirming key facts (cross-referencing)
[x] At least some high-credibility sources in your collection
[x] Coverage across different aspects of the research objectives
[x] Both overview content and specific technical details

## Completion Signal
When you have gathered sufficient high-quality information (aim for {expected_total_results} quality sources with extracted content), respond with:

RESEARCH_COMPLETE: [Summary of what you found, including:
- Number of sources gathered
- Key themes discovered
- Any notable gaps or areas needing more research
- Confidence level in the gathered information]"""
        
        # Create autonomous agent using LangChain's create_agent
        agent_graph = create_agent(
            self.llm,
            self.tools,
            system_prompt=system_prompt
        )
        
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                
                # Prepare input
                objectives_text = "\n".join(f"- {obj}" for obj in state.plan.objectives)
                queries_text = "\n".join(
                    f"- {q.query} (Purpose: {q.purpose})" 
                    for q in state.plan.search_queries
                )
                
                # Estimate input tokens
                input_message = f"""## Research Mission Brief

### Topic Under Investigation:
{state.research_topic}

### Research Objectives (All must be addressed):
{objectives_text}

### Planned Search Queries (Execute strategically):
{queries_text}

---

### Your Mission:
1. Execute the search queries above using the web_search tool
2. Evaluate results for credibility and relevance
3. Extract full content from the most authoritative sources using extract_webpage_content
4. Ensure you gather information that addresses ALL research objectives
5. Prioritize recent, authoritative sources over older or less credible ones

### Quality Targets:
- Gather from at least {config.max_search_queries * config.max_search_results_per_query} different sources
- Extract full content from the top 5-8 most relevant pages
- Ensure coverage across all research objectives
- Include at least some academic, government, or official documentation sources if available

Begin your systematic research now. Execute searches and extract content until you have comprehensive coverage."""
                
                input_tokens = estimate_tokens(input_message)
                
                # Execute autonomous research
                result = await agent_graph.ainvoke({
                    "messages": [{"role": "user", "content": input_message}]
                })
                
                # Track LLM call (approximation - agent may make multiple calls)
                duration = time.time() - start_time
                
                # Extract messages from result
                messages = result.get('messages', [])
                output_text = ""
                if messages:
                    output_text = str(messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1]))
                
                output_tokens = estimate_tokens(output_text)
                
                # Extract search results from messages
                # We need to track tool calls and results within the messages
                search_results = []
                from src.state import SearchResult
                
                for msg in messages:
                    # Check for tool calls in message
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            if tool_call.get('name') == 'web_search':
                                # This is a search request, we'll get results in next message
                                pass
                    
                    # Check for tool responses
                    if hasattr(msg, 'name') and msg.name == 'web_search':
                        # Parse tool response
                        try:
                            content = msg.content
                            if isinstance(content, str):
                                import json
                                tool_results = json.loads(content)
                            else:
                                tool_results = content
                            
                            if isinstance(tool_results, list):
                                for item in tool_results:
                                    if isinstance(item, dict):
                                        search_results.append(SearchResult(
                                            query=item.get('query', ''),
                                            title=item.get('title', ''),
                                            url=item.get('url', ''),
                                            snippet=item.get('snippet', ''),
                                            content=None
                                        ))
                        except Exception as e:
                            logger.warning(f"Error parsing tool result: {e}")
                    
                    # Check for content extraction results
                    if hasattr(msg, 'name') and msg.name == 'extract_webpage_content':
                        try:
                            content = msg.content
                            # Find the corresponding search result and update it
                            # Note: This is a simplified approach, might need refinement
                            if search_results and content:
                                # Update the most recent search result without content
                                for sr in reversed(search_results):
                                    if not sr.content:
                                        sr.content = content
                                        break
                        except Exception as e:
                            logger.warning(f"Error updating content: {e}")
                
                logger.info(f"Autonomous agent collected {len(search_results)} results")
                
                # Calculate total extracted content
                total_extracted_chars = sum(
                    len(r.content) if r.content else 0 
                    for r in search_results
                )
                extracted_count = sum(1 for r in search_results if r.content)
                
                # Emit extraction completion
                await emit_extraction_complete(extracted_count, total_extracted_chars)
                
                if not search_results:
                    await emit_error("Agent did not collect any search results")
                    raise ValueError("Agent did not collect any search results")
            
                # Score all results first
                scored_results = self.credibility_scorer.score_search_results(search_results)
                
                # Filter by minimum credibility score
                filtered_scored = [
                    item for item in scored_results
                    if item['credibility']['score'] >= config.min_credibility_score
                ]
                
                # Extract filtered results and scores (already sorted by score, highest first)
                credibility_scores = [item['credibility'] for item in filtered_scored]
                sorted_results = [item['result'] for item in filtered_scored]
                
                logger.info(f"Filtered {len(search_results)} -> {len(sorted_results)} results (min_credibility={config.min_credibility_score})")
                
                # Mark queries as completed
                for q in state.plan.search_queries:
                    q.completed = True
                
                call_detail = {
                    'agent': 'ResearchSearcher',
                    'operation': 'autonomous_search',
                    'model': config.model_name,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'duration': round(duration, 2),
                    'results_count': len(sorted_results),
                    'original_results_count': len(search_results),
                    'min_credibility_score': config.min_credibility_score,
                    'attempt': attempt + 1
                }
                
                # Return dict updates - LangGraph merges into state
                return {
                    "search_results": sorted_results,
                    "credibility_scores": credibility_scores,
                    "current_stage": "synthesizing",
                    "iterations": state.iterations + 1,
                    "llm_calls": state.llm_calls + 1,
                    "total_input_tokens": state.total_input_tokens + input_tokens,
                    "total_output_tokens": state.total_output_tokens + output_tokens,
                    "llm_call_details": state.llm_call_details + [call_detail]
                }
                
            except Exception as e:
                logger.warning(f"Search attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.max_retries - 1:
                    logger.error(f"Search failed after {self.max_retries} attempts")
                    return {
                        "error": f"Search failed: {str(e)}",
                        "iterations": state.iterations + 1
                    }
                else:
                    await asyncio.sleep(2 ** attempt)
        
        # Fallback if all retries exhausted
        return {
            "error": "Search failed: Maximum retries exceeded",
            "iterations": state.iterations + 1
        }


class ResearchSynthesizer:
    """Autonomous agent responsible for synthesizing research findings."""
    
    def __init__(self):
        self.llm = get_llm(temperature=0.3, model_override=config.summarization_model)
        self.tools = get_research_tools(agent_type="synthesis")
        self.max_retries = 3
        
    async def synthesize(self, state: ResearchState) -> dict:
        """Autonomously synthesize key findings using tools and reasoning.
        
        Returns dict with key findings that LangGraph will merge into state.
        """
        logger.info(f"Synthesizing findings from {len(state.search_results)} results")
        
        if not state.search_results:
            await emit_error("No search results to synthesize")
            return {"error": "No search results to synthesize"}
        
        # Emit synthesis start
        await emit_synthesis_start(len(state.search_results))
        
        # Create system prompt for autonomous synthesis agent
        system_prompt = """You are a senior research analyst specializing in synthesizing complex information into accurate, actionable insights. Your task is to analyze search results and extract verified, well-supported findings.

## Your Available Tools
- **extract_insights_from_text(text, focus)**: Extract specific insights from text content

## Source Credibility Framework

Each source has a credibility rating. Apply this hierarchy strictly:

### HIGH Credibility (Score >=70) - Primary Sources
- Government and institutional sources
- Peer-reviewed research and academic papers
- Official documentation and specifications
- Established news organizations with editorial standards
=> **TRUST**: Use as primary basis for findings

### MEDIUM Credibility (Score 40-69) - Supporting Sources
- Industry publications and tech blogs
- Expert commentary and analysis
- Well-maintained wikis and documentation
=> **VERIFY**: Cross-reference with HIGH sources; use to add context

### LOW Credibility (Score <40) - Supplementary Only
- Personal blogs, forums, user comments
- Sources without clear authorship
- Outdated or unverified content
=> **CAUTION**: Only use if corroborated by higher-credibility sources

## Synthesis Methodology

### Step 1: Identify Core Facts
- What claims appear in multiple HIGH-credibility sources?
- What are the foundational facts that most sources agree on?
- Extract specific data points: numbers, dates, names, technical specifications

### Step 2: Detect and Resolve Conflicts
When sources contradict each other:
1. Check credibility scores - trust higher-rated sources
2. Check recency - newer information may supersede older
3. Check specificity - primary sources trump secondary summaries
4. If unresolvable, note the disagreement in findings

### Step 3: Synthesize Key Findings
For each finding, ensure:
- **Accuracy**: Only include information that appears in the sources
- **Attribution**: Note which source numbers support the finding [1], [2], etc.
- **Specificity**: Include concrete details, not vague generalities
- **Balance**: Present multiple perspectives if sources differ

### Step 4: Quality Control
Before finalizing, verify:
[x] No claims are made without source support
[x] HIGH-credibility sources are prioritized
[x] Contradictions are acknowledged, not ignored
[x] Findings directly address research objectives
[x] Technical accuracy is maintained (don't oversimplify incorrectly)

## Output Format

Return findings as a JSON array of strings. Each finding should:
- Be a complete, standalone insight
- Include source references where applicable
- Be specific enough to be useful (avoid generic statements)
- Focus on facts over opinions (unless opinion is from recognized experts)

Example format:
[
    "Finding 1: [Specific fact or insight] - supported by sources [1], [3]",
    "Finding 2: [Technical detail with specifics] - per official documentation [2]",
    "Finding 3: [Trend or development] - noted across multiple industry sources [4], [5], [6]"
]

## Anti-Hallucination Rules
DO NOT invent statistics, dates, or specifics not in sources
DO NOT make claims beyond what sources support
DO NOT present speculation as fact
DO NOT ignore source credibility ratings
DO say "sources indicate" or "according to [source]" for less certain claims
DO note when information is limited or conflicting"""
        
        # Create autonomous synthesis agent
        agent_graph = create_agent(
            self.llm,
            self.tools,
            system_prompt=system_prompt
        )
        
        # Progressive truncation strategy
        max_results = 20
        
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                
                # Adjust result count based on attempt
                current_max = max(5, max_results - (attempt * 5))
                
                # Prepare search results text with credibility information
                results_to_use = state.search_results[:current_max]
                credibility_scores_to_use = state.credibility_scores[:current_max] if state.credibility_scores else []
                
                results_text = "\n\n".join([
                    f"[{i+1}] {r.title}\n"
                    f"URL: {r.url}\n"
                    f"Credibility: {cred.get('level', 'unknown').upper()} (Score: {cred.get('score', 'N/A')}/100) - {', '.join(cred.get('factors', []))}\n"
                    f"Snippet: {r.snippet}\n" +
                    (f"Content: {r.content[:300]}..." if r.content else "")
                    for i, (r, cred) in enumerate(zip(results_to_use, credibility_scores_to_use))
                ])
                
                # If credibility scores don't match (shouldn't happen, but handle gracefully)
                if len(results_to_use) != len(credibility_scores_to_use):
                    # Fallback: format without credibility if mismatch
                    results_text = "\n\n".join([
                        f"[{i+1}] {r.title}\nURL: {r.url}\nSnippet: {r.snippet}\n" +
                        (f"Content: {r.content[:300]}..." if r.content else "")
                        for i, r in enumerate(results_to_use)
                    ])
                
                # Prepare input message for the autonomous agent
                input_message = f"""## Research Synthesis Task

### Topic: {state.research_topic}

### Your Mission:
Analyze the search results below and extract the most important, accurate, and well-supported findings.

---

### Search Results with Credibility Scores:
{results_text}

---

### Synthesis Instructions:

1. **Extract Key Facts**: Identify the core factual claims across sources
2. **Cross-Reference**: Note which findings are supported by multiple sources
3. **Resolve Conflicts**: When sources disagree, trust higher-credibility sources
4. **Maintain Specificity**: Include specific details, numbers, and technical information
5. **Note Limitations**: Flag areas where information is sparse or contradictory

### Output Requirements:
Return a JSON array of 10-15 key findings. Each finding should:
- Be a complete, specific statement (not vague generalizations)
- Reference source numbers when citing facts: "...according to [1]" or "...per [3], [5]"
- Focus on facts that directly address the research topic
- Prioritize findings from HIGH-credibility sources

Example format:
[
    "The technology uses [specific mechanism] to achieve [specific outcome], enabling [specific capability] [1]",
    "According to official documentation [2], the key components include: [list specific items]",
    "Industry adoption has grown to [specific metric], with major deployments at [specific examples] [3], [5]",
    "Experts note challenges including [specific challenge 1] and [specific challenge 2] [4]"
]

Analyze the sources now and extract your findings:"""
                
                # Estimate input tokens
                input_tokens = estimate_tokens(input_message)
                
                # Execute autonomous synthesis
                result = await agent_graph.ainvoke({
                    "messages": [{"role": "user", "content": input_message}]
                })
                
                # Track LLM call
                duration = time.time() - start_time
                
                # Extract final response
                messages = result.get('messages', [])
                output_text = ""
                if messages:
                    last_msg = messages[-1]
                    output_text = str(last_msg.content if hasattr(last_msg, 'content') else str(last_msg))
                
                output_tokens = estimate_tokens(output_text)
                
                call_detail = {
                    'agent': 'ResearchSynthesizer',
                    'operation': 'autonomous_synthesis',
                    'model': config.summarization_model,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'duration': round(duration, 2),
                    'attempt': attempt + 1
                }
                
                # Parse the JSON response
                import json
                import re
                
                # Try to extract JSON array from the response
                json_match = re.search(r'\[(.*?)\]', output_text, re.DOTALL)
                
                key_findings = []
                if json_match:
                    try:
                        findings = json.loads(json_match.group(0))
                        if isinstance(findings, list):
                            key_findings = [
                                str(f)  # Convert all items to strings (handles int, dict, etc.)
                                for f in findings
                            ]
                        else:
                            key_findings = [str(findings)]
                    except json.JSONDecodeError:
                        pass
                
                # If JSON parsing failed or empty, use fallback extraction
                if not key_findings:
                    # Look for bullet points or numbered items
                    lines = output_text.split('\n')
                    for line in lines:
                        line = line.strip().lstrip('-').lstrip('*').lstrip('>').strip()
                        # Remove numbering like "1.", "2.", etc.
                        line = re.sub(r'^\d+\.\s*', '', line)
                        if len(line) > 30 and not line.startswith('[') and not line.startswith(']'):
                            key_findings.append(line)
                    
                    # Limit to reasonable number
                    key_findings = key_findings[:15]
                
                # If still empty, create basic findings from search results
                if not key_findings and state.search_results:
                    logger.warning("Agent produced no findings, creating basic ones from results")
                    key_findings = [
                        f"{r.title}: {r.snippet[:100]}..."
                        for r in state.search_results[:10]
                        if r.snippet
                    ]
                
                logger.info(f"Extracted {len(key_findings)} key findings")
                
                # Emit synthesis completion
                await emit_synthesis_complete(len(key_findings))
                
                # Return dict updates - LangGraph merges into state
                return {
                    "key_findings": key_findings,
                    "current_stage": "reporting",
                    "iterations": state.iterations + 1,
                    "llm_calls": state.llm_calls + 1,
                    "total_input_tokens": state.total_input_tokens + input_tokens,
                    "total_output_tokens": state.total_output_tokens + output_tokens,
                    "llm_call_details": state.llm_call_details + [call_detail]
                }
                
            except Exception as e:
                logger.warning(f"Synthesis attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.max_retries - 1:
                    logger.error(f"Synthesis failed after {self.max_retries} attempts")
                    return {
                        "error": f"Synthesis failed: {str(e)}",
                        "iterations": state.iterations + 1
                    }
                else:
                    await asyncio.sleep(2 ** attempt)
        
        # Fallback if all retries exhausted
        return {
            "error": "Synthesis failed: Maximum retries exceeded",
            "iterations": state.iterations + 1
        }


class ReportWriter:
    """Autonomous agent responsible for writing research reports."""
    
    def __init__(self, citation_style: str = 'apa'):
        self.llm = get_llm(temperature=0.7)
        self.tools = get_research_tools(agent_type="writing")
        self.max_retries = 3
        self.citation_style = citation_style
        self.citation_formatter = CitationFormatter()
        
    async def write_report(self, state: ResearchState) -> dict:
        """Write the final research report with validation and retry.
        
        Returns dict with report data that LangGraph will merge into state.
        """
        logger.info("Writing final report")
        
        if not state.plan or not state.key_findings:
            await emit_error("Insufficient data for report generation")
            return {"error": "Insufficient data for report generation"}
        
        # Emit writing start
        await emit_writing_start(len(state.plan.report_outline))
        
        # Track total LLM calls for report generation
        report_llm_calls = 0
        report_input_tokens = 0
        report_output_tokens = 0
        report_call_details = []
        
        for attempt in range(self.max_retries):
            try:
                # Generate each section with retry
                report_sections = []
                total_sections = len(state.plan.report_outline)
                
                for section_idx, section_title in enumerate(state.plan.report_outline, 1):
                    # Emit progress for each section
                    await emit_writing_section(section_title, section_idx, total_sections)
                    
                    section, section_tokens = await self._write_section(
                        state.research_topic,
                        section_title,
                        state.key_findings,
                        state.search_results
                    )
                    if section:
                        report_sections.append(section)
                        if section_tokens:
                            report_llm_calls += 1
                            report_input_tokens += section_tokens['input_tokens']
                            report_output_tokens += section_tokens['output_tokens']
                            report_call_details.append(section_tokens)
                
                # Validate minimum quality
                if not report_sections:
                    raise ValueError("No report sections generated")
                
                # Create temporary state for compilation
                temp_state = ResearchState(
                    research_topic=state.research_topic,
                    plan=state.plan,
                    report_sections=report_sections
                )
                
                # Compile final report
                final_report = self._compile_report(temp_state)
                
                # Format citations in specified style
                if state.search_results:
                    final_report = self.citation_formatter.update_report_citations(
                        final_report,
                        style=self.citation_style,
                        search_results=state.search_results
                    )
                
                # Add credibility information to report if available
                if state.credibility_scores:
                    high_cred_sources = [
                        i+1 for i, score in enumerate(state.credibility_scores)
                        if score.get('level') == 'high'
                    ]
                    if high_cred_sources:
                        final_report += f"\n\n---\n\n**Note:** {len(high_cred_sources)} high-credibility sources were prioritized in this research."
                
                # Validate report length
                if len(final_report) < 500:
                    raise ValueError("Report too short - insufficient content")
                
                logger.info(f"Report generation complete: {len(final_report)} chars")
                
                # Emit writing completion
                await emit_writing_complete(len(final_report))
                
                # Return dict updates - LangGraph merges into state
                return {
                    "report_sections": report_sections,
                    "final_report": final_report,
                    "current_stage": "complete",
                    "iterations": state.iterations + 1,
                    "llm_calls": state.llm_calls + report_llm_calls,
                    "total_input_tokens": state.total_input_tokens + report_input_tokens,
                    "total_output_tokens": state.total_output_tokens + report_output_tokens,
                    "llm_call_details": state.llm_call_details + report_call_details
                }
                
            except Exception as e:
                logger.warning(f"Report attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.max_retries - 1:
                    logger.error(f"Report generation failed after {self.max_retries} attempts")
                    return {
                        "error": f"Report writing failed: {str(e)}",
                        "iterations": state.iterations + 1
                    }
                else:
                    await asyncio.sleep(2 ** attempt)
        
        # Fallback if all retries exhausted
        return {
            "error": "Report generation failed: Maximum retries exceeded",
            "iterations": state.iterations + 1
        }
    
    async def _write_section(
        self,
        topic: str,
        section_title: str,
        findings: List[str],
        search_results: List
    ) -> tuple:
        """Autonomously write a single report section using tools."""
        logger.info(f"Writing section: {section_title}")
        
        # Create system prompt for section writing
        system_prompt = f"""You are a distinguished research writer and subject matter expert. Your task is to write authoritative, accurate, and well-structured report sections that inform and educate readers.

## Writing Standards

### Content Quality Requirements
1. **Minimum Length**: {config.min_section_words} words - ensure you write comprehensive, detailed content
2. **Factual Accuracy**: Every claim must be grounded in the provided findings
3. **Proper Citations**: Use inline citations [1], [2], etc. for all factual claims
4. **Balanced Perspective**: Present multiple viewpoints when they exist
5. **Technical Precision**: Use correct terminology; don't oversimplify incorrectly

### Structure & Formatting (Markdown)
- Use **bold** for key terms and important concepts
- Use bullet points or numbered lists for multiple items
- Use subheadings (### or ####) to organize complex sections
- Include specific examples, data points, or case studies when available
- Maintain logical flow from one paragraph to the next

### Writing Style Guidelines
- **Tone**: Professional, authoritative, but accessible
- **Voice**: Third-person academic style (avoid "I", "we", "you")
- **Clarity**: Explain complex concepts clearly; define technical terms
- **Conciseness**: Every sentence should add value; avoid filler
- **Precision**: Use specific language; avoid vague qualifiers like "very" or "many"

## Critical Accuracy Rules

### DO
- Base all claims on the provided key findings
- Cite sources for factual statements: "According to [1]..." or "Research indicates [2]..."
- Distinguish between established facts and emerging trends
- Note limitations or caveats when relevant
- Use specific numbers, dates, and names from sources
- Acknowledge when evidence is limited: "Available data suggests..."

### DO NOT
- Invent statistics, percentages, or specific numbers not in findings
- Make claims that go beyond the provided information
- Present opinions as facts without attribution
- Ignore contradictions between sources
- Use placeholder text or generic filler content
- Oversimplify to the point of inaccuracy

## Section Writing Process

1. **Analyze**: Review the findings relevant to this section's topic
2. **Outline**: Mentally structure the key points to cover
3. **Draft**: Write comprehensive, detailed content with proper citations
4. **Refine**: Ensure logical flow, accuracy, and sufficient depth

## CRITICAL: Output Format

You MUST write the section content directly as your response. DO NOT use tools or provide meta-commentary.
Your entire response should be the section content in markdown format.

Start with the content immediately (the section title will be added automatically). 
Ensure proper spacing between paragraphs and aim for AT LEAST {config.min_section_words} words.

Example structure:
```
[Opening paragraph introducing the section topic]

[Main content paragraph with specific details and citations [1]]

### [Subheading if needed]

[Additional content with more citations [2], [3]]

[Concluding paragraph summarizing key points]
```"""
        
        # Create a simple chain without tools for cleaner content generation
        # Tools were causing issues with content extraction
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        
        try:
            start_time = time.time()
            
            # Prepare input message with source context
            sources_context = ""
            if search_results:
                sources_context = "\n\nAvailable Sources for Citation:\n" + "\n".join(
                    f"[{i+1}] {r.title} ({r.url})"
                    for i, r in enumerate(search_results[:15])  # Top 15 sources
                )
            
            input_message = f"""## Assignment: Write Report Section

**Research Topic**: {topic}
**Section Title**: {section_title}
**Minimum Word Count**: {config.min_section_words} words

---

### Key Findings to Incorporate:
{chr(10).join(f"- {f}" for f in findings)}

{sources_context}

---

### Instructions:
1. Write a comprehensive section that covers the topic "{section_title}" thoroughly
2. Incorporate the key findings above, adding context and explanation
3. Use inline citations [1], [2], etc. when referencing specific facts from sources
4. Maintain academic rigor while being accessible to general readers
5. Use markdown formatting for structure (bold, lists, subheadings as needed)
6. Ensure your response is AT LEAST {config.min_section_words} words

IMPORTANT: Your response should ONLY contain the section content in markdown format. 
Do NOT use any tools. Do NOT provide meta-commentary. Just write the section content directly.

Write the section content now:"""
            
            # Estimate input tokens
            input_tokens = estimate_tokens(input_message)
            
            # Execute section writing using simple chain
            chain = prompt | self.llm | StrOutputParser()
            content = await chain.ainvoke({"input": input_message})
            
            # Content should now be a clean string
            if not isinstance(content, str):
                content = str(content)
            
            # Track LLM call
            duration = time.time() - start_time
            output_tokens = estimate_tokens(content)
            call_detail = {
                'agent': 'ReportWriter',
                'operation': f'write_section_{section_title[:30]}',
                'model': config.model_name,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'duration': round(duration, 2)
            }
            
            # Validate content is not empty
            if not content or len(content.strip()) < 50:
                logger.warning(f"Section '{section_title}' generated insufficient content: {len(content)} chars")
                # Try to generate a basic section from findings if agent failed
                if findings:
                    logger.info(f"Creating fallback content for section '{section_title}'")
                    content = f"\n\n{chr(10).join(findings[:3])}\n\n"
                else:
                    logger.error(f"Cannot create section '{section_title}' - no content and no findings")
                    return None, None
            
            # Extract cited sources
            import re
            citations = re.findall(r'\[(\d+)\]', content)
            source_urls = []
            for cite_num in set(citations):
                idx = int(cite_num) - 1
                if 0 <= idx < len(search_results):
                    source_urls.append(search_results[idx].url)
            
            section = ReportSection(
                title=section_title,
                content=content,
                sources=source_urls
            )
            
            logger.info(f"Successfully wrote section '{section_title}': {len(content)} chars")
            return section, call_detail
            
        except Exception as e:
            logger.error(f"Error writing section '{section_title}': {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None, None
    
    def _compile_report(self, state: ResearchState) -> str:
        """Compile all sections into final report."""
        # Count actual sources from search results
        search_results = getattr(state, 'search_results', []) or []
        report_sections = getattr(state, 'report_sections', []) or []
        
        # Get unique URLs from search results
        unique_sources = set()
        for result in search_results:
            if hasattr(result, 'url') and result.url:
                unique_sources.add(result.url)
        
        # Also collect from report sections if they have sources
        for section in report_sections:
            if hasattr(section, 'sources'):
                unique_sources.update(section.sources)
        
        source_count = len(unique_sources) if unique_sources else len(search_results)
        
        report_parts = [
            f"# {state.research_topic}\n",
            f"**Deep Research Report**\n",
            f"\n## Executive Summary\n",
            f"This report provides a comprehensive analysis of {state.research_topic}. ",
            f"The research was conducted across **{source_count} sources** ",
            f"and synthesized into **{len(report_sections)} key sections**.\n",
            f"\n## Research Objectives\n"
        ]
        
        if state.plan and hasattr(state.plan, 'objectives'):
            for i, obj in enumerate(state.plan.objectives, 1):
                report_parts.append(f"{i}. {obj}\n")
        
        report_parts.append("\n---\n")
        
        # Add all sections
        has_references_section = False
        for section in report_sections:
            # Check if content already starts with the title as a heading
            content = section.content.strip()
            
            # Check if this section contains References
            if "## References" in content or section.title.lower() == "references":
                has_references_section = True
            
            if content.startswith(f"## {section.title}"):
                # Content already has heading, use as-is
                report_parts.append(f"\n{content}\n\n")
            else:
                # Add heading before content
                report_parts.append(f"\n## {section.title}\n\n")
                report_parts.append(content)
                report_parts.append("\n")
        
        # Only add references if not already present in sections
        if not has_references_section:
            # Add references from search results
            report_parts.append("\n---\n\n## References\n\n")
        
        # Build a list of (url, title) tuples from search results
        source_info = []
        seen_urls = set()
        
        for result in search_results:
            if hasattr(result, 'url') and result.url and result.url not in seen_urls:
                seen_urls.add(result.url)
                title = getattr(result, 'title', '')
                source_info.append((result.url, title))
        
        # Add sources from sections if available (if not already included)
        for section in report_sections:
            if hasattr(section, 'sources'):
                for url in section.sources:
                    if url not in seen_urls:
                        seen_urls.add(url)
                        source_info.append((url, ''))
        
        # Add formatted references (only once, outside the loop)
        if not has_references_section:
            if source_info:
                from src.utils.citations import CitationFormatter
                formatter = CitationFormatter()
                for i, (url, title) in enumerate(source_info[:30], 1):  # Top 30 sources
                    # Format citation in APA style
                    citation = formatter.format_apa(url, title)
                    report_parts.append(f"{i}. {citation}\n")
            else:
                report_parts.append("*No sources were available for this research.*\n")
        
        return "".join(report_parts)

