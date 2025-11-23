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
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert research planner. Given a research topic, create a comprehensive research plan.

Your plan should include:
1. Clear research objectives (3-5 objectives)
2. Strategic search queries to gather information (up to {max_queries} queries)
   - Create diverse queries covering different aspects
   - Include both broad overview queries and specific deep-dive queries
   - Consider current trends, historical context, and future implications
3. An outline for the final report (up to {max_sections} sections)

Be strategic and thorough. The search queries will be used by an autonomous agent that can:
- Execute web searches
- Extract full content from web pages
- Gather information iteratively

Design your queries to maximize information gathering."""),
            ("human", """Research Topic: {topic}

Create a detailed research plan in JSON format:
{{
    "topic": "the research topic",
    "objectives": ["objective 1", "objective 2", ...],
    "search_queries": [
        {{"query": "search query 1", "purpose": "why this query"}},
        {{"query": "search query 2", "purpose": "why this query"}},
        ...
    ],
    "report_outline": ["section 1 title", "section 2 title", ...]
}}""")
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
            return {"error": "No research plan available"}
        
        logger.info(f"Autonomous agent researching: {len(state.plan.search_queries)} planned queries")
        
        # Create system prompt for autonomous agent with config-based limits
        max_searches = config.max_search_queries
        max_results_per_search = config.max_search_results_per_query
        expected_total_results = max_searches * max_results_per_search
        
        system_prompt = f"""You are an expert research assistant with access to web search and content extraction tools.

Your task is to efficiently research the given topic by:
1. Executing the planned search queries (limit to {max_searches} searches maximum)
2. Extracting detailed content from the most relevant sources
3. Gathering sufficient information to answer the research objectives

Guidelines:
- Use web_search to find relevant sources (limit to {max_searches} searches, {max_results_per_search} results each)
- Use extract_webpage_content to read full articles from the top {expected_total_results} promising URLs
- Focus on authoritative, credible sources
- Prioritize quality over quantity
- Extract content from {expected_total_results} high-quality sources

When you have gathered sufficient information ({expected_total_results} search results with content extractions), respond with:
RESEARCH_COMPLETE: [brief summary of what you found]"""
        
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
                input_message = f"""Research Topic: {state.research_topic}

Research Objectives:
{objectives_text}

Planned Search Queries:
{queries_text}

Begin your research. Use the tools to gather comprehensive information."""
                
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
                
                if not search_results:
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
            return {"error": "No search results to synthesize"}
        
        # Create system prompt for autonomous synthesis agent
        system_prompt = """You are an expert research synthesis agent. Your task is to analyze search results and extract key findings.

You have access to the extract_insights_from_text tool which can help extract specific insights from text.

IMPORTANT: Each source has a credibility score (HIGH/MEDIUM/LOW) and score (0-100). 
- PRIORITIZE information from HIGH-credibility sources (score ≥70)
- When sources contradict each other, trust HIGH-credibility sources over MEDIUM or LOW
- Use MEDIUM-credibility sources (score 40-69) to supplement but not override HIGH-credibility sources
- Be cautious with LOW-credibility sources (score <40) - only use if no other sources are available

For each finding:
- Summarize the key insight
- Note which sources support it (prefer citing high-credibility sources)
- Identify any contradictions or debates (resolve using credibility hierarchy)
- Be concise but comprehensive

Focus on the most important and relevant information from the most credible sources. When ready, return your findings as a JSON array of strings."""
        
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
                input_message = f"""Research Topic: {state.research_topic}

Search Results:
{results_text}

Please analyze these search results and extract key findings. You may use the extract_insights_from_text tool to help identify important insights. Return your findings as a JSON array of strings."""
                
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
            return {"error": "Insufficient data for report generation"}
        
        # Track total LLM calls for report generation
        report_llm_calls = 0
        report_input_tokens = 0
        report_output_tokens = 0
        report_call_details = []
        
        for attempt in range(self.max_retries):
            try:
                # Generate each section with retry
                report_sections = []
                for section_title in state.plan.report_outline:
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
        system_prompt = """You are an expert research writer agent. Write comprehensive, well-structured sections for research reports.

You have access to:
- format_citation: Format citations in academic styles
- validate_section_quality: Check if your section meets quality standards

Requirements:
- Minimum {min_words} words
- Use clear, academic language
- Include specific facts and data
- Cite sources using [1], [2], etc.
- Use markdown formatting
- Be objective and balanced

You may use validate_section_quality to check your work before finalizing."""
        
        # Create autonomous writing agent
        agent_graph = create_agent(
            self.llm,
            self.tools,
            system_prompt=system_prompt
        )
        
        try:
            start_time = time.time()
            
            # Prepare input message
            input_message = f"""Research Topic: {topic}
Section Title: {section_title}
Minimum Words: {config.min_section_words}

Key Findings:
{chr(10).join(f"- {f}" for f in findings)}

Please write this section in markdown format. Use inline citations [1], [2], etc. to reference sources."""
            
            # Estimate input tokens
            input_tokens = estimate_tokens(input_message)
            
            # Execute autonomous section writing
            result = await agent_graph.ainvoke({
                "messages": [{"role": "user", "content": input_message}]
            })
            
            # Extract content from result
            messages = result.get('messages', [])
            content = ""
            if messages:
                last_msg = messages[-1]
                # Handle different content formats
                if hasattr(last_msg, 'content'):
                    msg_content = last_msg.content
                    # If content is a list (like from tool responses), extract text
                    if isinstance(msg_content, list):
                        content = ""
                        for item in msg_content:
                            if isinstance(item, dict) and 'text' in item:
                                content += item['text']
                            elif isinstance(item, dict) and 'type' in item and item['type'] == 'text':
                                content += item.get('text', '')
                            else:
                                content += str(item)
                    else:
                        content = str(msg_content)
                else:
                    content = str(last_msg)
            
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
            
            return section, call_detail
            
        except Exception as e:
            logger.error(f"Error writing section '{section_title}': {str(e)}")
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

