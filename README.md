# Deep Research Agent

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2.57+-green.svg)](https://github.com/langchain-ai/langgraph)

A production-ready **multi-agent autonomous research system** built with LangGraph and LangChain. Four specialized autonomous agents work together to conduct comprehensive research on any topic and generate detailed, citation-backed reports with credibility scoring and quality metrics.

## ✨ Features

### Core Capabilities
- **Multi-Agent Architecture**: 4 specialized autonomous agents orchestrated by LangGraph's StateGraph
- **Agent Specialization**: ResearchPlanner, ResearchSearcher, ResearchSynthesizer, ReportWriter
- **Intelligent Tool Calling**: Each agent autonomously uses specialized tools
- **Dynamic Research Planning**: Planning agent analyzes topics and generates strategies
- **Autonomous Search**: Search agent decides when to search and what content to extract
- **Intelligent Synthesis**: Analysis agent extracts insights using specialized tools
- **Quality-Driven Writing**: Writing agent validates and formats reports autonomously
- **StateGraph Coordination**: Seamless multi-agent pipeline orchestration

### Advanced Features
- **Autonomous Decision Making**: Agent adapts research strategy based on findings
- **Quality Over Quantity**: Focused on 5-8 high-quality sources with deep content extraction
- **Research Caching**: Avoid redundant searches with intelligent caching (7-day TTL)
- **Source Credibility Scoring**: Automatic domain authority analysis, filtering, and prioritization of high-quality sources
- **Section Quality Validation**: Automatic validation of report sections for length and quality
- **Research History**: Track and search past research projects
- **LLM Usage Tracking**: Monitor API calls, token usage, and costs in real-time
- **Multi-Format Export**: Generate reports in Markdown, HTML, and plain text formats
- **Web Interface**: Interactive Chainlit-based UI for easy research

## 🏗️ Architecture

### System Overview

The agent implements a multi-stage pipeline orchestrated by LangGraph's StateGraph:

```
┌───────────────────────────────────────────────────────────────────┐
│         Multi-Agent Deep Research System (StateGraph)             │
└───────────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  ResearchState     │
                    │  (Shared State)    │
                    └─────────┬─────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
   ┌────▼──────────┐    ┌────▼──────────┐    ┌────▼──────────┐
   │ ResearchPlanner│───▶│ResearchSearcher│──▶│ResearchSynth  │
   │  [Planning     │    │  [Search        │   │  [Analysis    │
   │   Agent]       │    │   Agent]        │   │   Agent]      │
   │                │    │                 │   │               │
   │ Tools:         │    │ Tools:          │   │ Tools:        │
   │ • analyze_     │    │ • web_search    │   │ • extract_    │
   │   research_    │    │ • extract_      │   │   insights_   │
   │   topic        │    │   webpage       │   │   from_text   │
   └────────────────┘    └─────────────────┘   └───────┬───────┘
                                                        │
                                                ┌───────▼───────┐
                                                │ ReportWriter  │
                                                │  [Writing     │
                                                │   Agent]      │
                                                │               │
                                                │ Tools:        │
                                                │ • format_     │
                                                │   citation    │
                                                │ • validate_   │
                                                │   section     │
                                                └───────────────┘
```

### Key Features

1. **Autonomous Agent Architecture**: LLM agent with tool-calling capabilities for dynamic research
2. **Intelligent Tool Selection**: Agent decides when to search and what content to extract
3. **Adaptive Research Strategy**: Agent adjusts approach based on intermediate findings
4. **Retry Logic with Exponential Backoff**: All agents retry failed operations up to 3 times
5. **Progressive Truncation**: Automatically reduces data size on token limit errors
6. **Validation at Each Stage**: Validates output quality before proceeding
7. **Enhanced Error Handling**: Graceful degradation with detailed error messages
8. **Quality Thresholds**: Minimum requirements for search results and report length
9. **Conditional Routing**: Smart workflow decisions based on validation results
10. **Quality Over Quantity**: Focused approach with 5-8 high-quality sources

### Core Components

#### 1. State Management (`src/state.py`)

Centralized state using Pydantic models:

```python
class ResearchState(BaseModel):
    research_topic: str
    plan: Optional[ResearchPlan]
    search_results: List[SearchResult]
    key_findings: List[str]
    report_sections: List[ReportSection]
    final_report: Optional[str]
    current_stage: Literal["planning", "searching", "synthesizing", "reporting", "complete"]
    error: Optional[str]
    
    # LLM tracking
    llm_calls: int
    total_input_tokens: int
    total_output_tokens: int
    llm_call_details: List[Dict]
    
    # Quality metrics
    quality_score: Optional[Dict]
    credibility_scores: List[Dict]
```

#### 2. Agent Nodes (`src/agents.py`)

**ResearchPlanner**
- Analyzes research topic
- Generates 3-5 research objectives
- Creates 3 targeted search queries
- Designs report outline
- Provides guidance for autonomous research agent
- **Retry logic**: 3 attempts with exponential backoff
- **Validation**: Ensures valid plan structure before proceeding
- **LLM Tracking**: Tracks API calls and token usage

**ResearchSearcher** (Autonomous Agent)
- **LangChain-powered autonomous agent** using `create_agent()`
- **Dynamically decides** when to search and what to extract
- **Tool-calling capabilities**: Uses `web_search` and `extract_webpage_content` tools
- Adapts research strategy based on intermediate findings
- Autonomously extracts full content from 3-5 promising sources
- Target: 5-8 high-quality search results
- **Quality-focused**: Prioritizes depth over breadth
- **Error tolerance**: Continues with available results on failures
- **Credibility scoring & filtering**: 
  - Automatically scores all sources (0-100) based on domain authority, HTTPS, suspicious patterns
  - Filters out low-credibility sources below minimum threshold (default: 40)
  - Sorts remaining sources by credibility (highest first)
  - Tracks credibility scores for downstream prioritization
- **Transparent tool usage**: All tool calls are logged and traceable

**ResearchSynthesizer**
- Analyzes aggregated search results with credibility awareness
- **Credibility-aware synthesis**: 
  - Receives credibility scores (HIGH/MEDIUM/LOW) for each source
  - Explicitly instructed to prioritize HIGH-credibility sources (score ≥70)
  - Resolves contradictions using credibility hierarchy
  - Prefers citing high-credibility sources in findings
- Extracts key insights and findings
- Identifies patterns and contradictions
- **Progressive truncation**: Reduces data on token limit errors
- **Retry logic**: 3 attempts with adaptive data reduction
- **Fallback parsing**: JSON and text-based extraction
- **LLM Tracking**: Monitors token usage during synthesis

**ReportWriter**
- Generates structured report sections
- Maintains consistent academic tone
- Adds proper citations with configurable styles (APA, MLA, Chicago, IEEE)
- Compiles final markdown document
- **Quality validation**: Minimum length requirements (500+ chars)
- **Retry logic**: Re-generates on quality failures
- **Section validation**: Ensures all sections are generated
- **LLM Tracking**: Tracks token usage per section
- **Multi-format export**: Generates Markdown, HTML, and TXT versions

#### 3. Tools Layer (`src/utils/tools.py`)

LangChain tools decorated with `@tool` for autonomous agent use:

**`web_search` Tool**
- **LLM-callable**: Agent decides when to invoke
- DuckDuckGo integration via `duckduckgo-search` library
- Returns 3 results per query (configurable)
- Rich descriptions for intelligent tool selection
- Async search operations
- **Autonomous usage**: Agent can call multiple times with different queries

**`extract_webpage_content` Tool**
- **LLM-callable**: Agent chooses which URLs to extract
- HTML parsing with BeautifulSoup4
- Removes navigation, scripts, and styling
- Extracts main content intelligently (up to 5000 chars)
- Timeout and error handling
- **Autonomous usage**: Agent selects most promising sources

**Supporting Classes** (`src/utils/web_utils.py`):
- `WebSearchTool`: Underlying search implementation
- `ContentExtractor`: Underlying extraction implementation

#### 4. Enhanced Modules

**Research Cache** (`src/utils/cache.py`)
- File-based caching with 7-day TTL
- MD5-based topic hashing
- Automatic expiration cleanup
- Reduces redundant API calls

**Credibility Scorer** (`src/utils/credibility.py`)
- **Domain authority analysis**: Recognizes trusted domains (.edu, .gov, .ac.in, reputable news sites)
- **Credibility scoring**: 0-100 scale based on multiple factors:
  - Trusted domains (+30 points)
  - HTTPS enabled (+5 points)
  - Academic/research paths (+10 points)
  - Suspicious patterns (-20 points: suspicious TLDs, URL shorteners, personal blogs)
  - Domain structure analysis
- **Automatic filtering**: Removes sources below minimum credibility threshold (configurable, default: 40)
- **Automatic prioritization**: Sorts sources by credibility score (highest first)
- **Credibility levels**: HIGH (≥70), MEDIUM (40-69), LOW (<40)
- **Integration**: Credibility scores passed to synthesis agent for intelligent source weighting

**Report Exporter** (`src/utils/exports.py`)
- Markdown export (default)
- HTML export with beautiful styling
- Plain text export (markdown stripped)
- Consistent formatting across formats

**Citation Formatter** (`src/utils/citations.py`)
- Multiple citation styles: APA, MLA, Chicago, IEEE
- Automatic reference section formatting
- Metadata extraction from search results
- Configurable via environment variable

**Section Quality Validator** (`src/utils/tools.py`)
- Validates report sections for minimum length and quality
- Provides feedback on section quality
- Integrated into ReportWriter for automatic validation

**Research History** (`src/utils/history.py`)
- Persistent research tracking
- Search functionality by topic
- Metadata storage (quality scores, statistics)
- Automatic cleanup (keeps last 100 entries)

**LLM Tracker** (`src/llm_tracker.py`)
- Real-time token usage monitoring
- API call tracking per agent
- Duration measurement
- Token estimation (1 token ≈ 4 characters)

#### 5. Workflow Graph (`src/graph.py`)

LangGraph StateGraph with enhanced validation routing:

```python
workflow = StateGraph(ResearchState)
workflow.add_node("plan", planner.plan)
workflow.add_node("search", searcher.search)
workflow.add_node("synthesize", synthesizer.synthesize)
workflow.add_node("write_report", writer.write_report)

# Validation functions check:
# - Error states
# - Minimum data thresholds
# - Output quality requirements

def should_continue_after_plan(state):
    if state.error or not state.plan.search_queries:
        return END
    return "search"

def should_continue_after_search(state):
    if state.error or len(state.search_results) < 2:
        return END
    return "synthesize"
```

### Data Flow

1. **Input**: User provides research topic
2. **Cache Check**: System checks for cached results (if enabled)
3. **Planning**: LLM generates research plan with 3 queries and outline (tracked)
4. **Autonomous Research**: 
   - Agent receives research objectives and guidance
   - **Agent autonomously decides**: Which queries to execute, when to extract content
   - **Tool calls**: `web_search` called 2-3 times for different queries
   - **Tool calls**: `extract_webpage_content` called 3-5 times for promising sources
   - Agent adapts strategy based on findings
   - Agent determines when research is complete (5-8 quality sources)
5. **Credibility Scoring & Filtering**: 
   - All sources scored (0-100) based on domain authority, HTTPS, suspicious patterns
   - Low-credibility sources filtered out (below minimum threshold)
   - Remaining sources sorted by credibility (highest first)
6. **Credibility-Aware Synthesis**: 
   - LLM receives credibility scores for each source
   - Explicitly instructed to prioritize HIGH-credibility sources
   - Resolves contradictions using credibility hierarchy
   - Extracts key findings with preference for authoritative sources (tracked)
7. **Report Generation**: LLM writes each section based on findings (tracked)
8. **Citation Formatting**: References formatted according to selected style
9. **Multi-format Export**: Report exported as Markdown, HTML, and TXT
10. **Output**: Reports saved to `outputs/` directory, history updated
11. **Metrics Display**: LLM usage stats and recommendations shown

### Configuration (`src/config.py`)

Environment-based configuration with Pydantic validation:

```python
class ResearchConfig(BaseModel):
    google_api_key: str  # Gemini API key
    model_name: str = "gemini-2.5-flash"
    summarization_model: str = "gemini-2.5-flash"
    max_search_queries: int = 3          # Planned queries for agent
    max_search_results_per_query: int = 3  # Results per search
    max_parallel_searches: int = 3
    min_credibility_score: int = 40      # Minimum credibility score (0-100) to filter sources
    max_report_sections: int = 8
    citation_style: str = "apa"  # apa, mla, chicago, ieee
    
# Agent autonomously targets: 5-8 total results with 3-5 full content extractions
# Low-credibility sources are automatically filtered out
```

## 🚀 Installation

### Prerequisites

- Python 3.11+
- pip or uv package manager
- Google Gemini API key ([Get one free](https://makersuite.google.com/app/apikey))

### Setup

```bash
# Clone the repository
git clone https://github.com/tarun7r/deep-research-agent.git
cd deep-research-agent

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
# Create a .env file in the root directory
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

## 📖 Usage

### Command Line

```bash
# Interactive mode
python main.py

# Direct topic
python main.py "Impact of quantum computing on cryptography"
```

### Programmatic API

```python
import asyncio
from src.graph import run_research

async def research():
    state = await run_research("Topic here", verbose=True, use_cache=True)
    
    # Access report
    print(state["final_report"])
    
    # Access LLM metrics
    print(f"LLM Calls: {state['llm_calls']}")
    print(f"Input Tokens: {state['total_input_tokens']:,}")
    print(f"Output Tokens: {state['total_output_tokens']:,}")
    print(f"Total Tokens: {state['total_input_tokens'] + state['total_output_tokens']:,}")
    
    # Access quality score
    if state.get("quality_score"):
        print(f"Quality: {state['quality_score']['total_score']}/100")
        print(f"Level: {state['quality_score']['quality_level']}")

asyncio.run(research())
```

### Web Interface (Chainlit)

```bash
# Start the web interface
chainlit run app.py --host 127.0.0.1 --port 8000
```

The web interface provides:
- Interactive chat-based research
- Real-time progress updates
- Quality metrics display
- LLM usage statistics
- Multiple format downloads
- Research history

### Custom Configuration

```python
from src.config import ResearchConfig

config = ResearchConfig(
    model_name="gemini-2.5-flash",
    max_search_queries=10,
    max_report_sections=15,
    citation_style="mla"
)
```

## 📄 Output Format

Generated reports follow this structure:

```markdown
# [Research Topic]

**Deep Research Report**

## Research Objectives
1. [Objective 1]
2. [Objective 2]
...

---

## [Section 1 Title]
[Content with inline citations [1], [2]]

## [Section 2 Title]
[Content with inline citations [3], [4]]

---

## References
1. [Formatted citation according to selected style]
2. [Formatted citation according to selected style]
...
```

### Export Formats

Reports are automatically exported in three formats:
- **Markdown** (`.md`) - Original format with full markdown syntax
- **HTML** (`.html`) - Beautifully styled web-ready format
- **Plain Text** (`.txt`) - Markdown stripped, plain text version

### LLM Usage Tracking

Real-time monitoring of:
- **API Calls**: Total number of LLM invocations
- **Input Tokens**: Tokens sent to the model
- **Output Tokens**: Tokens generated by the model
- **Total Tokens**: Combined usage for cost estimation
- **Call Details**: Per-agent breakdown with durations

## ⚙️ Configuration

Environment variables in `.env`:

```bash
# Required
GEMINI_API_KEY=your_api_key_here

# Optional - Autonomous Agent Settings
MAX_SEARCH_QUERIES=3              # Planned queries for agent (agent may do more/less)
MAX_SEARCH_RESULTS_PER_QUERY=3    # Results per search (quality over quantity)
MAX_PARALLEL_SEARCHES=3           # Concurrent operations
MIN_CREDIBILITY_SCORE=40          # Minimum credibility score (0-100) to filter low-quality sources
MAX_REPORT_SECTIONS=8             # Report sections

# Agent autonomously targets: 5-8 search results with 3-5 full content extractions
# Low-credibility sources are automatically filtered out before synthesis

# Citation Style (optional)
CITATION_STYLE=apa  # Options: apa, mla, chicago, ieee

# LangSmith (optional monitoring)
LANGCHAIN_TRACING_V2=false
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=deep-research-agent
```

## 📁 Project Structure

```
deep-research-agent/
├── src/
│   ├── __init__.py       # Package initialization
│   ├── config.py         # Configuration management
│   ├── state.py          # State models (Pydantic)
│   ├── agents.py         # Autonomous agent implementations
│   ├── graph.py          # LangGraph workflow
│   ├── llm_tracker.py    # LLM call and token tracking
│   └── utils/
│       ├── __init__.py   # Utils package
│       ├── tools.py      # LangChain tools (@tool decorated)
│       ├── web_utils.py  # Search & extraction implementations
│       ├── cache.py      # Research result caching
│       ├── credibility.py    # Source credibility scoring
│       ├── exports.py    # Multi-format export utilities
│       ├── citations.py  # Citation formatting
│       └── history.py    # Research history tracking
├── outputs/              # Generated reports (MD, HTML, TXT)
├── .cache/               # Cache and history storage
│   ├── research/         # Cached research results
│   └── research_history.json  # Research history
├── main.py               # CLI entry point
├── app.py                # Chainlit web interface
├── chainlit.md           # Chainlit welcome message
├── requirements.txt      # Python dependencies
├── pyproject.toml        # Project metadata
├── LICENSE               # MIT License
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

### Key Benefits

- **Adaptive**: Agent adjusts research strategy based on findings
- **Intelligent**: Decides which sources warrant deep content extraction
- **Efficient**: Quality-focused approach with fewer but better sources
- **Extensible**: Easy to add new tools for the agent to use
- **Transparent**: All tool calls are logged and traceable

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

MIT License - See [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [LangGraph](https://github.com/langchain-ai/langgraph) and [LangChain](https://github.com/langchain-ai/langchain)
- Powered by [Google Gemini](https://ai.google.dev/)
- Web search via [DuckDuckGo](https://duckduckgo.com/)
