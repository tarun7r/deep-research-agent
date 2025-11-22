# Deep Research Agent

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2.57+-green.svg)](https://github.com/langchain-ai/langgraph)

A production-ready multi-agent autonomous research system built with LangGraph and LangChain. Four specialized agents work together to conduct comprehensive research on any topic and generate detailed, citation-backed reports with credibility scoring and quality metrics.

## Demo

https://github.com/tarun7r/deep-research-agent/assets/demo.mp4

*Watch the full demo video to see the Deep Research Agent in action, showcasing the multi-agent workflow, real-time progress updates, and comprehensive report generation.*

## Features

### Core Capabilities

- **Multi-Agent Architecture**: Four specialized autonomous agents (ResearchPlanner, ResearchSearcher, ResearchSynthesizer, ReportWriter) orchestrated by LangGraph's StateGraph. Each agent operates independently with its own tools and decision-making logic.

- **Autonomous Research**: The search agent dynamically decides when to search, which queries to execute, and which sources warrant deep content extraction. This adaptive approach ensures quality over quantity, typically targeting 5-8 high-quality sources.

- **Credibility Scoring**: Automatic source evaluation using domain authority analysis. Sources are scored (0-100) based on trusted domains (.edu, .gov), HTTPS, suspicious patterns, and academic indicators. Low-credibility sources are automatically filtered before synthesis.

- **Quality Validation**: Section-level validation ensures minimum length requirements (500+ characters) and quality standards. Retry logic with exponential backoff handles failures gracefully, with up to 3 attempts per operation.

- **Multi-Format Export**: Reports are automatically exported in three formats: Markdown (original), HTML (styled for web), and plain text (markdown stripped).

- **LLM Usage Tracking**: Real-time monitoring of API calls, input/output tokens, and estimated costs. Per-agent breakdowns help identify optimization opportunities.

- **Research Caching**: Intelligent file-based caching with 7-day TTL reduces redundant API calls. MD5-based topic hashing ensures accurate cache lookups.

- **Web Interface**: Interactive Chainlit-based UI provides real-time progress updates, quality metrics, and multiple format downloads.

## Architecture

The system implements a four-stage pipeline orchestrated by LangGraph's StateGraph:

```
ResearchPlanner → ResearchSearcher → ResearchSynthesizer → ReportWriter
```

### Agent Responsibilities

**ResearchPlanner**
- Analyzes research topics and generates 3-5 research objectives
- Creates 3 targeted search queries covering different aspects
- Designs report outline with up to 8 sections
- Provides strategic guidance for the autonomous search agent

**ResearchSearcher** (Autonomous Agent)
- LangChain-powered autonomous agent using `create_agent()`
- Dynamically decides which queries to execute and when to extract content
- Uses `web_search` and `extract_webpage_content` tools autonomously
- Adapts research strategy based on intermediate findings
- Targets 5-8 high-quality sources with deep content extraction
- All sources are scored for credibility and filtered before synthesis

**ResearchSynthesizer**
- Analyzes aggregated search results with credibility awareness
- Prioritizes HIGH-credibility sources (score ≥70) in findings
  - Resolves contradictions using credibility hierarchy
- Extracts key insights and identifies patterns
- Progressive truncation handles token limit errors gracefully

**ReportWriter**
- Generates structured report sections with consistent academic tone
- Adds proper citations with configurable styles (APA, MLA, Chicago, IEEE)
- Validates section quality and re-generates on failures
- Compiles final markdown document with reference section

### Workflow

1. **Planning**: LLM generates research plan with objectives, queries, and outline
2. **Autonomous Search**: Agent executes searches and extracts content from promising sources
3. **Credibility Scoring**: All sources scored and filtered (default threshold: 40)
4. **Synthesis**: Findings extracted with credibility-aware prioritization
5. **Report Generation**: Structured sections written with citations
6. **Export**: Reports saved in multiple formats to `outputs/` directory

## Installation

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

## Usage

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

asyncio.run(research())
```

### Web Interface

```bash
# Start the web interface
chainlit run app.py --host 127.0.0.1 --port 8000
```

The web interface provides:
- Interactive chat-based research
- Real-time progress updates with stage indicators
- Quality metrics and LLM usage statistics
- Multiple format downloads (Markdown, HTML, TXT)
- Research history tracking

## Configuration

Environment variables in `.env`:

```bash
# Required
GEMINI_API_KEY=your_api_key_here

# Optional - Model Settings
MODEL_NAME=gemini-2.5-flash
SUMMARIZATION_MODEL=gemini-2.5-flash

# Optional - Search Settings
MAX_SEARCH_QUERIES=3              # Planned queries for agent
MAX_SEARCH_RESULTS_PER_QUERY=3    # Results per search
MAX_PARALLEL_SEARCHES=3           # Concurrent operations
MIN_CREDIBILITY_SCORE=40          # Minimum credibility score (0-100) to filter sources

# Optional - Report Settings
MAX_REPORT_SECTIONS=8             # Maximum report sections
CITATION_STYLE=apa                 # Options: apa, mla, chicago, ieee

# Optional - LangSmith Monitoring
LANGCHAIN_TRACING_V2=false
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=deep-research-agent
```

### Configuration Details

- **MIN_CREDIBILITY_SCORE**: Sources below this threshold are filtered out before synthesis. Default 40 ensures only reasonable-quality sources are used.
- **MAX_SEARCH_QUERIES**: Number of planned queries. The autonomous agent may execute more or fewer based on findings.
- **CITATION_STYLE**: Choose from APA, MLA, Chicago, or IEEE citation formats.

## Output Format

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

Reports are automatically exported in three formats:
- **Markdown** (`.md`) - Original format with full markdown syntax
- **HTML** (`.html`) - Styled web-ready format
- **Plain Text** (`.txt`) - Markdown stripped, plain text version

All reports are saved to the `outputs/` directory with timestamps.

## Project Structure

```
deep-research-agent/
├── src/
│   ├── __init__.py       # Package initialization
│   ├── config.py         # Configuration management (Pydantic models)
│   ├── state.py          # State models (ResearchState, ResearchPlan, etc.)
│   ├── agents.py         # Agent implementations (Planner, Searcher, Synthesizer, Writer)
│   ├── graph.py          # LangGraph workflow orchestration
│   ├── llm_tracker.py    # LLM call and token tracking
│   └── utils/
│       ├── __init__.py   # Utils package
│       ├── tools.py      # LangChain tools (@tool decorated for agents)
│       ├── web_utils.py  # Search & extraction implementations
│       ├── cache.py      # Research result caching (7-day TTL)
│       ├── credibility.py # Source credibility scoring and filtering
│       ├── exports.py    # Multi-format export utilities
│       ├── citations.py  # Citation formatting (APA, MLA, Chicago, IEEE)
│       └── history.py    # Research history tracking
├── outputs/              # Generated reports (MD, HTML, TXT)
├── .cache/               # Cache and history storage
│   ├── research/         # Cached research results
│   └── research_history.json  # Research history
├── main.py               # CLI entry point
├── app.py                # Chainlit web interface
├── requirements.txt      # Python dependencies
├── pyproject.toml        # Project metadata
├── LICENSE               # MIT License
└── README.md             # This file
```

## Key Components

### State Management (`src/state.py`)

Centralized state using Pydantic models tracks research progress, search results, findings, and LLM usage metrics throughout the workflow.

### Tools Layer (`src/utils/tools.py`)

LangChain tools decorated with `@tool` enable autonomous agent tool-calling:
- `web_search`: DuckDuckGo integration for web searches
- `extract_webpage_content`: BeautifulSoup4-based content extraction

### Credibility Scorer (`src/utils/credibility.py`)

Evaluates sources based on:
- Domain authority (trusted domains: +30 points)
- HTTPS enabled (+5 points)
- Academic/research paths (+10 points)
- Suspicious patterns (-20 points)

Sources are automatically filtered and sorted by credibility before synthesis.

## Development Note

The core ideation, architecture design, and logic of this project are the result of original research and understanding. While AI tools were used to assist with code restructuring and implementation, the fundamental concepts, agent workflows, credibility scoring methodology, and overall system design reflect independent research and development.

## License

MIT License - See [LICENSE](LICENSE) file for details.

## Acknowledgments

Built with [LangGraph](https://github.com/langchain-ai/langgraph) and [LangChain](https://github.com/langchain-ai/langchain). Powered by [Google Gemini](https://ai.google.dev/). Web search via [DuckDuckGo](https://duckduckgo.com/).
