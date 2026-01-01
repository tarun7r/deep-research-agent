"""Interactive Chainlit interface for Deep Research Agent with enhanced UX."""

import asyncio
import chainlit as cl
from pathlib import Path
from datetime import datetime
from typing import Optional

from src.config import config
from src.state import ResearchState
from src.graph import create_research_graph
from src.utils.exports import ReportExporter
from src.utils.history import ResearchHistory
from src.callbacks import (
    progress_callback, 
    ProgressUpdate, 
    ResearchStage,
    emit_complete
)


# ═══════════════════════════════════════════════════════════════════════════════
# VISUAL CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

STAGE_EMOJI = {
    ResearchStage.INITIALIZING: "",
    ResearchStage.PLANNING: "",
    ResearchStage.SEARCHING: "",
    ResearchStage.EXTRACTING: "",
    ResearchStage.SYNTHESIZING: "",
    ResearchStage.WRITING: "",
    ResearchStage.COMPLETE: "",
    ResearchStage.ERROR: ""
}

STAGE_COLORS = {
    ResearchStage.INITIALIZING: "#6b7280",
    ResearchStage.PLANNING: "#8b5cf6",
    ResearchStage.SEARCHING: "#3b82f6",
    ResearchStage.EXTRACTING: "#06b6d4",
    ResearchStage.SYNTHESIZING: "#10b981",
    ResearchStage.WRITING: "#f59e0b",
    ResearchStage.COMPLETE: "#22c55e",
    ResearchStage.ERROR: "#ef4444"
}

STAGE_NAMES = {
    ResearchStage.INITIALIZING: "Initializing",
    ResearchStage.PLANNING: "Planning Strategy",
    ResearchStage.SEARCHING: "Web Search",
    ResearchStage.EXTRACTING: "Content Extraction",
    ResearchStage.SYNTHESIZING: "AI Synthesis",
    ResearchStage.WRITING: "Report Writing",
    ResearchStage.COMPLETE: "Complete",
    ResearchStage.ERROR: "Error"
}


# ═══════════════════════════════════════════════════════════════════════════════
# PROGRESS DISPLAY
# ═══════════════════════════════════════════════════════════════════════════════

class EnhancedProgressDisplay:
    """Modern progress display with animated visuals."""
    
    def __init__(self):
        self.message: cl.Message = None
        self.step_messages: dict = {}  # Track individual step messages
        self.updates: list[ProgressUpdate] = []
        self.current_stage: ResearchStage = ResearchStage.INITIALIZING
        self.start_time: datetime = None
        self.topic: str = ""
        
    async def initialize(self, topic: str):
        """Initialize with a sleek progress card."""
        self.start_time = datetime.now()
        self.updates = []
        self.topic = topic
        self.current_stage = ResearchStage.INITIALIZING
        
        content = self._render()
        self.message = cl.Message(content=content)
        await self.message.send()
    
    async def update(self, progress_update: ProgressUpdate):
        """Update with smooth transitions."""
        self.updates.append(progress_update)
        self.current_stage = progress_update.stage
        
        if self.message:
            self.message.content = self._render()
            await self.message.update()
    
    def _get_elapsed(self) -> str:
        """Get formatted elapsed time."""
        if not self.start_time:
            return "0s"
        delta = datetime.now() - self.start_time
        minutes, seconds = divmod(delta.seconds, 60)
        if minutes > 0:
            return f"{minutes}m {seconds}s"
        return f"{seconds}s"
    
    def _render_progress_bar(self, pct: float) -> str:
        """Render a modern progress bar."""
        filled = int(25 * pct / 100)
        empty = 25 - filled
        bar = "█" * filled + "░" * empty
        return f"`[{bar}]` **{pct:.0f}%**"
    
    def _render_stage_pipeline(self) -> str:
        """Render the stage pipeline with status indicators."""
        stages = [
            ResearchStage.PLANNING,
            ResearchStage.SEARCHING,
            ResearchStage.EXTRACTING,
            ResearchStage.SYNTHESIZING,
            ResearchStage.WRITING,
            ResearchStage.COMPLETE
        ]
        
        current_idx = -1
        if self.current_stage in stages:
            current_idx = stages.index(self.current_stage)
        
        lines = []
        for idx, stage in enumerate(stages):
            emoji = STAGE_EMOJI.get(stage, "")
            name = STAGE_NAMES.get(stage, stage.value)
            
            if idx < current_idx:
                # Completed
                lines.append(f"  [DONE] ~~{name}~~")
            elif idx == current_idx:
                # Current - active with animation hint
                lines.append(f"  **{name}** <- *in progress*")
            else:
                # Pending
                lines.append(f"  [ ] {name}")
        
        return "\n".join(lines)
    
    def _render_activity_feed(self) -> str:
        """Render recent activity with timestamps."""
        if not self.updates:
            return "*Initializing research agent...*"
        
        lines = []
        recent = self.updates[-6:]  # Last 6 updates
        
        for update in reversed(recent):
            emoji = STAGE_EMOJI.get(update.stage, "")
            time_str = update.timestamp.strftime("%H:%M:%S")
            msg = update.message
            
            line = f"`{time_str}` {emoji}{msg}" if emoji else f"`{time_str}` {msg}"
            if update.details:
                # Truncate long details
                details = update.details[:60] + "..." if len(update.details) > 60 else update.details
                line += f"\n> _{details}_"
            lines.append(line)
        
        return "\n\n".join(lines)
    
    def _render(self) -> str:
        """Render the complete progress display."""
        elapsed = self._get_elapsed()
        
        # Get current progress percentage
        pct = 0
        if self.updates:
            for update in reversed(self.updates):
                if update.progress_pct is not None:
                    pct = update.progress_pct
                    break
        
        progress_bar = self._render_progress_bar(pct)
        pipeline = self._render_stage_pipeline()
        activity = self._render_activity_feed()
        
        return f"""
## Research in Progress

**Topic:** *{self.topic}*

---

### Progress

{progress_bar}

Elapsed: **{elapsed}**

---

### Pipeline Status

{pipeline}

---

### Activity Feed

{activity}
"""


# ═══════════════════════════════════════════════════════════════════════════════
# RESEARCH RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

async def run_research_with_updates(topic: str, progress_display: EnhancedProgressDisplay):
    """Run research with real-time updates."""
    progress_callback.reset()
    
    async def on_progress(update: ProgressUpdate):
        await progress_display.update(update)
    
    progress_callback.register_async(on_progress)
    
    try:
        initial_state = ResearchState(research_topic=topic)
        graph = create_research_graph()
        final_state = await graph.ainvoke(initial_state)
        
        search_results = final_state.get('search_results', [])
        key_findings = final_state.get('key_findings', [])
        await emit_complete(topic, len(search_results), len(key_findings))
        
        return final_state
    finally:
        progress_callback.unregister(on_progress)


# ═══════════════════════════════════════════════════════════════════════════════
# CHAT HANDLERS
# ═══════════════════════════════════════════════════════════════════════════════

@cl.on_chat_start
async def start():
    """Initialize with an engaging welcome experience."""
    
    # Store session state
    cl.user_session.set("research_count", 0)
    
    # Create action buttons for quick start
    actions = [
        cl.Action(
            name="example_quantum",
            payload={"topic": "Future of quantum computing in 2025"},
            label="Quantum Computing"
        ),
        cl.Action(
            name="example_ai",
            payload={"topic": "Latest breakthroughs in AI agents and autonomous systems"},
            label="AI Agents"
        ),
        cl.Action(
            name="example_climate",
            payload={"topic": "Emerging technologies in climate change mitigation"},
            label="Climate Tech"
        ),
        cl.Action(
            name="show_history",
            payload={},
            label="View History"
        ),
        cl.Action(
            name="show_settings",
            payload={},
            label="Settings"
        )
    ]
    
    welcome_content = f"""
# Deep Research Agent

**Transform questions into comprehensive, well-sourced research reports.**

I analyze your research topic, search authoritative sources across the web, evaluate source credibility, synthesize key findings, and generate professional reports with proper citations all in minutes.

---

## What I Do

| Step | Description |
|------|-------------|
| **Plan** | Create strategic research objectives and search queries |
| **Search** | Query the web using DuckDuckGo for authoritative sources |
| **Extract** | Pull full content from high-credibility websites |
| **Synthesize** | Analyze and cross-reference findings with AI |
| **Write** | Generate a comprehensive, cited report |

---

## Quick Start

Click a topic below or type your own research question:

"""
    
    await cl.Message(
        content=welcome_content,
        actions=actions
    ).send()


@cl.action_callback("example_quantum")
async def on_example_quantum(action: cl.Action):
    """Handle quantum computing example."""
    topic = action.payload.get("topic")
    await start_research(topic)


@cl.action_callback("example_ai")
async def on_example_ai(action: cl.Action):
    """Handle AI agents example."""
    topic = action.payload.get("topic")
    await start_research(topic)


@cl.action_callback("example_climate")
async def on_example_climate(action: cl.Action):
    """Handle climate tech example."""
    topic = action.payload.get("topic")
    await start_research(topic)


@cl.action_callback("show_history")
async def on_show_history(action: cl.Action):
    """Show research history."""
    history = ResearchHistory()
    entries = history.get_recent(limit=10)
    
    if not entries:
        await cl.Message(
            content="**Research History**\n\n*No previous research found. Start your first research above!*"
        ).send()
        return
    
    content = "# Research History\n\n"
    content += "| Date | Topic | Sources | Findings |\n"
    content += "|------|-------|---------|----------|\n"
    
    for entry in entries:
        date = entry.get('timestamp', 'N/A')
        if isinstance(date, str) and len(date) > 10:
            date = date[:10]
        topic = entry.get('topic', 'Unknown')[:40]
        if len(entry.get('topic', '')) > 40:
            topic += "..."
        sources = entry.get('metadata', {}).get('sources', 'N/A')
        findings = entry.get('metadata', {}).get('findings', 'N/A')
        content += f"| {date} | {topic} | {sources} | {findings} |\n"
    
    await cl.Message(content=content).send()


@cl.action_callback("show_settings")
async def on_show_settings(action: cl.Action):
    """Show current configuration."""
    content = f"""# Current Settings

## Model Configuration
| Setting | Value |
|---------|-------|
| Provider | `{config.model_provider}` |
| Model | `{config.model_name}` |
| Summarization Model | `{config.summarization_model}` |

## Search Configuration
| Setting | Value |
|---------|-------|
| Max Search Queries | `{config.max_search_queries}` |
| Results per Query | `{config.max_search_results_per_query}` |
| Min Credibility Score | `{config.min_credibility_score}` |

## Report Configuration
| Setting | Value |
|---------|-------|
| Max Sections | `{config.max_report_sections}` |
| Min Words/Section | `{config.min_section_words}` |
| Citation Style | `{config.citation_style.upper()}` |

---

*To change settings, modify your `.env` file and restart the app.*
"""
    await cl.Message(content=content).send()


@cl.action_callback("download_md")
async def on_download_md(action: cl.Action):
    """Handle markdown download."""
    file_path = action.payload.get("path")
    if file_path and Path(file_path).exists():
        elements = [cl.File(name=Path(file_path).name, path=file_path, display="inline")]
        await cl.Message(content="**Markdown Report:**", elements=elements).send()


@cl.action_callback("download_html")
async def on_download_html(action: cl.Action):
    """Handle HTML download."""
    file_path = action.payload.get("path")
    if file_path and Path(file_path).exists():
        elements = [cl.File(name=Path(file_path).name, path=file_path, display="inline")]
        await cl.Message(content="**HTML Report:**", elements=elements).send()


@cl.action_callback("download_txt")
async def on_download_txt(action: cl.Action):
    """Handle TXT download."""
    file_path = action.payload.get("path")
    if file_path and Path(file_path).exists():
        elements = [cl.File(name=Path(file_path).name, path=file_path, display="inline")]
        await cl.Message(content="**Plain Text Report:**", elements=elements).send()


@cl.action_callback("view_sources")
async def on_view_sources(action: cl.Action):
    """Show detailed source list."""
    sources = action.payload.get("sources", [])
    credibility = action.payload.get("credibility", [])
    
    if not sources:
        await cl.Message(content="*No sources available.*").send()
        return
    
    content = "# Source Analysis\n\n"
    content += "| # | Credibility | Source | URL |\n"
    content += "|---|-------------|--------|-----|\n"
    
    for i, source in enumerate(sources[:30]):
        title = source.get('title', 'Unknown')[:35]
        if len(source.get('title', '')) > 35:
            title += "..."
        url = source.get('url', 'N/A')
        
        # Get credibility badge
        cred = credibility[i] if i < len(credibility) else {}
        level = cred.get('level', 'unknown')
        score = cred.get('score', 'N/A')
        
        badge = "[HIGH]" if level == 'high' else "[MED]" if level == 'medium' else "[LOW]"
        content += f"| {i+1} | {badge} {score} | {title} | [Link]({url}) |\n"
    
    await cl.Message(content=content).send()


@cl.action_callback("view_findings")
async def on_view_findings(action: cl.Action):
    """Show key findings in detail."""
    findings = action.payload.get("findings", [])
    
    if not findings:
        await cl.Message(content="*No findings available.*").send()
        return
    
    content = "# Key Findings\n\n"
    for i, finding in enumerate(findings, 1):
        content += f"**{i}.** {finding}\n\n"
    
    await cl.Message(content=content).send()


async def start_research(topic: str):
    """Start research with the given topic."""
    
    # Validate config
    try:
        config.validate_config()
    except ValueError as e:
        await cl.Message(
            content=f"""## Configuration Error

**Error:** {str(e)}

Please ensure your `.env` file is properly configured with the required API keys.

```
# Example .env configuration
GEMINI_API_KEY=your-api-key-here
MODEL_PROVIDER=gemini
MODEL_NAME=gemini-2.5-flash
```
"""
        ).send()
        return
    
    # Increment research count
    count = cl.user_session.get("research_count", 0) + 1
    cl.user_session.set("research_count", count)
    
    # Show configuration summary
    await cl.Message(
        content=f"""## Starting Research #{count}

**Topic:** *{topic}*

| Configuration | Value |
|---------------|-------|
| Model | `{config.model_name}` |
| Max Queries | `{config.max_search_queries}` |
| Max Sections | `{config.max_report_sections}` |

*Research will begin in a moment...*
"""
    ).send()
    
    # Initialize progress display
    progress_display = EnhancedProgressDisplay()
    await progress_display.initialize(topic)
    
    try:
        # Run research
        final_state = await run_research_with_updates(topic, progress_display)
        
        # Check for errors
        if final_state.get("error"):
            await cl.Message(
                content=f"""## Research Failed

**Error:** {final_state.get('error')}

Please try again or simplify your research topic.
"""
            ).send()
            return
        
        # Extract results
        search_results = final_state.get('search_results', [])
        key_findings = final_state.get('key_findings', [])
        report_sections = final_state.get('report_sections', [])
        credibility_scores = final_state.get('credibility_scores', [])
        
        # Count metrics
        unique_sources = set()
        for result in search_results:
            if hasattr(result, 'url') and result.url:
                unique_sources.add(result.url)
        
        high_cred = sum(1 for s in credibility_scores if s.get('level') == 'high')
        medium_cred = sum(1 for s in credibility_scores if s.get('level') == 'medium')
        
        # LLM metrics
        llm_calls = final_state.get('llm_calls', 0)
        total_input = final_state.get('total_input_tokens', 0)
        total_output = final_state.get('total_output_tokens', 0)
        total_tokens = total_input + total_output
        
        # Elapsed time
        elapsed = 0
        if progress_display.start_time:
            elapsed = (datetime.now() - progress_display.start_time).seconds
        
        # Create interactive summary with action buttons
        summary_actions = [
            cl.Action(
                name="view_sources",
                payload={
                    "sources": [{"title": r.title, "url": r.url} for r in search_results[:30]],
                    "credibility": credibility_scores[:30]
                },
                label="View Sources"
            ),
            cl.Action(
                name="view_findings",
                payload={"findings": key_findings[:20]},
                label="View Findings"
            )
        ]
        
        summary_content = f"""
## Research Complete!

### Research Metrics

| Category | Metric | Value |
|----------|--------|-------|
| Sources | Total Unique | **{len(unique_sources)}** |
| | High Credibility | **{high_cred}** [HIGH] |
| | Medium Credibility | **{medium_cred}** [MED] |
| Analysis | Key Insights | **{len(key_findings)}** |
| | Report Sections | **{len(report_sections)}** |
| Performance | Total Time | **{elapsed}s** |
| | LLM Calls | **{llm_calls}** |
| | Tokens Used | **{total_tokens:,}** |

---

*Click buttons below to explore the data:*
"""
        
        await cl.Message(
            content=summary_content,
            actions=summary_actions
        ).send()
        
        # Save and display report
        if final_state.get("final_report"):
            report = final_state["final_report"]
            
            # Save files
            output_dir = Path("outputs")
            output_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_topic = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in topic)
            safe_topic = safe_topic[:30].strip()
            filename = f"{safe_topic}_{timestamp}.md"
            output_file = output_dir / filename
            output_file.write_text(report, encoding='utf-8')
            
            # Export formats
            exporter = ReportExporter()
            base_path = output_file.with_suffix('')
            html_file = exporter.export(report, base_path, format='html')
            txt_file = exporter.export(report, base_path, format='txt')
            
            # Add to history
            history = ResearchHistory()
            history.add_research(
                topic=topic,
                output_file=output_file,
                metadata={
                    'sources': len(unique_sources),
                    'sections': len(report_sections),
                    'findings': len(key_findings),
                    'elapsed_seconds': elapsed,
                    'total_tokens': total_tokens
                }
            )
            
            # Download actions
            download_actions = [
                cl.Action(
                    name="download_md",
                    payload={"path": str(output_file)},
                    label="Markdown"
                ),
                cl.Action(
                    name="download_html",
                    payload={"path": str(html_file)},
                    label="HTML"
                ),
                cl.Action(
                    name="download_txt",
                    payload={"path": str(txt_file)},
                    label="Text"
                )
            ]
            
            await cl.Message(
                content=f"""## Download Report

Your report has been saved! Choose a format:

| Format | File | Size |
|--------|------|------|
| Markdown | `{filename}` | {output_file.stat().st_size:,} bytes |
| HTML | `{html_file.name}` | {html_file.stat().st_size:,} bytes |
| Plain Text | `{txt_file.name}` | {txt_file.stat().st_size:,} bytes |
""",
                actions=download_actions
            ).send()
            
            # Display the report
            await cl.Message(
                content=f"""## Full Report

---

{report}
"""
            ).send()
            
            # Prompt for next research
            next_actions = [
                cl.Action(
                    name="example_quantum",
                    payload={"topic": f"Latest developments in {topic.split()[0] if topic.split() else 'technology'}"},
                    label="Related Topic"
                )
            ]
            
            await cl.Message(
                content="""---

## Ready for More?

Type another research topic or click a suggested action above!

*Tip: You can ask follow-up questions about specific aspects of your research.*
""",
                actions=next_actions
            ).send()
        
        else:
            await cl.Message(
                content="**Warning:** No report was generated. Please try again with a different topic."
            ).send()
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        await cl.Message(
            content=f"""## Unexpected Error

**Error:** {str(e)}

<details>
<summary>Technical Details (click to expand)</summary>

```python
{error_details}
```

</details>

Please check the logs and try again.
"""
        ).send()


@cl.on_message
async def main(message: cl.Message):
    """Handle user messages."""
    topic = message.content.strip()
    
    if not topic:
        await cl.Message(
            content="Please provide a research topic."
        ).send()
        return
    
    # Check for special commands
    if topic.lower() in ["/history", "history", "show history"]:
        await on_show_history(cl.Action(name="show_history", payload={}))
        return
    
    if topic.lower() in ["/settings", "settings", "show settings"]:
        await on_show_settings(cl.Action(name="show_settings", payload={}))
        return
    
    if topic.lower() in ["/help", "help"]:
        await cl.Message(
            content="""## Help

### Commands
- `/history` - View your research history
- `/settings` - View current configuration
- `/help` - Show this help message

### Tips
- Be specific with your research topics for better results
- Use questions like "What are the latest trends in..."
- Include time context like "in 2025" for current information

### Example Topics
- "Future of quantum computing in 2025"
- "How do transformer models work?"
- "Best practices for microservices architecture"
"""
        ).send()
        return
    
    await start_research(topic)


if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)
