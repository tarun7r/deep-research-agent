"""Interactive Chainlit interface for Deep Research Agent with real-time progress updates."""

import asyncio
import chainlit as cl
from pathlib import Path
from datetime import datetime

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


# Stage markers for visual display
STAGE_ICONS = {
    ResearchStage.INITIALIZING: "[...]",
    ResearchStage.PLANNING: "[1/5]",
    ResearchStage.SEARCHING: "[2/5]",
    ResearchStage.EXTRACTING: "[3/5]",
    ResearchStage.SYNTHESIZING: "[4/5]",
    ResearchStage.WRITING: "[5/5]",
    ResearchStage.COMPLETE: "[OK]",
    ResearchStage.ERROR: "[ERR]"
}

STAGE_NAMES = {
    ResearchStage.INITIALIZING: "Initializing",
    ResearchStage.PLANNING: "Planning Research",
    ResearchStage.SEARCHING: "Searching Web",
    ResearchStage.EXTRACTING: "Extracting Content",
    ResearchStage.SYNTHESIZING: "Synthesizing Findings",
    ResearchStage.WRITING: "Writing Report",
    ResearchStage.COMPLETE: "Complete",
    ResearchStage.ERROR: "Error"
}


class ProgressDisplay:
    """Manages the progress display for a research session."""
    
    def __init__(self):
        self.message: cl.Message = None
        self.updates: list[ProgressUpdate] = []
        self.current_stage: ResearchStage = ResearchStage.INITIALIZING
        self.start_time: datetime = None
        
    async def initialize(self, topic: str):
        """Initialize the progress display."""
        self.start_time = datetime.now()
        self.updates = []
        self.current_stage = ResearchStage.INITIALIZING
        
        content = self._render_progress(topic)
        self.message = cl.Message(content=content)
        await self.message.send()
    
    async def update(self, progress_update: ProgressUpdate):
        """Update the progress display with a new update."""
        self.updates.append(progress_update)
        self.current_stage = progress_update.stage
        
        if self.message:
            self.message.content = self._render_progress()
            await self.message.update()
    
    def _render_progress(self, topic: str = None) -> str:
        """Render the progress display as markdown."""
        # Calculate elapsed time
        elapsed = ""
        if self.start_time:
            delta = datetime.now() - self.start_time
            elapsed = f" ({delta.seconds}s)"
        
        # Build progress bar
        stages_order = [
            ResearchStage.PLANNING,
            ResearchStage.SEARCHING,
            ResearchStage.EXTRACTING,
            ResearchStage.SYNTHESIZING,
            ResearchStage.WRITING,
            ResearchStage.COMPLETE
        ]
        
        # Get current progress percentage
        current_pct = 0
        if self.updates:
            for update in reversed(self.updates):
                if update.progress_pct is not None:
                    current_pct = update.progress_pct
                    break
        
        # Build visual progress bar
        bar_length = 20
        filled = int(bar_length * current_pct / 100)
        bar = "#" * filled + "-" * (bar_length - filled)
        
        content = f"""## Research Progress{elapsed}

**Progress:** [{bar}] {current_pct:.0f}%

---

"""
        
        # Show stage status
        current_stage_idx = -1
        if self.current_stage in stages_order:
            current_stage_idx = stages_order.index(self.current_stage)
        
        for idx, stage in enumerate(stages_order):
            icon = STAGE_ICONS.get(stage, "[...]")
            name = STAGE_NAMES.get(stage, stage.value)
            
            if idx < current_stage_idx:
                # Completed stage
                content += f"[DONE] ~~{name}~~\n"
            elif idx == current_stage_idx:
                # Current stage
                content += f"**{icon} {name}** <- *Current*\n"
            else:
                # Pending stage
                content += f"[ ] {name}\n"
        
        content += "\n---\n\n"
        
        # Show recent activity log (last 8 updates)
        content += "### Activity Log\n\n"
        
        if self.updates:
            recent_updates = self.updates[-8:]
            for update in reversed(recent_updates):
                icon = STAGE_ICONS.get(update.stage, "*")
                time_str = update.timestamp.strftime("%H:%M:%S")
                
                msg = f"`{time_str}` {icon} **{update.message}**"
                if update.details:
                    msg += f"\n   _{update.details}_"
                content += msg + "\n\n"
        else:
            content += "_Starting research..._\n"
        
        return content


async def run_research_with_updates(topic: str, progress_display: ProgressDisplay):
    """Run research with real-time updates to the UI."""
    
    # Reset callback state
    progress_callback.reset()
    
    # Register async callback for UI updates
    async def on_progress(update: ProgressUpdate):
        await progress_display.update(update)
    
    progress_callback.register_async(on_progress)
    
    try:
        # Initialize state
        initial_state = ResearchState(research_topic=topic)
        
        # Create graph
        graph = create_research_graph()
        
        # Execute workflow and get final state
        final_state = await graph.ainvoke(initial_state)
        
        # Emit completion
        search_results = final_state.get('search_results', [])
        key_findings = final_state.get('key_findings', [])
        await emit_complete(topic, len(search_results), len(key_findings))
        
        return final_state
        
    finally:
        # Cleanup callback
        progress_callback.unregister(on_progress)


@cl.on_chat_start
async def start():
    """Initialize the chat session."""
    await cl.Message(
        content="""# Deep Research Agent

Welcome! I'm your AI research assistant powered by **LangGraph** and **Gemini**.

## How it works:
1. **Tell me** what you want to research
2. I'll **search** the web for authoritative sources
3. **Synthesize** findings using AI
4. **Generate** a comprehensive report

## Features:
- Real-time web search with DuckDuckGo
- Source credibility scoring
- Multiple export formats (MD, HTML, TXT)
- Live progress tracking

---

**What would you like to research today?** 

_Example topics:_
- "Future of quantum computing in 2025"
- "How does WebSocket streaming work?"
- "Best practices for microservices architecture"
""",
        author="Research Agent"
    ).send()


@cl.on_message
async def main(message: cl.Message):
    """Handle user messages."""
    
    topic = message.content.strip()
    
    if not topic:
        await cl.Message(
            content="WARNING: Please provide a research topic.",
            author="System"
        ).send()
        return
    
    # Validate config
    try:
        config.validate_config()
    except ValueError as e:
        await cl.Message(
            content=f"**Configuration Error:** {str(e)}\n\n"
                    "Please set your API key in the `.env` file.",
            author="System"
        ).send()
        return
    
    # Show starting message
    await cl.Message(
        content=f"""## Starting Research

**Topic:** _{topic}_

**Configuration:**
- Model: `{config.model_name}`
- Max Queries: `{config.max_search_queries}`
- Max Sections: `{config.max_report_sections}`

_Research will begin shortly..._
""",
        author="Research Agent"
    ).send()
    
    # Initialize progress display
    progress_display = ProgressDisplay()
    await progress_display.initialize(topic)
    
    try:
        # Run research with updates
        final_state = await run_research_with_updates(topic, progress_display)
        
        # Check for errors
        if final_state.get("error"):
            await cl.Message(
                content=f"## Research Failed\n\n{final_state.get('error')}",
                author="System"
            ).send()
            return
        
        # Display detailed summary with metrics
        search_results = final_state.get('search_results', [])
        key_findings = final_state.get('key_findings', [])
        report_sections = final_state.get('report_sections', [])
        credibility_scores = final_state.get('credibility_scores', [])
        
        # Count unique sources
        unique_sources = set()
        for result in search_results:
            if hasattr(result, 'url') and result.url:
                unique_sources.add(result.url)
        
        # Count high-credibility sources
        high_cred_count = sum(1 for score in credibility_scores if score.get('level') == 'high')
        medium_cred_count = sum(1 for score in credibility_scores if score.get('level') == 'medium')
        
        # Get LLM tracking info
        llm_calls = final_state.get('llm_calls', 0)
        total_input_tokens = final_state.get('total_input_tokens', 0)
        total_output_tokens = final_state.get('total_output_tokens', 0)
        total_tokens = total_input_tokens + total_output_tokens
        
        # Calculate elapsed time
        elapsed_seconds = 0
        if progress_display.start_time:
            elapsed_seconds = (datetime.now() - progress_display.start_time).seconds
        
        summary_content = f"""## Research Complete!

### Data Collected
| Metric | Value |
|--------|-------|
| Unique Sources | **{len(unique_sources)}** |
| High Credibility | **{high_cred_count}** |
| Medium Credibility | **{medium_cred_count}** |
| Key Insights | **{len(key_findings)}** |
| Report Sections | **{len(report_sections)}** |

### Performance
| Metric | Value |
|--------|-------|
| Total Time | **{elapsed_seconds}s** |
| LLM Calls | **{llm_calls}** |
| Input Tokens | **{total_input_tokens:,}** |
| Output Tokens | **{total_output_tokens:,}** |
| Total Tokens | **{total_tokens:,}** |
"""
        
        await cl.Message(
            content=summary_content,
            author="Research Agent"
        ).send()
        
        # Save and display report
        if final_state.get("final_report"):
            report = final_state["final_report"]
            
            # Save to file
            output_dir = Path("outputs")
            output_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_topic = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in topic)
            safe_topic = safe_topic[:30].strip()
            filename = f"{safe_topic}_{timestamp}.md"
            output_file = output_dir / filename
            output_file.write_text(report, encoding='utf-8')
            
            # Add to history
            history = ResearchHistory()
            history.add_research(
                topic=topic,
                output_file=output_file,
                metadata={
                    'sources': len(unique_sources),
                    'sections': len(report_sections),
                    'findings': len(key_findings),
                    'elapsed_seconds': elapsed_seconds,
                    'total_tokens': total_tokens
                }
            )
            
            report_header = f"""## Final Report

**Report Statistics:**
- Length: **{len(report):,}** characters
- Saved to: `{output_file}`

---

{report}"""
            
            await cl.Message(
                content=report_header,
                author="Research Agent"
            ).send()
            
            # Export to multiple formats
            exporter = ReportExporter()
            base_path = output_file.with_suffix('')
            
            # Export HTML
            html_file = exporter.export(report, base_path, format='html')
            
            # Export TXT
            txt_file = exporter.export(report, base_path, format='txt')
            
            # Offer downloads
            elements = [
                cl.File(
                    name=filename,
                    path=str(output_file),
                    display="inline"
                ),
                cl.File(
                    name=html_file.name,
                    path=str(html_file),
                    display="inline"
                ),
                cl.File(
                    name=txt_file.name,
                    path=str(txt_file),
                    display="inline"
                )
            ]
            
            await cl.Message(
                content=f"""## Download Report

Download your report in multiple formats:

| Format | File |
|--------|------|
| Markdown | `{filename}` |
| HTML | `{html_file.name}` |
| Plain Text | `{txt_file.name}` |
""",
                elements=elements,
                author="Research Agent"
            ).send()
            
            # Ask for next research with suggestions
            await cl.Message(
                content="""---

## Ready for Another Research?

Type your next research topic below, or try one of these:

- *"Future trends in [your industry]"*
- *"Comparative analysis of [topic A] vs [topic B]"*
- *"Best practices for [specific challenge]"*
- *"Impact of [technology/trend] on [domain]"*

**What would you like to research next?**""",
                author="Research Agent"
            ).send()
        else:
            await cl.Message(
                content="WARNING: No report was generated. Please try again.",
                author="System"
            ).send()
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        await cl.Message(
            content=f"""## Unexpected Error

**Error:** {str(e)}

<details>
<summary>Technical Details</summary>

```
{error_details}
```

</details>

Please check the logs and try again.
""",
            author="System"
        ).send()


if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)
