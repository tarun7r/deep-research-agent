"""Interactive Chainlit interface for Deep Research Agent."""

import asyncio
import chainlit as cl
from pathlib import Path
from datetime import datetime

from src.config import config
from src.state import ResearchState
from src.graph import create_research_graph
from src.utils.exports import ReportExporter
from src.utils.history import ResearchHistory


async def run_research_with_updates(topic: str):
    """Run research with real-time updates to the UI."""
    
    # Initialize state
    initial_state = ResearchState(research_topic=topic)
    
    # Create graph
    graph = create_research_graph()
    
    # Create progress message with spinner
    progress_msg = cl.Message(content="")
    await progress_msg.send()
    
    stages = {
        "plan": {"name": "Planning Research", "done": False, "details": ""},
        "search": {"name": "Searching Web", "done": False, "details": ""},
        "synthesize": {"name": "Synthesizing Findings", "done": False, "details": ""},
        "write_report": {"name": "Writing Report", "done": False, "details": ""}
    }
    
    def update_progress():
        """Update the progress display with details."""
        content = "### Research Progress\n\n"
        for stage_key, stage_info in stages.items():
            if stage_info["done"]:
                status = f"**{stage_info['name']}**"
                if stage_info.get("details"):
                    status += f" - {stage_info['details']}"
                content += f"{status}\n"
            elif any(s["done"] for s in list(stages.values())[:list(stages.keys()).index(stage_key)]):
                content += f"**{stage_info['name']}** (in progress...)\n"
            else:
                content += f"{stage_info['name']}\n"
        return content
    
    # Update progress initially
    progress_msg.content = update_progress()
    await progress_msg.update()
    
    # Execute workflow and get final state
    final_state = await graph.ainvoke(initial_state)
    
    # Update with actual results
    if final_state.get("plan"):
        plan = final_state["plan"]
        stages["plan"]["done"] = True
        stages["plan"]["details"] = f"{len(plan.search_queries)} queries generated"
    
    if final_state.get("search_results"):
        stages["search"]["done"] = True
        stages["search"]["details"] = f"{len(final_state['search_results'])} sources found"
    
    if final_state.get("key_findings"):
        stages["synthesize"]["done"] = True
        stages["synthesize"]["details"] = f"{len(final_state['key_findings'])} insights extracted"
    
    if final_state.get("final_report"):
        stages["write_report"]["done"] = True
        stages["write_report"]["details"] = f"{len(final_state['final_report'])} characters"
    
    # Mark all stages as complete
    for stage in stages.values():
        stage["done"] = True
    
    progress_msg.content = update_progress() + "\n\n**All research stages completed successfully!**"
    await progress_msg.update()
    
    return final_state


@cl.on_chat_start
async def start():
    """Initialize the chat session."""
    await cl.Message(
        content="# Deep Research Agent\n\n"
                "Welcome! I'm your AI research assistant powered by LangGraph.\n\n"
                "**How it works:**\n"
                "1. Tell me what you want to research\n"
                "2. I'll create a research plan\n"
                "3. Search the web using DuckDuckGo\n"
                "4. Synthesize findings with Gemini AI\n"
                "5. Generate a comprehensive report\n\n"
                "**What would you like to research today?**",
        author="Research Agent"
    ).send()


@cl.on_message
async def main(message: cl.Message):
    """Handle user messages."""
    
    topic = message.content.strip()
    
    if not topic:
        await cl.Message(
            content="Please provide a research topic.",
            author="System"
        ).send()
        return
    
    # Validate config
    try:
        config.validate_config()
    except ValueError as e:
        await cl.Message(
            content=f"Configuration Error: {str(e)}\n\n"
                    "Please set your GEMINI_API_KEY in the .env file.",
            author="System"
        ).send()
        return
    
    # Show starting message
    await cl.Message(
        content=f"**Starting Research**\n\n"
                f"Topic: *{topic}*\n\n"
                f"This will take 2-5 minutes. Please wait...",
        author="Research Agent"
    ).send()
    
    try:
        # Run research with updates
        final_state = await run_research_with_updates(topic)
        
        # Check for errors
        if final_state.get("error"):
            await cl.Message(
                content=f"**Research Failed**\n\n{final_state.get('error')}",
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
        
        # Get LLM tracking info
        llm_calls = final_state.get('llm_calls', 0)
        total_input_tokens = final_state.get('total_input_tokens', 0)
        total_output_tokens = final_state.get('total_output_tokens', 0)
        total_tokens = total_input_tokens + total_output_tokens
        
        summary_content = f"""### Research Summary

**Data Collected:**
- **Sources:** {len(unique_sources)} unique websites ({high_cred_count} high-credibility)
- **Search Results:** {len(search_results)} total results
- **Key Insights:** {len(key_findings)} findings extracted
- **Report Sections:** {len(report_sections)} sections generated
- **Processing Iterations:** {final_state.get('iterations', 0)}

**Report Statistics:**
- **Total Length:** {len(final_state.get('final_report', ''))} characters
- **Research Time:** ~{final_state.get('iterations', 0) * 15} seconds

**LLM Usage:**
- **API Calls:** {llm_calls} calls
- **Input Tokens:** {total_input_tokens:,} tokens
- **Output Tokens:** {total_output_tokens:,} tokens
- **Total Tokens:** {total_tokens:,} tokens
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
                    'findings': len(key_findings)
                }
            )
            
            preview = report
            
            report_header = f"""### Final Report Generated

**Report Details:**
- Length: {len(report):,} characters
- Saved to: `{output_file}`
- Format: Markdown

---

{preview}"""
            
            
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
                content="### Download Report\n\nDownload the report in multiple formats:\n"
                       f"- Markdown: `{filename}`\n"
                       f"- HTML: `{html_file.name}`\n"
                       f"- Plain Text: `{txt_file.name}`",
                elements=elements,
                author="Research Agent"
            ).send()
            
            # Ask for next research with suggestions
            await cl.Message(
                content="""---

### Ready for Another Research?

Type your next research topic below, or try these suggestions:
- "Future trends in [your industry]"
- "Comparative analysis of [topic A] vs [topic B]"
- "Best practices for [specific challenge]"
- "Impact of [technology/trend] on [domain]"

**What would you like to research next?**""",
                author="Research Agent"
            ).send()
        else:
            await cl.Message(
                content="No report was generated. Please try again.",
                author="System"
            ).send()
    
    except Exception as e:
        await cl.Message(
            content=f"**Unexpected Error**\n\n{str(e)}\n\n"
                    "Please check the logs and try again.",
            author="System"
        ).send()


if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)

