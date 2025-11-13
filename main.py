"""Main entry point for the Deep Research Agent."""

import asyncio
import sys
from pathlib import Path
import logging

from src.config import config
from src.graph import run_research

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Main function to run the research agent."""
    
    # Validate configuration
    try:
        config.validate_config()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    
    # Get research topic
    if len(sys.argv) > 1:
        topic = " ".join(sys.argv[1:])
    else:
        print("\nDeep Research Agent")
        print("=" * 50)
        topic = input("\nEnter your research topic: ").strip()
    
    if not topic:
        logger.error("No research topic provided")
        sys.exit(1)
    
    print(f"\n[INFO] Starting deep research on: {topic}\n")
    print("This may take several minutes. Please wait...\n")
    
    try:
        # Run the research workflow
        final_state = await run_research(topic, verbose=True)
        
        # LangGraph returns dict with state - access fields directly
        # Check for errors
        if final_state.get("error"):
            logger.error(f"Research failed: {final_state.get('error')}")
            sys.exit(1)
        
        # Display results
        print("\n" + "=" * 80)
        print("RESEARCH COMPLETE")
        print("=" * 80)
        
        if final_state.get("plan"):
            plan = final_state["plan"]
            print(f"\nResearch Plan Summary:")
            print(f"  - Objectives: {len(plan.objectives)}")
            print(f"  - Search Queries: {len(plan.search_queries)}")
            print(f"  - Report Sections: {len(plan.report_outline)}")
        
        print(f"\nResearch Data Summary:")
        print(f"  - Search Results: {len(final_state.get('search_results', []))}")
        print(f"  - Key Findings: {len(final_state.get('key_findings', []))}")
        print(f"  - Report Sections: {len(final_state.get('report_sections', []))}")
        print(f"  - Iterations: {final_state.get('iterations', 0)}")
        
        # Save the report
        if final_state.get("final_report"):
            output_dir = Path("outputs")
            output_dir.mkdir(exist_ok=True)
            
            # Create safe filename
            safe_topic = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in topic)
            safe_topic = safe_topic[:50].strip()
            
            output_file = output_dir / f"{safe_topic}.md"
            final_report = final_state["final_report"]
            output_file.write_text(final_report, encoding='utf-8')
            
            print(f"\n[SUCCESS] Report saved to: {output_file}")
            print(f"          Report length: {len(final_report)} characters")
            
            # Display a preview
            print("\n" + "=" * 80)
            print("REPORT PREVIEW")
            print("=" * 80)
            print(final_report[:1500])
            if len(final_report) > 1500:
                print(f"\n... (showing first 1500 of {len(final_report)} characters)")
            print("\n" + "=" * 80)
            
        else:
            logger.warning("No report was generated")
        
    except KeyboardInterrupt:
        print("\n\n[WARNING] Research interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"[ERROR] Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

