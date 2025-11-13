"""Export research reports to various formats."""

from pathlib import Path
from typing import Optional
import logging
import markdown
from datetime import datetime

logger = logging.getLogger(__name__)


class ReportExporter:
    """Export reports to various formats."""
    
    def __init__(self):
        self.supported_formats = ['markdown', 'html', 'txt']
    
    def export_markdown(self, content: str, output_path: Path) -> Path:
        """Export as markdown (already in markdown format)."""
        output_path.write_text(content, encoding='utf-8')
        logger.info(f"Exported markdown to {output_path}")
        return output_path
    
    def export_html(self, content: str, output_path: Path) -> Path:
        """Export as HTML with styling."""
        # Convert markdown to HTML
        html_content = markdown.markdown(
            content,
            extensions=['extra', 'codehilite', 'tables']
        )
        
        # Wrap in styled HTML template
        html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Research Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
            background-color: #fff;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 5px;
        }}
        h3 {{
            color: #555;
            margin-top: 25px;
        }}
        code {{
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}
        pre {{
            background-color: #f4f4f4;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }}
        blockquote {{
            border-left: 4px solid #3498db;
            margin: 0;
            padding-left: 20px;
            color: #666;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        a {{
            color: #3498db;
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ecf0f1;
            color: #7f8c8d;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    {html_content}
    <div class="footer">
        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
</body>
</html>"""
        
        output_path.write_text(html_template, encoding='utf-8')
        logger.info(f"Exported HTML to {output_path}")
        return output_path
    
    def export_txt(self, content: str, output_path: Path) -> Path:
        """Export as plain text (strip markdown)."""
        import re
        # Remove markdown formatting
        text = content
        # Remove headers
        text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
        # Remove bold/italic
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        # Remove links but keep text
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        # Remove code blocks
        text = re.sub(r'```[^`]+```', '', text, flags=re.DOTALL)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        
        output_path.write_text(text, encoding='utf-8')
        logger.info(f"Exported text to {output_path}")
        return output_path
    
    def export(self, content: str, output_path: Path, format: str = 'markdown') -> Path:
        """Export content to specified format.
        
        Args:
            content: Report content (markdown)
            output_path: Output file path
            format: Export format ('markdown', 'html', 'txt')
        
        Returns:
            Path to exported file
        """
        format = format.lower()
        
        if format not in self.supported_formats:
            raise ValueError(f"Unsupported format: {format}. Supported: {self.supported_formats}")
        
        # Adjust file extension if needed
        if format == 'html' and not output_path.suffix == '.html':
            output_path = output_path.with_suffix('.html')
        elif format == 'txt' and not output_path.suffix == '.txt':
            output_path = output_path.with_suffix('.txt')
        elif format == 'markdown' and not output_path.suffix in ['.md', '.markdown']:
            output_path = output_path.with_suffix('.md')
        
        if format == 'markdown':
            return self.export_markdown(content, output_path)
        elif format == 'html':
            return self.export_html(content, output_path)
        elif format == 'txt':
            return self.export_txt(content, output_path)
        else:
            raise ValueError(f"Export not implemented for format: {format}")

