#!/usr/bin/env python3
"""
WordPress Documentation Parser
Converts scraped HTML documentation to clean, structured markdown.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional
from bs4 import BeautifulSoup, NavigableString
import markdownify
from rich.console import Console
from rich.progress import track
import hashlib

console = Console()

class WPDocsParser:
    def __init__(self, input_dir: Path = Path("data/raw"), 
                 output_dir: Path = Path("data/processed")):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def parse_all_docs(self):
        """Parse all HTML files in the raw directory."""
        html_files = list(self.input_dir.glob("*.html"))
        console.print(f"[cyan]Found {len(html_files)} HTML files to parse[/cyan]")
        
        parsed_count = 0
        for html_file in track(html_files, description="Parsing documents"):
            if self._parse_html_file(html_file):
                parsed_count += 1
                
        console.print(f"[green]Successfully parsed {parsed_count} documents[/green]")
        
    def _parse_html_file(self, html_file: Path) -> bool:
        """Parse a single HTML file to markdown."""
        try:
            # Load metadata
            meta_file = html_file.with_suffix('.html.meta.json')
            if meta_file.exists():
                metadata = json.loads(meta_file.read_text())
            else:
                metadata = {"url": "unknown", "section": "unknown"}
            
            # Parse HTML
            html_content = html_file.read_text(encoding='utf-8')
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract main content
            content = self._extract_main_content(soup)
            if not content:
                console.print(f"[yellow]No main content found in {html_file.name}[/yellow]")
                return False
                
            # Convert to markdown
            markdown = self._html_to_markdown(content)
            
            # Clean and structure markdown
            structured_content = self._structure_markdown(markdown, metadata)
            
            # Save processed file
            output_file = self.output_dir / f"{html_file.stem}.md"
            output_file.write_text(structured_content, encoding='utf-8')
            
            # Save structured metadata
            self._save_structured_metadata(html_file.stem, metadata, structured_content)
            
            return True
            
        except Exception as e:
            console.print(f"[red]Error parsing {html_file.name}: {e}[/red]")
            return False
            
    def _extract_main_content(self, soup: BeautifulSoup) -> Optional[BeautifulSoup]:
        """Extract the main documentation content from the page."""
        # Try different content selectors
        selectors = [
            'main.site-main',
            'article.page',
            'div.entry-content',
            'div.content-area',
            'div#content',
            'main'
        ]
        
        for selector in selectors:
            content = soup.select_one(selector)
            if content:
                # Remove navigation, sidebars, comments
                for element in content.select('.navigation, .sidebar, .comments, nav, aside'):
                    element.decompose()
                return content
                
        return None
        
    def _html_to_markdown(self, html_content: BeautifulSoup) -> str:
        """Convert HTML content to markdown with custom rules."""
        # Pre-process code blocks
        self._process_code_blocks(html_content)
        
        # Convert to markdown
        markdown = markdownify.markdownify(
            str(html_content),
            heading_style="ATX",
            code_language="php",  # default for WP
            strip=['nav', 'footer', 'aside']
        )
        
        # Post-process markdown
        markdown = self._clean_markdown(markdown)
        
        return markdown
        
    def _process_code_blocks(self, soup: BeautifulSoup):
        """Pre-process code blocks to preserve formatting."""
        for pre in soup.find_all('pre'):
            # Detect language from class
            code_elem = pre.find('code')
            if code_elem:
                classes = code_elem.get('class', [])
                lang = 'php'  # default
                
                for cls in classes:
                    if 'language-' in cls:
                        lang = cls.replace('language-', '')
                        break
                        
                # Add language marker
                code_elem['data-lang'] = lang
                
    def _clean_markdown(self, markdown: str) -> str:
        """Clean up converted markdown."""
        # Remove excessive newlines
        markdown = re.sub(r'\n{3,}', '\n\n', markdown)
        
        # Fix code block formatting
        markdown = re.sub(r'```\n\n', '```\n', markdown)
        markdown = re.sub(r'\n\n```', '\n```', markdown)
        
        # Remove HTML comments
        markdown = re.sub(r'<!--.*?-->', '', markdown, flags=re.DOTALL)
        
        # Fix WordPress shortcodes that shouldn't be in docs
        markdown = re.sub(r'\[/?[^\]]+\]', '', markdown)
        
        return markdown.strip()
        
    def _structure_markdown(self, markdown: str, metadata: Dict) -> str:
        """Add structure and metadata to markdown."""
        # Extract title from first heading
        title_match = re.search(r'^#\s+(.+)$', markdown, re.MULTILINE)
        title = title_match.group(1) if title_match else "Untitled"
        
        # Build structured document
        structured = f"""---
title: {title}
source_url: {metadata.get('url', 'unknown')}
section: {metadata.get('section', 'unknown')}
wp_version: {metadata.get('wp_version', '6.5')}
---

{markdown}
"""
        
        return structured
        
    def _save_structured_metadata(self, doc_id: str, original_meta: Dict, content: str):
        """Save enhanced metadata for the parsed document."""
        # Extract additional metadata from content
        sections = re.findall(r'^#{1,3}\s+(.+)$', content, re.MULTILINE)
        code_blocks = len(re.findall(r'```', content))
        
        # Detect content type
        content_type = self._detect_content_type(content)
        
        # Calculate content hash for deduplication
        content_hash = hashlib.md5(content.encode()).hexdigest()
        
        enhanced_meta = {
            **original_meta,
            "doc_id": doc_id,
            "sections": sections[:10],  # top 10 sections
            "code_blocks": code_blocks,
            "content_type": content_type,
            "word_count": len(content.split()),
            "content_hash": content_hash
        }
        
        meta_file = self.output_dir / f"{doc_id}.meta.json"
        meta_file.write_text(json.dumps(enhanced_meta, indent=2))
        
    def _detect_content_type(self, content: str) -> str:
        """Detect the type of documentation content."""
        content_lower = content.lower()
        
        if 'function' in content_lower and 'parameters' in content_lower:
            return "reference"
        elif 'how to' in content_lower or 'step' in content_lower:
            return "tutorial"
        elif 'troubleshoot' in content_lower or 'error' in content_lower:
            return "troubleshooting"
        elif 'security' in content_lower or 'vulnerability' in content_lower:
            return "security"
        elif 'rest api' in content_lower or 'endpoint' in content_lower:
            return "api"
        else:
            return "general"


def main():
    parser = WPDocsParser()
    parser.parse_all_docs()
    
    # Generate parsing summary
    processed_files = list(Path("data/processed").glob("*.md"))
    
    summary = {
        "total_parsed": len(processed_files),
        "content_types": {},
        "sections": {}
    }
    
    # Analyze parsed content
    for md_file in processed_files:
        meta_file = md_file.with_suffix('.md.meta.json')
        if meta_file.exists():
            meta = json.loads(meta_file.read_text())
            content_type = meta.get('content_type', 'unknown')
            section = meta.get('section', 'unknown')
            
            summary['content_types'][content_type] = summary['content_types'].get(content_type, 0) + 1
            summary['sections'][section] = summary['sections'].get(section, 0) + 1
            
    summary_file = Path("data/processed/parsing_summary.json")
    summary_file.write_text(json.dumps(summary, indent=2))
    
    console.print("[green]Parsing complete![/green]")
    console.print(f"Content types: {summary['content_types']}")
    console.print(f"Sections: {summary['sections']}")


if __name__ == "__main__":
    main()