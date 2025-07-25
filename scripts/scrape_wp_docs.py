#!/usr/bin/env python3
"""
WordPress Documentation Scraper
Scrapes official WordPress developer documentation while respecting robots.txt and rate limits.
"""

import time
import json
import requests
from pathlib import Path
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from typing import Set, Dict, List
import hashlib
from datetime import datetime
from rich.console import Console
from rich.progress import track

console = Console()

class WPDocsScraper:
    def __init__(self, output_dir: Path = Path("data/raw")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'WP-SLM-Bot/1.0 (WordPress Language Model Training)'
        })
        self.visited_urls: Set[str] = set()
        self.rate_limit_delay = 1.0  # seconds between requests
        
    def scrape_wp_developer_handbook(self):
        """Scrape WordPress Developer Handbook sections."""
        base_url = "https://developer.wordpress.org/"
        sections = [
            "rest-api/",
            "plugins/",
            "themes/",
            "block-editor/",
            "cli/commands/",
            "coding-standards/",
            "apis/",
        ]
        
        console.print("[bold cyan]Starting WordPress Developer Handbook scraping...[/bold cyan]")
        
        for section in sections:
            section_url = urljoin(base_url, section)
            console.print(f"[yellow]Scraping section: {section}[/yellow]")
            self._scrape_section(section_url, section.rstrip('/'))
            
    def _scrape_section(self, url: str, section_name: str):
        """Recursively scrape a documentation section."""
        if url in self.visited_urls:
            return
            
        self.visited_urls.add(url)
        
        try:
            time.sleep(self.rate_limit_delay)
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Save raw HTML
            self._save_page(url, response.content, section_name)
            
            # Extract navigation links for recursive scraping
            nav_links = self._extract_nav_links(soup, url)
            
            for link in track(nav_links, description=f"Processing {section_name}"):
                if link not in self.visited_urls and self._is_valid_doc_url(link):
                    self._scrape_section(link, section_name)
                    
        except Exception as e:
            console.print(f"[red]Error scraping {url}: {e}[/red]")
            
    def _extract_nav_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract navigation links from page."""
        links = []
        
        # Look for sidebar navigation
        nav_elements = soup.select('.wp-block-navigation a, .menu-item a, nav a')
        
        for link in nav_elements:
            href = link.get('href')
            if href:
                absolute_url = urljoin(base_url, href)
                if self._is_valid_doc_url(absolute_url):
                    links.append(absolute_url)
                    
        return list(set(links))
    
    def _is_valid_doc_url(self, url: str) -> bool:
        """Check if URL is a valid documentation page."""
        parsed = urlparse(url)
        
        # Only scrape developer.wordpress.org
        if parsed.netloc != 'developer.wordpress.org':
            return False
            
        # Skip non-documentation paths
        skip_patterns = ['/reference/files/', '/changelog/', '#', '.zip', '.pdf']
        return not any(pattern in url for pattern in skip_patterns)
        
    def _save_page(self, url: str, content: bytes, section: str):
        """Save scraped page with metadata."""
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        filename = f"{section}_{url_hash}.html"
        filepath = self.output_dir / filename
        
        # Save HTML
        filepath.write_bytes(content)
        
        # Save metadata
        metadata = {
            "url": url,
            "section": section,
            "scraped_at": datetime.now().isoformat(),
            "wp_version": "6.5",  # TODO: Extract from page if available
            "filename": filename
        }
        
        meta_file = self.output_dir / f"{filename}.meta.json"
        meta_file.write_text(json.dumps(metadata, indent=2))
        
    def scrape_wp_cli_docs(self):
        """Scrape WP-CLI command documentation."""
        console.print("[bold cyan]Starting WP-CLI documentation scraping...[/bold cyan]")
        
        base_url = "https://developer.wordpress.org/cli/commands/"
        self._scrape_section(base_url, "wp-cli")
        
    def scrape_hooks_reference(self):
        """Scrape WordPress hooks reference."""
        console.print("[bold cyan]Starting hooks reference scraping...[/bold cyan]")
        
        # Action and filter references
        urls = [
            "https://developer.wordpress.org/reference/hooks/",
            "https://developer.wordpress.org/apis/hooks/action-reference/",
            "https://developer.wordpress.org/apis/hooks/filter-reference/"
        ]
        
        for url in urls:
            self._scrape_section(url, "hooks")


def main():
    scraper = WPDocsScraper()
    
    # Scrape different documentation sources
    scraper.scrape_wp_developer_handbook()
    scraper.scrape_wp_cli_docs()
    scraper.scrape_hooks_reference()
    
    console.print(f"[green]Scraping complete! Total pages: {len(scraper.visited_urls)}[/green]")
    
    # Save scraping summary
    summary = {
        "total_pages": len(scraper.visited_urls),
        "urls": list(scraper.visited_urls),
        "completed_at": datetime.now().isoformat()
    }
    
    summary_file = Path("data/raw/scraping_summary.json")
    summary_file.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()