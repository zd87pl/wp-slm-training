#!/usr/bin/env python3
"""
Build SFT (Supervised Fine-Tuning) instruction-response pairs from parsed documentation.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import random
from rich.console import Console
from rich.progress import track
import hashlib

console = Console()

@dataclass
class InstructionPair:
    id: str
    source_url: str
    license: str
    prompt: str
    response: str
    tags: List[str]
    wp_version: str
    meta: Dict


class SFTBuilder:
    def __init__(self, input_dir: Path = Path("data/processed"),
                 output_dir: Path = Path("data/sft")):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load prompt templates
        self.prompt_templates = self._load_prompt_templates()
        
    def _load_prompt_templates(self) -> Dict[str, List[str]]:
        """Load prompt templates for different content types."""
        return {
            "howto": [
                "How do I {action} in WordPress?",
                "What's the best way to {action} using WordPress?",
                "Can you show me how to {action} in WordPress with code examples?",
                "I need to {action} in my WordPress site. How should I do it?",
                "Explain how to {action} in WordPress step by step."
            ],
            "reference": [
                "What does the {function} function do in WordPress?",
                "Explain the {function} WordPress function with examples.",
                "What are the parameters for {function} in WordPress?",
                "How do I use the {function} function correctly?",
                "Show me the syntax and usage of {function} in WordPress."
            ],
            "api": [
                "How do I {action} using the WordPress REST API?",
                "Give me a REST API example to {action} in WordPress.",
                "What's the endpoint to {action} in WordPress REST API?",
                "Show me how to authenticate and {action} via WP REST API.",
                "Write a cURL command to {action} using WordPress REST API."
            ],
            "hook": [
                "How do I use the {hook} hook in WordPress?",
                "What's the purpose of the {hook} action/filter?",
                "Show me an example of using {hook} in a plugin.",
                "When should I use the {hook} hook?",
                "What parameters does the {hook} hook accept?"
            ],
            "security": [
                "How do I secure {feature} in WordPress?",
                "What are the security best practices for {feature}?",
                "How can I prevent {vulnerability} in WordPress?",
                "Show me how to implement secure {feature} in WordPress.",
                "What security measures should I take when {action}?"
            ],
            "troubleshooting": [
                "I'm getting {error} in WordPress. How do I fix it?",
                "My WordPress site shows {error}. What's the solution?",
                "How do I debug {issue} in WordPress?",
                "What causes {error} and how can I resolve it?",
                "Help me troubleshoot {issue} in my WordPress installation."
            ]
        }
        
    def build_all_pairs(self):
        """Build instruction pairs from all processed documents."""
        md_files = list(self.input_dir.glob("*.md"))
        console.print(f"[cyan]Processing {len(md_files)} documents[/cyan]")
        
        all_pairs = []
        
        for md_file in track(md_files, description="Building instruction pairs"):
            pairs = self._build_pairs_from_doc(md_file)
            all_pairs.extend(pairs)
            
        console.print(f"[green]Generated {len(all_pairs)} instruction pairs[/green]")
        
        # Save pairs
        self._save_pairs(all_pairs)
        
    def _build_pairs_from_doc(self, md_file: Path) -> List[InstructionPair]:
        """Build instruction pairs from a single document."""
        try:
            # Load document and metadata
            content = md_file.read_text(encoding='utf-8')
            meta_file = md_file.with_suffix('.md.meta.json')
            
            if meta_file.exists():
                metadata = json.loads(meta_file.read_text())
            else:
                metadata = {}
                
            # Parse front matter
            front_matter, body = self._parse_front_matter(content)
            
            # Extract sections
            sections = self._extract_sections(body)
            
            # Generate pairs for each section
            pairs = []
            for section in sections:
                section_pairs = self._generate_pairs_for_section(
                    section, metadata, front_matter
                )
                pairs.extend(section_pairs)
                
            return pairs
            
        except Exception as e:
            console.print(f"[red]Error processing {md_file.name}: {e}[/red]")
            return []
            
    def _parse_front_matter(self, content: str) -> Tuple[Dict, str]:
        """Parse YAML front matter from markdown."""
        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                # Simple YAML parsing
                front_matter = {}
                for line in parts[1].strip().split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        front_matter[key.strip()] = value.strip()
                return front_matter, parts[2]
                
        return {}, content
        
    def _extract_sections(self, content: str) -> List[Dict]:
        """Extract sections from markdown content."""
        sections = []
        
        # Split by headings
        heading_pattern = r'^(#{1,3})\s+(.+)$'
        parts = re.split(heading_pattern, content, flags=re.MULTILINE)
        
        # Group into sections
        i = 1  # Skip first empty part
        while i < len(parts) - 2:
            level = len(parts[i])
            title = parts[i + 1]
            content = parts[i + 2] if i + 2 < len(parts) else ""
            
            sections.append({
                'level': level,
                'title': title,
                'content': content.strip()
            })
            
            i += 3
            
        return sections
        
    def _generate_pairs_for_section(self, section: Dict, metadata: Dict, 
                                   front_matter: Dict) -> List[InstructionPair]:
        """Generate instruction pairs for a section."""
        pairs = []
        
        # Determine content type
        content_type = self._detect_section_type(section)
        
        # Skip if section is too short
        if len(section['content']) < 100:
            return pairs
            
        # Extract key information
        extracted = self._extract_key_info(section, content_type)
        
        # Generate prompts based on content type
        if content_type in self.prompt_templates and extracted:
            templates = self.prompt_templates[content_type]
            
            # Create 1-3 variations
            num_variations = min(3, len(templates))
            selected_templates = random.sample(templates, num_variations)
            
            for template in selected_templates:
                prompt = self._fill_template(template, extracted)
                if prompt:
                    pair = self._create_pair(
                        prompt, section['content'], metadata, 
                        front_matter, content_type, extracted
                    )
                    pairs.append(pair)
                    
        return pairs
        
    def _detect_section_type(self, section: Dict) -> str:
        """Detect the type of content in a section."""
        title_lower = section['title'].lower()
        content_lower = section['content'].lower()
        
        if any(word in title_lower for word in ['function', 'method', 'class']):
            return "reference"
        elif 'hook' in title_lower or 'filter' in title_lower or 'action' in title_lower:
            return "hook"
        elif 'api' in title_lower or 'endpoint' in content_lower:
            return "api"
        elif any(word in content_lower for word in ['security', 'sanitize', 'escape', 'nonce']):
            return "security"
        elif any(word in content_lower for word in ['error', 'fix', 'troubleshoot', 'debug']):
            return "troubleshooting"
        else:
            return "howto"
            
    def _extract_key_info(self, section: Dict, content_type: str) -> Optional[Dict]:
        """Extract key information based on content type."""
        extracted = {}
        
        if content_type == "reference":
            # Extract function name
            func_match = re.search(r'`(\w+)\(`', section['content'])
            if func_match:
                extracted['function'] = func_match.group(1)
                
        elif content_type == "hook":
            # Extract hook name
            hook_match = re.search(r'`(\w+)`\s*(action|filter|hook)', section['content'])
            if hook_match:
                extracted['hook'] = hook_match.group(1)
                
        elif content_type == "api":
            # Extract action
            extracted['action'] = self._extract_action_from_title(section['title'])
            
        elif content_type == "security":
            # Extract feature or vulnerability
            extracted['feature'] = self._extract_topic_from_title(section['title'])
            extracted['vulnerability'] = self._extract_vulnerability(section['content'])
            
        elif content_type == "troubleshooting":
            # Extract error or issue
            error_match = re.search(r'(error|warning|notice):\s*(.+)', section['content'], re.I)
            if error_match:
                extracted['error'] = error_match.group(2)[:100]
            else:
                extracted['issue'] = self._extract_topic_from_title(section['title'])
                
        else:  # howto
            extracted['action'] = self._extract_action_from_title(section['title'])
            
        return extracted if extracted else None
        
    def _extract_action_from_title(self, title: str) -> str:
        """Extract action phrase from title."""
        # Remove common prefixes
        title = re.sub(r'^(How to|Tutorial:|Guide:)\s*', '', title, flags=re.I)
        
        # Convert to lowercase and clean
        action = title.lower().strip()
        
        # Remove WordPress mentions for cleaner prompts
        action = action.replace('wordpress', '').replace('wp', '').strip()
        
        return action or "perform this task"
        
    def _extract_topic_from_title(self, title: str) -> str:
        """Extract main topic from title."""
        # Remove common words
        stop_words = ['the', 'a', 'an', 'in', 'with', 'using', 'for']
        words = title.lower().split()
        topic_words = [w for w in words if w not in stop_words]
        
        return ' '.join(topic_words) or title.lower()
        
    def _extract_vulnerability(self, content: str) -> Optional[str]:
        """Extract vulnerability type from content."""
        vuln_patterns = [
            'sql injection', 'xss', 'cross-site scripting', 'csrf',
            'authentication bypass', 'privilege escalation'
        ]
        
        content_lower = content.lower()
        for vuln in vuln_patterns:
            if vuln in content_lower:
                return vuln
                
        return None
        
    def _fill_template(self, template: str, extracted: Dict) -> Optional[str]:
        """Fill a prompt template with extracted information."""
        try:
            # Use format with defaults
            filled = template
            for key, value in extracted.items():
                filled = filled.replace(f"{{{key}}}", value)
                
            # Check if all placeholders were filled
            if '{' in filled:
                return None
                
            return filled
            
        except Exception:
            return None
            
    def _create_pair(self, prompt: str, response: str, metadata: Dict,
                    front_matter: Dict, content_type: str, 
                    extracted: Dict) -> InstructionPair:
        """Create an instruction pair object."""
        # Generate unique ID
        pair_id = hashlib.md5(f"{prompt}{response}".encode()).hexdigest()[:12]
        
        # Extract tags
        tags = [content_type]
        if 'section' in metadata:
            tags.append(metadata['section'])
        tags.extend(extracted.keys())
        
        # Clean response
        response = self._clean_response(response)
        
        return InstructionPair(
            id=f"wp-{content_type}-{pair_id}",
            source_url=front_matter.get('source_url', metadata.get('url', 'unknown')),
            license="GPL-2.0+",
            prompt=prompt,
            response=response,
            tags=list(set(tags)),
            wp_version=front_matter.get('wp_version', '6.5'),
            meta={
                'content_type': content_type,
                'difficulty': self._estimate_difficulty(response),
                **extracted
            }
        )
        
    def _clean_response(self, response: str) -> str:
        """Clean and format response text."""
        # Remove excessive whitespace
        response = re.sub(r'\n{3,}', '\n\n', response)
        
        # Ensure code blocks are properly formatted
        response = re.sub(r'```(\w*)\n\n', r'```\1\n', response)
        
        # Trim to reasonable length (keep under ~1000 tokens)
        max_chars = 4000
        if len(response) > max_chars:
            # Try to cut at paragraph boundary
            response = response[:max_chars]
            last_para = response.rfind('\n\n')
            if last_para > max_chars * 0.8:
                response = response[:last_para]
                
        return response.strip()
        
    def _estimate_difficulty(self, response: str) -> str:
        """Estimate difficulty level based on response complexity."""
        # Simple heuristics
        code_blocks = len(re.findall(r'```', response))
        word_count = len(response.split())
        
        if code_blocks > 2 or word_count > 500:
            return "advanced"
        elif code_blocks > 0 or word_count > 200:
            return "intermediate"
        else:
            return "beginner"
            
    def _save_pairs(self, pairs: List[InstructionPair]):
        """Save instruction pairs to JSONL files."""
        # Shuffle for variety
        random.shuffle(pairs)
        
        # Write all pairs to single file
        output_file = self.output_dir / "all_pairs.jsonl"
        with output_file.open('w', encoding='utf-8') as f:
            for pair in pairs:
                f.write(json.dumps(asdict(pair), ensure_ascii=False) + '\n')
                
        # Create summary
        summary = {
            'total_pairs': len(pairs),
            'content_types': {},
            'difficulty_levels': {},
            'tags': {}
        }
        
        for pair in pairs:
            ct = pair.meta.get('content_type', 'unknown')
            diff = pair.meta.get('difficulty', 'unknown')
            
            summary['content_types'][ct] = summary['content_types'].get(ct, 0) + 1
            summary['difficulty_levels'][diff] = summary['difficulty_levels'].get(diff, 0) + 1
            
            for tag in pair.tags:
                summary['tags'][tag] = summary['tags'].get(tag, 0) + 1
                
        summary_file = self.output_dir / "generation_summary.json"
        summary_file.write_text(json.dumps(summary, indent=2))
        
        console.print(f"[green]Saved {len(pairs)} pairs to {output_file}[/green]")
        console.print(f"Content types: {summary['content_types']}")
        console.print(f"Difficulty distribution: {summary['difficulty_levels']}")


def main():
    builder = SFTBuilder()
    builder.build_all_pairs()


if __name__ == "__main__":
    main()