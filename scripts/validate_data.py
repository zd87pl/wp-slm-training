#!/usr/bin/env python3
"""
Validate data quality for WordPress SLM training.
Checks for data integrity, format correctness, and quality metrics.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import argparse
from rich.console import Console
from rich.table import Table
from rich.progress import track

console = Console()

class DataValidator:
    def __init__(self, data_dir: Path = Path("data")):
        self.data_dir = data_dir
        self.errors = []
        self.warnings = []
        self.stats = defaultdict(int)
        
    def validate_all(self) -> bool:
        """Run all validation checks."""
        console.print("[bold cyan]WordPress SLM Data Validation[/bold cyan]\n")
        
        all_valid = True
        
        # Check directory structure
        if not self._validate_directory_structure():
            all_valid = False
            
        # Validate raw data
        if self.data_dir.joinpath("raw").exists():
            console.print("[yellow]Validating raw data...[/yellow]")
            self._validate_raw_data()
            
        # Validate processed data
        if self.data_dir.joinpath("processed").exists():
            console.print("[yellow]Validating processed data...[/yellow]")
            self._validate_processed_data()
            
        # Validate SFT data
        if self.data_dir.joinpath("sft").exists():
            console.print("[yellow]Validating SFT datasets...[/yellow]")
            if not self._validate_sft_data():
                all_valid = False
                
        # Validate preference data
        if self.data_dir.joinpath("prefs").exists():
            console.print("[yellow]Validating preference data...[/yellow]")
            if not self._validate_preference_data():
                all_valid = False
                
        # Display results
        self._display_results()
        
        return all_valid and len(self.errors) == 0
        
    def _validate_directory_structure(self) -> bool:
        """Check if required directories exist."""
        required_dirs = ["raw", "processed", "sft", "prefs", "eval"]
        missing = []
        
        for dir_name in required_dirs:
            if not self.data_dir.joinpath(dir_name).exists():
                missing.append(dir_name)
                
        if missing:
            self.warnings.append(f"Missing directories: {', '.join(missing)}")
            
        return True
        
    def _validate_raw_data(self) -> bool:
        """Validate raw scraped data."""
        raw_dir = self.data_dir / "raw"
        html_files = list(raw_dir.glob("*.html"))
        
        self.stats['raw_html_files'] = len(html_files)
        
        # Check for metadata files
        for html_file in html_files[:10]:  # Sample check
            meta_file = html_file.with_suffix('.html.meta.json')
            if not meta_file.exists():
                self.warnings.append(f"Missing metadata for {html_file.name}")
            else:
                # Validate metadata format
                try:
                    with meta_file.open() as f:
                        meta = json.load(f)
                    required_fields = ['url', 'section', 'scraped_at']
                    for field in required_fields:
                        if field not in meta:
                            self.errors.append(f"Missing field '{field}' in {meta_file.name}")
                except json.JSONDecodeError:
                    self.errors.append(f"Invalid JSON in {meta_file.name}")
                    
        return True
        
    def _validate_processed_data(self) -> bool:
        """Validate processed markdown files."""
        processed_dir = self.data_dir / "processed"
        md_files = list(processed_dir.glob("*.md"))
        
        self.stats['processed_files'] = len(md_files)
        
        for md_file in track(md_files[:20], description="Validating processed files"):
            try:
                content = md_file.read_text(encoding='utf-8')
                
                # Check for front matter
                if not content.startswith('---'):
                    self.warnings.append(f"No front matter in {md_file.name}")
                    
                # Check minimum length
                if len(content) < 100:
                    self.warnings.append(f"Very short content in {md_file.name}")
                    
                # Check for code blocks
                code_blocks = re.findall(r'```', content)
                if len(code_blocks) % 2 != 0:
                    self.errors.append(f"Unclosed code block in {md_file.name}")
                    
            except Exception as e:
                self.errors.append(f"Error reading {md_file.name}: {e}")
                
        return True
        
    def _validate_sft_data(self) -> bool:
        """Validate SFT training data."""
        sft_dir = self.data_dir / "sft"
        valid = True
        
        # Check for required files
        required_files = ["train.jsonl", "val.jsonl", "test.jsonl"]
        for filename in required_files:
            filepath = sft_dir / filename
            if not filepath.exists():
                self.errors.append(f"Missing required SFT file: {filename}")
                valid = False
                continue
                
            # Validate JSONL format
            valid_lines = 0
            invalid_lines = 0
            seen_ids = set()
            
            with filepath.open('r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if not line.strip():
                        continue
                        
                    try:
                        data = json.loads(line)
                        
                        # Check required fields
                        required = ['prompt', 'response']
                        for field in required:
                            if field not in data:
                                self.errors.append(f"{filename}:{i+1} missing field '{field}'")
                                invalid_lines += 1
                                continue
                                
                        # Check for empty values
                        if not data['prompt'].strip():
                            self.errors.append(f"{filename}:{i+1} empty prompt")
                            invalid_lines += 1
                            continue
                            
                        if not data['response'].strip():
                            self.errors.append(f"{filename}:{i+1} empty response")
                            invalid_lines += 1
                            continue
                            
                        # Check for duplicates
                        if 'id' in data:
                            if data['id'] in seen_ids:
                                self.warnings.append(f"{filename}:{i+1} duplicate ID: {data['id']}")
                            seen_ids.add(data['id'])
                            
                        # Check prompt quality
                        if len(data['prompt']) < 10:
                            self.warnings.append(f"{filename}:{i+1} very short prompt")
                            
                        # Check response quality
                        if len(data['response']) < 50:
                            self.warnings.append(f"{filename}:{i+1} very short response")
                            
                        valid_lines += 1
                        
                    except json.JSONDecodeError as e:
                        self.errors.append(f"{filename}:{i+1} invalid JSON: {e}")
                        invalid_lines += 1
                        
            self.stats[f'{filename}_valid_lines'] = valid_lines
            self.stats[f'{filename}_invalid_lines'] = invalid_lines
            
            if invalid_lines > 0:
                valid = False
                
        # Check for data leakage between splits
        if all((sft_dir / f).exists() for f in required_files):
            self._check_data_leakage(sft_dir)
            
        return valid
        
    def _validate_preference_data(self) -> bool:
        """Validate preference data."""
        prefs_dir = self.data_dir / "prefs"
        valid = True
        
        # Check preference files
        pref_files = list(prefs_dir.glob("*.jsonl"))
        
        for filepath in pref_files:
            with filepath.open('r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if not line.strip():
                        continue
                        
                    try:
                        data = json.loads(line)
                        
                        # Check required fields
                        required = ['prompt', 'chosen', 'rejected']
                        for field in required:
                            if field not in data:
                                self.errors.append(f"{filepath.name}:{i+1} missing field '{field}'")
                                valid = False
                                continue
                                
                        # Check that chosen != rejected
                        if data.get('chosen') == data.get('rejected'):
                            self.errors.append(f"{filepath.name}:{i+1} chosen equals rejected")
                            valid = False
                            
                        # Check for score if present
                        if 'score' in data:
                            if not isinstance(data['score'], (int, float)):
                                self.errors.append(f"{filepath.name}:{i+1} invalid score type")
                            elif not 0 <= data['score'] <= 1:
                                self.warnings.append(f"{filepath.name}:{i+1} score out of [0,1] range")
                                
                    except json.JSONDecodeError as e:
                        self.errors.append(f"{filepath.name}:{i+1} invalid JSON: {e}")
                        valid = False
                        
        return valid
        
    def _check_data_leakage(self, sft_dir: Path):
        """Check for data leakage between train/val/test splits."""
        splits = {}
        
        for split in ['train', 'val', 'test']:
            filepath = sft_dir / f"{split}.jsonl"
            prompts = set()
            
            with filepath.open('r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        prompts.add(data['prompt'])
                        
            splits[split] = prompts
            
        # Check for overlaps
        train_val_overlap = len(splits['train'] & splits['val'])
        train_test_overlap = len(splits['train'] & splits['test'])
        val_test_overlap = len(splits['val'] & splits['test'])
        
        if train_val_overlap > 0:
            self.warnings.append(f"Train/Val overlap: {train_val_overlap} prompts")
        if train_test_overlap > 0:
            self.warnings.append(f"Train/Test overlap: {train_test_overlap} prompts")
        if val_test_overlap > 0:
            self.warnings.append(f"Val/Test overlap: {val_test_overlap} prompts")
            
    def _display_results(self):
        """Display validation results."""
        console.print("\n[bold]Validation Results[/bold]")
        
        # Statistics table
        if self.stats:
            table = Table(title="Data Statistics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            for key, value in sorted(self.stats.items()):
                table.add_row(key, str(value))
                
            console.print(table)
            
        # Errors
        if self.errors:
            console.print(f"\n[red]Errors ({len(self.errors)}):[/red]")
            for error in self.errors[:10]:  # Show first 10
                console.print(f"  • {error}")
            if len(self.errors) > 10:
                console.print(f"  ... and {len(self.errors) - 10} more")
                
        # Warnings
        if self.warnings:
            console.print(f"\n[yellow]Warnings ({len(self.warnings)}):[/yellow]")
            for warning in self.warnings[:10]:  # Show first 10
                console.print(f"  • {warning}")
            if len(self.warnings) > 10:
                console.print(f"  ... and {len(self.warnings) - 10} more")
                
        # Summary
        if not self.errors:
            console.print("\n[green]✓ All validations passed![/green]")
        else:
            console.print(f"\n[red]✗ Validation failed with {len(self.errors)} errors[/red]")


def main():
    parser = argparse.ArgumentParser(description="Validate WordPress SLM training data")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Data directory to validate")
    parser.add_argument("--fix", action="store_true",
                        help="Attempt to fix common issues")
    
    args = parser.parse_args()
    
    validator = DataValidator(Path(args.data_dir))
    
    if validator.validate_all():
        exit(0)
    else:
        exit(1)


if __name__ == "__main__":
    main()