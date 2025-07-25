#!/usr/bin/env python3
"""
Split SFT dataset into train/validation/test sets with stratification.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
from rich.console import Console
import hashlib

console = Console()

class DatasetSplitter:
    def __init__(self, input_file: Path = Path("data/sft/all_pairs.jsonl"),
                 output_dir: Path = Path("data/sft")):
        self.input_file = input_file
        self.output_dir = output_dir
        self.random_seed = 42
        random.seed(self.random_seed)
        
    def split_dataset(self, train_ratio: float = 0.9, 
                     val_ratio: float = 0.05, 
                     test_ratio: float = 0.05):
        """Split dataset with stratification by content type and source."""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, \
            "Ratios must sum to 1.0"
            
        console.print("[cyan]Loading dataset...[/cyan]")
        
        # Load all pairs
        pairs = self._load_pairs()
        console.print(f"Loaded {len(pairs)} instruction pairs")
        
        # Group by stratification keys
        grouped = self._group_pairs(pairs)
        
        # Split each group
        train_pairs = []
        val_pairs = []
        test_pairs = []
        
        for group_key, group_pairs in grouped.items():
            # Shuffle group
            random.shuffle(group_pairs)
            
            # Calculate split sizes
            n = len(group_pairs)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            
            # Split
            train_pairs.extend(group_pairs[:n_train])
            val_pairs.extend(group_pairs[n_train:n_train + n_val])
            test_pairs.extend(group_pairs[n_train + n_val:])
            
        # Final shuffle
        random.shuffle(train_pairs)
        random.shuffle(val_pairs)
        random.shuffle(test_pairs)
        
        console.print(f"[green]Split sizes - Train: {len(train_pairs)}, "
                     f"Val: {len(val_pairs)}, Test: {len(test_pairs)}[/green]")
        
        # Ensure no source URL leakage
        self._verify_no_leakage(train_pairs, val_pairs, test_pairs)
        
        # Save splits
        self._save_split(train_pairs, "train.jsonl")
        self._save_split(val_pairs, "val.jsonl")
        self._save_split(test_pairs, "test.jsonl")
        
        # Generate split statistics
        self._generate_split_stats(train_pairs, val_pairs, test_pairs)
        
    def _load_pairs(self) -> List[Dict]:
        """Load all instruction pairs from JSONL."""
        pairs = []
        
        with self.input_file.open('r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    pairs.append(json.loads(line))
                    
        return pairs
        
    def _group_pairs(self, pairs: List[Dict]) -> Dict[str, List[Dict]]:
        """Group pairs by content type and section for stratification."""
        grouped = defaultdict(list)
        
        for pair in pairs:
            # Create stratification key
            content_type = pair.get('meta', {}).get('content_type', 'unknown')
            section = pair.get('tags', ['unknown'])[0] if pair.get('tags') else 'unknown'
            
            key = f"{content_type}_{section}"
            grouped[key].append(pair)
            
        console.print(f"Created {len(grouped)} stratification groups")
        return grouped
        
    def _verify_no_leakage(self, train: List[Dict], val: List[Dict], test: List[Dict]):
        """Verify no source URL appears in multiple splits."""
        # Extract source URLs by split
        train_urls = {self._get_source_hash(p) for p in train}
        val_urls = {self._get_source_hash(p) for p in val}
        test_urls = {self._get_source_hash(p) for p in test}
        
        # Check for overlaps
        train_val_overlap = train_urls & val_urls
        train_test_overlap = train_urls & test_urls
        val_test_overlap = val_urls & test_urls
        
        if train_val_overlap or train_test_overlap or val_test_overlap:
            console.print("[red]Warning: Source URL leakage detected![/red]")
            console.print(f"Train-Val overlap: {len(train_val_overlap)}")
            console.print(f"Train-Test overlap: {len(train_test_overlap)}")
            console.print(f"Val-Test overlap: {len(val_test_overlap)}")
        else:
            console.print("[green]✓ No source URL leakage detected[/green]")
            
    def _get_source_hash(self, pair: Dict) -> str:
        """Get hash of source URL for deduplication."""
        url = pair.get('source_url', 'unknown')
        return hashlib.md5(url.encode()).hexdigest()
        
    def _save_split(self, pairs: List[Dict], filename: str):
        """Save a data split to JSONL file."""
        output_file = self.output_dir / filename
        
        with output_file.open('w', encoding='utf-8') as f:
            for pair in pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + '\n')
                
        console.print(f"Saved {len(pairs)} pairs to {output_file}")
        
    def _generate_split_stats(self, train: List[Dict], val: List[Dict], test: List[Dict]):
        """Generate statistics for data splits."""
        stats = {
            'split_sizes': {
                'train': len(train),
                'val': len(val),
                'test': len(test)
            },
            'random_seed': self.random_seed
        }
        
        # Analyze distribution for each split
        for split_name, split_data in [('train', train), ('val', val), ('test', test)]:
            stats[split_name] = self._analyze_split(split_data)
            
        # Save statistics
        stats_file = self.output_dir / "split_statistics.json"
        stats_file.write_text(json.dumps(stats, indent=2))
        
        # Print summary
        console.print("\n[bold]Split Statistics:[/bold]")
        for split_name in ['train', 'val', 'test']:
            split_stats = stats[split_name]
            console.print(f"\n{split_name.upper()}:")
            console.print(f"  Content types: {split_stats['content_types']}")
            console.print(f"  Avg prompt length: {split_stats['avg_prompt_length']:.1f} chars")
            console.print(f"  Avg response length: {split_stats['avg_response_length']:.1f} chars")
            
    def _analyze_split(self, pairs: List[Dict]) -> Dict:
        """Analyze a data split."""
        analysis = {
            'content_types': defaultdict(int),
            'difficulties': defaultdict(int),
            'prompt_lengths': [],
            'response_lengths': []
        }
        
        for pair in pairs:
            # Content type
            ct = pair.get('meta', {}).get('content_type', 'unknown')
            analysis['content_types'][ct] += 1
            
            # Difficulty
            diff = pair.get('meta', {}).get('difficulty', 'unknown')
            analysis['difficulties'][diff] += 1
            
            # Lengths
            analysis['prompt_lengths'].append(len(pair.get('prompt', '')))
            analysis['response_lengths'].append(len(pair.get('response', '')))
            
        # Calculate averages
        return {
            'content_types': dict(analysis['content_types']),
            'difficulties': dict(analysis['difficulties']),
            'avg_prompt_length': sum(analysis['prompt_lengths']) / len(analysis['prompt_lengths']) if analysis['prompt_lengths'] else 0,
            'avg_response_length': sum(analysis['response_lengths']) / len(analysis['response_lengths']) if analysis['response_lengths'] else 0,
            'total_examples': len(pairs)
        }


def main():
    splitter = DatasetSplitter()
    splitter.split_dataset()


if __name__ == "__main__":
    main()