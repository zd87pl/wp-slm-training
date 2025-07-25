#!/usr/bin/env python3
"""
Generate preference data by scoring candidate responses.
Uses heuristics, external models, or WordPress validation.
"""

import json
import re
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse
import requests
from rich.console import Console
from rich.progress import track
import docker
from dataclasses import dataclass
import time

console = Console()

@dataclass
class PreferenceScore:
    """Score for a candidate response."""
    total_score: float
    code_quality: float
    correctness: float
    completeness: float
    security: float
    explanation: str


class PreferenceLabeler:
    def __init__(self, use_docker_wp: bool = False, 
                 judge_model_api: Optional[str] = None):
        """Initialize preference labeler."""
        self.use_docker_wp = use_docker_wp
        self.judge_model_api = judge_model_api
        
        if use_docker_wp:
            self._init_docker()
            
    def _init_docker(self):
        """Initialize Docker client for WordPress testing."""
        try:
            self.docker_client = docker.from_env()
            console.print("[cyan]Docker client initialized[/cyan]")
        except Exception as e:
            console.print(f"[red]Docker init failed: {e}[/red]")
            self.use_docker_wp = False
            
    def label_candidates(self, candidates_file: str) -> List[Dict]:
        """Label candidate responses to create preference pairs."""
        preferences = []
        
        # Load candidates
        with open(candidates_file, 'r') as f:
            candidate_data = [json.loads(line) for line in f if line.strip()]
            
        for data in track(candidate_data, description="Labeling candidates"):
            prompt = data['prompt']
            candidates = data['candidates']
            
            if len(candidates) < 2:
                continue
                
            # Score each candidate
            scores = []
            for candidate in candidates:
                score = self._score_response(prompt, candidate)
                scores.append(score)
                
            # Create preference pairs
            for i in range(len(candidates)):
                for j in range(i + 1, len(candidates)):
                    if scores[i].total_score > scores[j].total_score:
                        chosen_idx, rejected_idx = i, j
                    else:
                        chosen_idx, rejected_idx = j, i
                        
                    # Only create pair if there's meaningful difference
                    score_diff = abs(scores[chosen_idx].total_score - 
                                   scores[rejected_idx].total_score)
                    
                    if score_diff > 0.1:  # Threshold for meaningful difference
                        preferences.append({
                            'prompt': prompt,
                            'chosen': candidates[chosen_idx],
                            'rejected': candidates[rejected_idx],
                            'score': scores[chosen_idx].total_score,
                            'score_diff': score_diff,
                            'chosen_scores': {
                                'code_quality': scores[chosen_idx].code_quality,
                                'correctness': scores[chosen_idx].correctness,
                                'completeness': scores[chosen_idx].completeness,
                                'security': scores[chosen_idx].security
                            },
                            'rejected_scores': {
                                'code_quality': scores[rejected_idx].code_quality,
                                'correctness': scores[rejected_idx].correctness,
                                'completeness': scores[rejected_idx].completeness,
                                'security': scores[rejected_idx].security
                            }
                        })
                        
        return preferences
        
    def _score_response(self, prompt: str, response: str) -> PreferenceScore:
        """Score a response using multiple criteria."""
        scores = {
            'code_quality': self._score_code_quality(response),
            'correctness': self._score_correctness(prompt, response),
            'completeness': self._score_completeness(prompt, response),
            'security': self._score_security(response)
        }
        
        # Weight the scores
        weights = {
            'code_quality': 0.3,
            'correctness': 0.4,
            'completeness': 0.2,
            'security': 0.1
        }
        
        total_score = sum(scores[k] * weights[k] for k in scores)
        
        explanation = f"Scores: " + ", ".join(
            f"{k}={scores[k]:.2f}" for k in scores
        )
        
        return PreferenceScore(
            total_score=total_score,
            code_quality=scores['code_quality'],
            correctness=scores['correctness'],
            completeness=scores['completeness'],
            security=scores['security'],
            explanation=explanation
        )
        
    def _score_code_quality(self, response: str) -> float:
        """Score code quality based on heuristics."""
        score = 0.5  # Base score
        
        # Extract code blocks
        code_blocks = re.findall(r'```(?:\w+)?\n(.*?)```', response, re.DOTALL)
        
        if not code_blocks:
            # No code when expected
            if any(keyword in response.lower() for keyword in ['function', 'code', 'example']):
                return 0.2
            return 0.5
            
        for code in code_blocks:
            # Check for WordPress coding standards
            if 'function' in code:
                # Function names should be lowercase with underscores
                if re.search(r'function\s+[a-z_]+\s*\(', code):
                    score += 0.1
                    
            # Check for proper escaping
            if 'echo' in code or 'print' in code:
                if any(esc in code for esc in ['esc_html', 'esc_attr', 'esc_url', 'wp_kses']):
                    score += 0.2
                else:
                    score -= 0.1
                    
            # Check for nonce verification in forms
            if 'POST' in code or 'GET' in code:
                if 'nonce' in code or 'verify_nonce' in code:
                    score += 0.1
                    
            # Check for proper hooks
            if 'add_action' in code or 'add_filter' in code:
                # Should have proper priority and args count
                if re.search(r'add_(action|filter)\s*\([^,]+,[^,]+,\s*\d+', code):
                    score += 0.1
                    
        return max(0, min(1, score))
        
    def _score_correctness(self, prompt: str, response: str) -> float:
        """Score correctness of the response."""
        score = 0.6  # Base score
        
        # Check if response addresses the prompt
        prompt_lower = prompt.lower()
        response_lower = response.lower()
        
        # Extract key terms from prompt
        key_terms = []
        if 'rest api' in prompt_lower:
            key_terms.extend(['wp-json', 'rest_', 'endpoint'])
        if 'custom post type' in prompt_lower:
            key_terms.extend(['register_post_type', 'post_type'])
        if 'hook' in prompt_lower or 'filter' in prompt_lower:
            key_terms.extend(['add_action', 'add_filter', 'do_action', 'apply_filters'])
        if 'security' in prompt_lower:
            key_terms.extend(['sanitize', 'escape', 'nonce', 'capability'])
        if 'enqueue' in prompt_lower:
            key_terms.extend(['wp_enqueue_script', 'wp_enqueue_style'])
            
        # Check for presence of key terms
        for term in key_terms:
            if term in response_lower:
                score += 0.1
                
        # Check for WordPress functions
        wp_functions = re.findall(r'\b(wp_\w+|get_\w+|the_\w+|is_\w+|has_\w+)\b', response)
        if wp_functions:
            score += min(0.2, len(set(wp_functions)) * 0.05)
            
        # If using Docker, validate code
        if self.use_docker_wp and code_blocks:
            validation_score = self._validate_with_docker(code_blocks[0])
            score = score * 0.7 + validation_score * 0.3
            
        return max(0, min(1, score))
        
    def _score_completeness(self, prompt: str, response: str) -> float:
        """Score how complete the response is."""
        score = 0.5
        
        # Check structure
        has_explanation = len(response.split('\n\n')) > 1
        has_code = bool(re.findall(r'```', response))
        has_examples = 'example' in response.lower() or 'e.g.' in response.lower()
        
        if has_explanation:
            score += 0.2
        if has_code:
            score += 0.2
        if has_examples:
            score += 0.1
            
        # Check for important warnings or notes
        if any(word in response.lower() for word in ['note:', 'important:', 'warning:']):
            score += 0.1
            
        # Penalize very short responses
        if len(response) < 100:
            score -= 0.3
        elif len(response) > 500:
            score += 0.1
            
        return max(0, min(1, score))
        
    def _score_security(self, response: str) -> float:
        """Score security aspects of the response."""
        score = 0.7  # Base score (assume decent by default)
        
        code_blocks = re.findall(r'```(?:\w+)?\n(.*?)```', response, re.DOTALL)
        
        for code in code_blocks:
            # Check for SQL queries
            if re.search(r'\$wpdb->.*query|mysql_query', code):
                # Should use prepare()
                if '$wpdb->prepare' in code:
                    score += 0.2
                else:
                    score -= 0.3
                    
            # Check for user input handling
            if any(var in code for var in ['$_POST', '$_GET', '$_REQUEST']):
                # Should sanitize
                if any(func in code for func in ['sanitize_', 'wp_verify_nonce', 'esc_']):
                    score += 0.1
                else:
                    score -= 0.2
                    
            # Check for file operations
            if any(func in code for func in ['fopen', 'file_get_contents', 'include', 'require']):
                # Should validate paths
                if 'ABSPATH' in code or 'plugin_dir_path' in code:
                    score += 0.1
                else:
                    score -= 0.1
                    
        # Check for security mentions in explanation
        if any(term in response.lower() for term in ['sanitize', 'escape', 'validate', 'nonce']):
            score += 0.1
            
        return max(0, min(1, score))
        
    def _validate_with_docker(self, code: str) -> float:
        """Validate code using Docker WordPress instance."""
        try:
            # This is a simplified version - in production you'd want more robust testing
            # For now, just check if it's valid PHP
            result = subprocess.run(
                ['php', '-l'],
                input=f"<?php\n{code}\n?>",
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                return 0.8
            else:
                return 0.2
                
        except Exception as e:
            console.print(f"[yellow]Docker validation failed: {e}[/yellow]")
            return 0.5
            
    def label_with_judge_model(self, prompt: str, response1: str, 
                              response2: str) -> Tuple[int, float]:
        """Use an external judge model to compare responses."""
        if not self.judge_model_api:
            return 0, 0.5  # No preference
            
        judge_prompt = f"""Compare these two WordPress assistance responses and determine which is better.

Question: {prompt}

Response A:
{response1}

Response B:
{response2}

Consider: correctness, completeness, code quality, security practices, and clarity.
Which response is better? Respond with only "A" or "B" and a confidence score (0-1)."""
        
        try:
            # Make API call to judge model
            response = requests.post(
                self.judge_model_api,
                json={
                    'prompt': judge_prompt,
                    'max_tokens': 10,
                    'temperature': 0
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                text = result.get('text', '').strip()
                
                if 'A' in text:
                    return 1, 0.8
                elif 'B' in text:
                    return 2, 0.8
                    
        except Exception as e:
            console.print(f"[yellow]Judge model error: {e}[/yellow]")
            
        return 0, 0.5


def main():
    parser = argparse.ArgumentParser(description="Generate preference labels for candidates")
    parser.add_argument("--candidates", type=str, default="data/prefs/candidates.jsonl",
                        help="Path to candidates file")
    parser.add_argument("--output", type=str, default="data/prefs/preferences.jsonl",
                        help="Output path for preferences")
    parser.add_argument("--use-docker", action="store_true",
                        help="Use Docker WordPress for validation")
    parser.add_argument("--judge-api", type=str,
                        help="API endpoint for judge model")
    parser.add_argument("--min-score-diff", type=float, default=0.1,
                        help="Minimum score difference for preference")
    
    args = parser.parse_args()
    
    # Initialize labeler
    labeler = PreferenceLabeler(
        use_docker_wp=args.use_docker,
        judge_model_api=args.judge_api
    )
    
    # Generate preferences
    preferences = labeler.label_candidates(args.candidates)
    
    # Save preferences
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open('w') as f:
        for pref in preferences:
            f.write(json.dumps(pref) + '\n')
            
    console.print(f"[green]Generated {len(preferences)} preference pairs[/green]")
    
    # Print statistics
    if preferences:
        avg_score_diff = sum(p['score_diff'] for p in preferences) / len(preferences)
        console.print(f"[cyan]Average score difference: {avg_score_diff:.3f}[/cyan]")
        
        # Score distribution
        score_ranges = {'0.0-0.2': 0, '0.2-0.4': 0, '0.4-0.6': 0, '0.6-0.8': 0, '0.8-1.0': 0}
        for pref in preferences:
            score = pref['score']
            if score <= 0.2:
                score_ranges['0.0-0.2'] += 1
            elif score <= 0.4:
                score_ranges['0.2-0.4'] += 1
            elif score <= 0.6:
                score_ranges['0.4-0.6'] += 1
            elif score <= 0.8:
                score_ranges['0.6-0.8'] += 1
            else:
                score_ranges['0.8-1.0'] += 1
                
        console.print("[cyan]Score distribution:[/cyan]")
        for range_name, count in score_ranges.items():
            console.print(f"  {range_name}: {count}")


if __name__ == "__main__":
    main()