#!/usr/bin/env python3
"""
Evaluation script for WordPress SLM.
Tests model performance on various WordPress tasks.
"""

import json
import argparse
import subprocess
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import PeftModel
from rich.console import Console
from rich.table import Table
from rich.progress import track
import requests
from dataclasses import dataclass
import docker
import tempfile

console = Console()

@dataclass
class EvalResult:
    """Individual evaluation result."""
    prompt: str
    expected: Optional[str]
    generated: str
    metrics: Dict[str, float]
    passed: bool
    error: Optional[str] = None


class WPSLMEvaluator:
    def __init__(self, model_path: str, use_docker: bool = False):
        """Initialize evaluator with model."""
        self.model_path = model_path
        self.use_docker = use_docker
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        console.print(f"[cyan]Loading model from {model_path}...[/cyan]")
        self._load_model()
        
        if use_docker:
            self._init_docker()
            
    def _load_model(self):
        """Load model and tokenizer."""
        # Check if PEFT model
        peft_config_path = Path(self.model_path) / "adapter_config.json"
        is_peft = peft_config_path.exists()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        if is_peft:
            # Load PEFT model
            with open(peft_config_path, 'r') as f:
                peft_config = json.load(f)
            base_model_name = peft_config['base_model_name_or_path']
            
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=bnb_config,
                device_map="auto"
            )
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
        else:
            # Load full model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=bnb_config,
                device_map="auto"
            )
            
        self.model.eval()
        console.print("[green]Model loaded successfully[/green]")
        
    def _init_docker(self):
        """Initialize Docker client for WordPress testing."""
        try:
            self.docker_client = docker.from_env()
            # Check if WordPress container is running
            containers = self.docker_client.containers.list(
                filters={"name": "wp-slm-wordpress"}
            )
            if not containers:
                console.print("[yellow]WordPress container not found. Run 'make docker-wp' first.[/yellow]")
                self.use_docker = False
            else:
                self.wp_container = containers[0]
                console.print("[green]Connected to WordPress container[/green]")
        except Exception as e:
            console.print(f"[red]Docker init failed: {e}[/red]")
            self.use_docker = False
            
    def evaluate(self, test_file: str) -> Dict[str, float]:
        """Run evaluation on test set."""
        # Load test data
        test_data = self._load_test_data(test_file)
        console.print(f"[cyan]Loaded {len(test_data)} test examples[/cyan]")
        
        # Run evaluations
        results = []
        for item in track(test_data, description="Evaluating"):
            result = self._evaluate_single(item)
            results.append(result)
            
        # Compute metrics
        metrics = self._compute_metrics(results)
        
        # Display results
        self._display_results(results, metrics)
        
        return metrics
        
    def _load_test_data(self, test_file: str) -> List[Dict]:
        """Load test dataset."""
        test_data = []
        with open(test_file, 'r') as f:
            for line in f:
                if line.strip():
                    test_data.append(json.loads(line))
        return test_data
        
    def _evaluate_single(self, test_item: Dict) -> EvalResult:
        """Evaluate a single test item."""
        prompt = test_item['prompt']
        expected = test_item.get('response', test_item.get('expected'))
        
        # Generate response
        generated = self._generate_response(prompt)
        
        # Evaluate based on task type
        task_type = self._detect_task_type(prompt, test_item)
        metrics = {}
        passed = True
        error = None
        
        try:
            if task_type == "code_generation":
                metrics, passed = self._eval_code_generation(prompt, generated, expected)
            elif task_type == "api_endpoint":
                metrics, passed = self._eval_api_endpoint(prompt, generated)
            elif task_type == "security":
                metrics, passed = self._eval_security(prompt, generated)
            else:
                metrics, passed = self._eval_general(prompt, generated, expected)
        except Exception as e:
            error = str(e)
            passed = False
            
        return EvalResult(
            prompt=prompt,
            expected=expected,
            generated=generated,
            metrics=metrics,
            passed=passed,
            error=error
        )
        
    def _generate_response(self, prompt: str, max_new_tokens: int = 512) -> str:
        """Generate response from model."""
        # Format prompt
        formatted_prompt = f"""You are a helpful WordPress expert assistant. Answer the following question accurately and provide code examples when relevant.

USER: {prompt}
ASSISTANT: """
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,  # Low temperature for consistent eval
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
        # Decode
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant response
        if "ASSISTANT:" in generated:
            response = generated.split("ASSISTANT:")[-1].strip()
        else:
            response = generated[len(formatted_prompt):].strip()
            
        return response
        
    def _detect_task_type(self, prompt: str, test_item: Dict) -> str:
        """Detect the type of evaluation task."""
        # Check metadata first
        if 'task_type' in test_item:
            return test_item['task_type']
            
        # Heuristics
        prompt_lower = prompt.lower()
        if 'code' in prompt_lower or 'function' in prompt_lower or 'snippet' in prompt_lower:
            return "code_generation"
        elif 'rest api' in prompt_lower or 'endpoint' in prompt_lower:
            return "api_endpoint"
        elif 'security' in prompt_lower or 'sanitize' in prompt_lower or 'escape' in prompt_lower:
            return "security"
        else:
            return "general"
            
    def _eval_code_generation(self, prompt: str, generated: str, 
                             expected: Optional[str]) -> Tuple[Dict, bool]:
        """Evaluate code generation task."""
        metrics = {
            "has_code": 0.0,
            "syntax_valid": 0.0,
            "wp_functions": 0.0,
            "similarity": 0.0
        }
        
        # Extract code blocks
        code_blocks = re.findall(r'```(?:\w+)?\n(.*?)```', generated, re.DOTALL)
        
        if code_blocks:
            metrics["has_code"] = 1.0
            
            # Check PHP syntax
            for code in code_blocks:
                if self._check_php_syntax(code):
                    metrics["syntax_valid"] = 1.0
                    break
                    
            # Check for WordPress functions
            wp_functions = re.findall(
                r'\b(wp_\w+|get_\w+|add_action|add_filter|do_action|apply_filters)\b',
                generated
            )
            if wp_functions:
                metrics["wp_functions"] = min(1.0, len(set(wp_functions)) / 3)
                
        # Compare with expected if available
        if expected:
            from difflib import SequenceMatcher
            similarity = SequenceMatcher(None, generated, expected).ratio()
            metrics["similarity"] = similarity
            
        # Overall pass criteria
        passed = (
            metrics["has_code"] > 0.5 and
            metrics["syntax_valid"] > 0.5 and
            metrics["wp_functions"] > 0.3
        )
        
        return metrics, passed
        
    def _eval_api_endpoint(self, prompt: str, generated: str) -> Tuple[Dict, bool]:
        """Evaluate REST API endpoint task."""
        metrics = {
            "has_endpoint": 0.0,
            "valid_method": 0.0,
            "has_auth": 0.0,
            "testable": 0.0
        }
        
        # Check for endpoint
        endpoint_pattern = r'(wp-json|/wp/v2|rest_route)'
        if re.search(endpoint_pattern, generated):
            metrics["has_endpoint"] = 1.0
            
        # Check for HTTP method
        method_pattern = r'\b(GET|POST|PUT|DELETE|PATCH)\b'
        if re.search(method_pattern, generated, re.I):
            metrics["valid_method"] = 1.0
            
        # Check for authentication mention
        auth_keywords = ['nonce', 'authentication', 'cookie', 'application password', 'jwt']
        if any(keyword in generated.lower() for keyword in auth_keywords):
            metrics["has_auth"] = 1.0
            
        # Try to extract and test endpoint if Docker available
        if self.use_docker and metrics["has_endpoint"] > 0:
            metrics["testable"] = self._test_endpoint(generated)
            
        passed = metrics["has_endpoint"] > 0.5 and metrics["valid_method"] > 0.5
        
        return metrics, passed
        
    def _eval_security(self, prompt: str, generated: str) -> Tuple[Dict, bool]:
        """Evaluate security-related response."""
        metrics = {
            "mentions_sanitization": 0.0,
            "mentions_escaping": 0.0,
            "mentions_nonce": 0.0,
            "mentions_capabilities": 0.0,
            "has_secure_code": 0.0
        }
        
        response_lower = generated.lower()
        
        # Check security concepts
        if any(word in response_lower for word in ['sanitize', 'sanitization']):
            metrics["mentions_sanitization"] = 1.0
            
        if any(word in response_lower for word in ['escape', 'esc_html', 'esc_attr', 'esc_url']):
            metrics["mentions_escaping"] = 1.0
            
        if 'nonce' in response_lower:
            metrics["mentions_nonce"] = 1.0
            
        if any(word in response_lower for word in ['capability', 'current_user_can', 'permission']):
            metrics["mentions_capabilities"] = 1.0
            
        # Check for secure code patterns
        code_blocks = re.findall(r'```(?:\w+)?\n(.*?)```', generated, re.DOTALL)
        for code in code_blocks:
            if self._check_secure_code(code):
                metrics["has_secure_code"] = 1.0
                break
                
        # Pass if mentions at least 2 security concepts
        security_score = sum(metrics.values()) / len(metrics)
        passed = security_score >= 0.4
        
        return metrics, passed
        
    def _eval_general(self, prompt: str, generated: str, 
                     expected: Optional[str]) -> Tuple[Dict, bool]:
        """Evaluate general response."""
        metrics = {
            "length_appropriate": 0.0,
            "mentions_wordpress": 0.0,
            "coherent": 0.0,
            "similarity": 0.0
        }
        
        # Check length
        word_count = len(generated.split())
        if 50 <= word_count <= 500:
            metrics["length_appropriate"] = 1.0
        elif 20 <= word_count < 50:
            metrics["length_appropriate"] = 0.5
            
        # Check WordPress relevance
        if 'wordpress' in generated.lower() or 'wp' in re.findall(r'\bwp\b', generated.lower()):
            metrics["mentions_wordpress"] = 1.0
            
        # Basic coherence check (has sentences)
        if len(generated.split('.')) > 1:
            metrics["coherent"] = 1.0
            
        # Similarity if expected provided
        if expected:
            from difflib import SequenceMatcher
            similarity = SequenceMatcher(None, generated.lower(), expected.lower()).ratio()
            metrics["similarity"] = similarity
            
        passed = (
            metrics["length_appropriate"] > 0.3 and
            metrics["coherent"] > 0.5
        )
        
        return metrics, passed
        
    def _check_php_syntax(self, code: str) -> bool:
        """Check if PHP code has valid syntax."""
        try:
            # Wrap in PHP tags if not present
            if not code.strip().startswith('<?php'):
                code = f"<?php\n{code}\n?>"
                
            # Use PHP lint
            result = subprocess.run(
                ['php', '-l'],
                input=code,
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception:
            # If PHP not available, do basic checks
            return not any(error in code for error in ['Parse error', 'Fatal error'])
            
    def _check_secure_code(self, code: str) -> bool:
        """Check if code follows security best practices."""
        security_patterns = [
            r'sanitize_\w+',
            r'esc_\w+',
            r'wp_verify_nonce',
            r'current_user_can',
            r'wp_kses',
            r'absint',
            r'intval'
        ]
        
        # Check if uses security functions when handling user input
        if any(var in code for var in ['$_POST', '$_GET', '$_REQUEST']):
            return any(re.search(pattern, code) for pattern in security_patterns)
            
        # If no user input, consider it secure by default
        return True
        
    def _test_endpoint(self, response: str) -> float:
        """Test REST API endpoint if possible."""
        try:
            # Extract URL pattern
            url_match = re.search(r'(https?://[^\s]+/wp-json/[^\s]+)', response)
            if not url_match:
                return 0.0
                
            # Replace with local WordPress URL
            endpoint = url_match.group(1)
            endpoint = endpoint.replace('example.com', 'localhost:8080')
            endpoint = endpoint.replace('yoursite.com', 'localhost:8080')
            
            # Make request
            response = requests.get(endpoint, timeout=5)
            
            # Consider it working if not 404
            if response.status_code != 404:
                return 1.0
            else:
                return 0.0
                
        except Exception:
            return 0.0
            
    def _compute_metrics(self, results: List[EvalResult]) -> Dict[str, float]:
        """Compute overall metrics from results."""
        if not results:
            return {}
            
        # Aggregate metrics
        all_metrics = {}
        for result in results:
            for key, value in result.metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(value)
                
        # Compute averages
        avg_metrics = {
            key: sum(values) / len(values)
            for key, values in all_metrics.items()
        }
        
        # Add overall metrics
        avg_metrics['accuracy'] = sum(1 for r in results if r.passed) / len(results)
        avg_metrics['total_evaluated'] = len(results)
        
        return avg_metrics
        
    def _display_results(self, results: List[EvalResult], metrics: Dict[str, float]):
        """Display evaluation results."""
        # Create summary table
        table = Table(title="Evaluation Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Score", style="green")
        
        for key, value in metrics.items():
            if key == 'total_evaluated':
                table.add_row(key, str(int(value)))
            else:
                table.add_row(key, f"{value:.3f}")
                
        console.print(table)
        
        # Show failed examples
        failed = [r for r in results if not r.passed]
        if failed:
            console.print(f"\n[red]Failed {len(failed)} examples:[/red]")
            for i, result in enumerate(failed[:5]):  # Show first 5
                console.print(f"\n[yellow]Failed {i+1}:[/yellow]")
                console.print(f"Prompt: {result.prompt[:100]}...")
                if result.error:
                    console.print(f"Error: {result.error}")
                console.print(f"Metrics: {result.metrics}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate WordPress SLM")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to model to evaluate")
    parser.add_argument("--test-file", type=str, default="data/eval/test.jsonl",
                        help="Path to test file")
    parser.add_argument("--use-docker", action="store_true",
                        help="Use Docker WordPress for testing")
    parser.add_argument("--output", type=str, default="eval_results.json",
                        help="Output file for results")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = WPSLMEvaluator(args.model, use_docker=args.use_docker)
    
    # Run evaluation
    metrics = evaluator.evaluate(args.test_file)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(metrics, f, indent=2)
        
    console.print(f"\n[green]Results saved to {args.output}[/green]")


if __name__ == "__main__":
    main()