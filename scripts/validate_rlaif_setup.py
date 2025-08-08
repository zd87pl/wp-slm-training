#!/usr/bin/env python3

"""
WordPress SLM RLAIF Setup Validation Script
Validates that all required components are properly configured.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_status(message: str, status: str, details: str = ""):
    """Print formatted status message."""
    if status == "PASS":
        color = Colors.GREEN
        symbol = "✓"
    elif status == "WARN":
        color = Colors.YELLOW
        symbol = "⚠"
    elif status == "FAIL":
        color = Colors.RED
        symbol = "✗"
    else:
        color = Colors.BLUE
        symbol = "ℹ"
    
    print(f"{color}{symbol} {message}{Colors.END}")
    if details:
        print(f"   {details}")

def check_python_packages() -> List[Tuple[str, str, str]]:
    """Check required Python packages."""
    results = []
    required_packages = [
        "torch", "transformers", "datasets", "peft", 
        "accelerate", "openai", "aiohttp", "tqdm",
        "numpy", "pandas", "scikit-learn"
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            results.append((f"Python package '{package}'", "PASS", ""))
        except ImportError:
            results.append((f"Python package '{package}'", "FAIL", f"Install with: pip install {package}"))
    
    return results

def check_file_structure() -> List[Tuple[str, str, str]]:
    """Check required file structure."""
    results = []
    
    # Check for RLAIF pipeline files
    required_files = [
        ("training/ai_judge.py", "AI Judge system"),
        ("scripts/generate_reward_dataset.py", "Dataset generation script"),
        ("training/reward_model.py", "Reward model implementation"),
        ("training/RLAIF_USAGE_GUIDE.md", "Usage guide"),
        ("scripts/run_rlaif_pipeline.sh", "Pipeline runner script"),
    ]
    
    for file_path, description in required_files:
        if Path(file_path).exists():
            results.append((f"{description}", "PASS", f"Found: {file_path}"))
        else:
            results.append((f"{description}", "FAIL", f"Missing: {file_path}"))
    
    # Check for model directory
    model_paths = [
        "./models/wp-slm-rtx5090",
        "./models/wp-slm",
        "./wp-slm-model"
    ]
    
    model_found = False
    for model_path in model_paths:
        if Path(model_path).exists():
            # Check for PEFT files
            adapter_config = Path(model_path) / "adapter_config.json"
            adapter_model = Path(model_path) / "adapter_model.safetensors"
            
            if adapter_config.exists() and adapter_model.exists():
                results.append(("SFT model with PEFT adapters", "PASS", f"Found: {model_path}"))
                model_found = True
                break
            else:
                results.append(("SFT model", "WARN", f"Found directory but missing PEFT files: {model_path}"))
    
    if not model_found:
        results.append(("SFT model", "FAIL", "No valid SFT model found. Expected PEFT adapters."))
    
    return results

def check_environment() -> List[Tuple[str, str, str]]:
    """Check environment configuration."""
    results = []
    
    # Check OpenAI API key
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        if len(openai_key) > 20:  # Basic validation
            results.append(("OpenAI API Key", "PASS", "Environment variable set"))
        else:
            results.append(("OpenAI API Key", "WARN", "Key seems too short, verify it's correct"))
    else:
        results.append(("OpenAI API Key", "FAIL", "Set with: export OPENAI_API_KEY='your-key'"))
    
    # Check CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            memory = torch.cuda.get_device_properties(0).total_memory / 1024**3 if gpu_count > 0 else 0
            results.append(("CUDA GPU", "PASS", f"{gpu_count} GPU(s), {gpu_name}, {memory:.1f}GB"))
        else:
            results.append(("CUDA GPU", "WARN", "No CUDA GPU available, training will be slow"))
    except:
        results.append(("CUDA GPU", "FAIL", "Cannot check CUDA availability"))
    
    return results

def check_directories() -> List[Tuple[str, str, str]]:
    """Check and create required directories."""
    results = []
    required_dirs = [
        "datasets",
        "models", 
        "logs",
        "training",
        "scripts"
    ]
    
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            results.append((f"Directory '{dir_name}'", "PASS", ""))
        else:
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                results.append((f"Directory '{dir_name}'", "PASS", "Created"))
            except Exception as e:
                results.append((f"Directory '{dir_name}'", "FAIL", f"Cannot create: {e}"))
    
    return results

def estimate_system_requirements() -> List[Tuple[str, str, str]]:
    """Estimate system requirements for RLAIF pipeline."""
    results = []
    
    # Disk space check
    try:
        import shutil
        total, used, free = shutil.disk_usage(".")
        free_gb = free / (1024**3)
        
        if free_gb >= 10:
            results.append(("Disk space", "PASS", f"{free_gb:.1f}GB available"))
        else:
            results.append(("Disk space", "WARN", f"Only {free_gb:.1f}GB free, recommend 10GB+"))
    except:
        results.append(("Disk space", "INFO", "Could not check disk space"))
    
    # Memory estimation
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        if memory_gb >= 16:
            results.append(("System RAM", "PASS", f"{memory_gb:.1f}GB available"))
        else:
            results.append(("System RAM", "WARN", f"Only {memory_gb:.1f}GB RAM, recommend 16GB+"))
    except:
        results.append(("System RAM", "INFO", "Could not check system memory"))
    
    return results

def generate_sample_commands() -> None:
    """Generate sample commands for running the pipeline."""
    print(f"\n{Colors.BOLD}Sample Commands:{Colors.END}")
    print(f"{Colors.BLUE}# Quick start (1000 samples):{Colors.END}")
    print("./scripts/run_rlaif_pipeline.sh")
    print()
    print(f"{Colors.BLUE}# Large dataset (5000 samples):{Colors.END}")
    print("./scripts/run_rlaif_pipeline.sh --samples 5000 --batch-size 100")
    print()
    print(f"{Colors.BLUE}# Custom model path:{Colors.END}")
    print("./scripts/run_rlaif_pipeline.sh --model ./path/to/your/model")
    print()
    print(f"{Colors.BLUE}# Individual steps:{Colors.END}")
    print("python scripts/generate_reward_dataset.py --help")
    print("python training/reward_model.py --help")

def main():
    """Main validation function."""
    print(f"{Colors.BOLD}WordPress SLM RLAIF Setup Validation{Colors.END}")
    print("=" * 50)
    print()
    
    all_checks = [
        ("Python Packages", check_python_packages),
        ("File Structure", check_file_structure),
        ("Environment Configuration", check_environment),
        ("Directory Structure", check_directories),
        ("System Requirements", estimate_system_requirements),
    ]
    
    total_passed = 0
    total_failed = 0
    total_warned = 0
    
    for section_name, check_function in all_checks:
        print(f"{Colors.BOLD}{section_name}:{Colors.END}")
        results = check_function()
        
        for message, status, details in results:
            print_status(message, status, details)
            if status == "PASS":
                total_passed += 1
            elif status == "FAIL":
                total_failed += 1
            elif status == "WARN":
                total_warned += 1
        
        print()
    
    # Summary
    print(f"{Colors.BOLD}Summary:{Colors.END}")
    print(f"  {Colors.GREEN}Passed: {total_passed}{Colors.END}")
    if total_warned > 0:
        print(f"  {Colors.YELLOW}Warnings: {total_warned}{Colors.END}")
    if total_failed > 0:
        print(f"  {Colors.RED}Failed: {total_failed}{Colors.END}")
    
    if total_failed == 0:
        print(f"\n{Colors.GREEN}✓ RLAIF pipeline is ready to run!{Colors.END}")
        generate_sample_commands()
    else:
        print(f"\n{Colors.RED}✗ Please fix the failed checks before running the pipeline.{Colors.END}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())