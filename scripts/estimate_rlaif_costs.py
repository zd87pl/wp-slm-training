#!/usr/bin/env python3

"""
WordPress SLM RLAIF Cost Estimation Script
Estimates costs and resource requirements for RLAIF pipeline.
"""

import argparse
import json
from typing import Dict, Tuple

def estimate_api_costs(samples: int, model: str = "gpt-4") -> Dict[str, float]:
    """Estimate OpenAI API costs for dataset generation."""
    
    # Pricing per 1K tokens (approximate, check current pricing)
    pricing = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-3.5-turbo": {"input": 0.001, "output": 0.002}
    }
    
    if model not in pricing:
        model = "gpt-4"  # Default to GPT-4
    
    # Estimate tokens per request
    # Input: prompt + response (~2000 tokens average)
    # Output: evaluation + scores (~1000 tokens average)
    input_tokens_per_sample = 2000
    output_tokens_per_sample = 1000
    
    total_input_tokens = samples * input_tokens_per_sample
    total_output_tokens = samples * output_tokens_per_sample
    
    input_cost = (total_input_tokens / 1000) * pricing[model]["input"]
    output_cost = (total_output_tokens / 1000) * pricing[model]["output"]
    total_cost = input_cost + output_cost
    
    return {
        "total_samples": samples,
        "model_used": model,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost
    }

def estimate_training_costs(samples: int, gpu_type: str = "rtx5090") -> Dict[str, float]:
    """Estimate GPU training costs and time."""
    
    # GPU specifications and costs
    gpu_specs = {
        "rtx4090": {
            "vram": 24,
            "hourly_cost": 0.50,  # RunPod/cloud estimate
            "training_speed": 1.0  # Baseline
        },
        "rtx5090": {
            "vram": 32,
            "hourly_cost": 0.70,
            "training_speed": 1.3
        },
        "a100": {
            "vram": 80,
            "hourly_cost": 2.50,
            "training_speed": 2.0
        },
        "h100": {
            "vram": 80,
            "hourly_cost": 4.00,
            "training_speed": 3.0
        }
    }
    
    if gpu_type not in gpu_specs:
        gpu_type = "rtx5090"
    
    specs = gpu_specs[gpu_type]
    
    # Estimate training time based on dataset size
    base_time_hours = max(0.5, samples / 2000)  # Rough estimate
    adjusted_time = base_time_hours / specs["training_speed"]
    
    training_cost = adjusted_time * specs["hourly_cost"]
    
    return {
        "gpu_type": gpu_type,
        "vram_gb": specs["vram"],
        "estimated_training_hours": adjusted_time,
        "hourly_cost": specs["hourly_cost"],
        "total_training_cost": training_cost,
        "samples": samples
    }

def estimate_storage_requirements(samples: int) -> Dict[str, float]:
    """Estimate storage requirements."""
    
    # Rough estimates
    bytes_per_sample = 2048  # JSON sample with scores
    dataset_size_mb = (samples * bytes_per_sample) / (1024 * 1024)
    
    # Additional storage for models, logs, etc.
    base_model_gb = 7.0  # TinyLlama base model
    adapter_files_mb = 50  # PEFT adapter files
    reward_model_gb = 1.5  # Reward model
    logs_mb = 100  # Training logs
    
    total_storage_gb = (
        dataset_size_mb / 1024 +
        base_model_gb +
        adapter_files_mb / 1024 +
        reward_model_gb +
        logs_mb / 1024
    )
    
    return {
        "dataset_size_mb": dataset_size_mb,
        "reward_dataset_samples": samples,
        "total_storage_gb": total_storage_gb,
        "breakdown": {
            "dataset_mb": dataset_size_mb,
            "base_model_gb": base_model_gb,
            "adapters_mb": adapter_files_mb,
            "reward_model_gb": reward_model_gb,
            "logs_mb": logs_mb
        }
    }

def print_cost_breakdown(api_costs: Dict, training_costs: Dict, storage: Dict):
    """Print detailed cost breakdown."""
    
    print("💰 RLAIF Pipeline Cost Estimation")
    print("=" * 50)
    print()
    
    # API Costs
    print(f"📡 OpenAI API Costs ({api_costs['model_used']}):")
    print(f"   Samples: {api_costs['total_samples']:,}")
    print(f"   Input tokens: {api_costs['total_input_tokens']:,}")
    print(f"   Output tokens: {api_costs['total_output_tokens']:,}")
    print(f"   Input cost: ${api_costs['input_cost']:.2f}")
    print(f"   Output cost: ${api_costs['output_cost']:.2f}")
    print(f"   Total API cost: ${api_costs['total_cost']:.2f}")
    print()
    
    # Training Costs
    print(f"🖥️  GPU Training Costs ({training_costs['gpu_type']}):")
    print(f"   GPU VRAM: {training_costs['vram_gb']}GB")
    print(f"   Training time: {training_costs['estimated_training_hours']:.1f} hours")
    print(f"   Hourly rate: ${training_costs['hourly_cost']:.2f}")
    print(f"   Total training cost: ${training_costs['total_training_cost']:.2f}")
    print()
    
    # Storage Requirements
    print(f"💾 Storage Requirements:")
    print(f"   Dataset: {storage['dataset_size_mb']:.1f}MB")
    print(f"   Total storage: {storage['total_storage_gb']:.1f}GB")
    print()
    
    # Total Costs
    total_cost = api_costs['total_cost'] + training_costs['total_training_cost']
    print(f"💵 Total Estimated Cost: ${total_cost:.2f}")
    print(f"   API: ${api_costs['total_cost']:.2f}")
    print(f"   Training: ${training_costs['total_training_cost']:.2f}")
    print()
    
    # Recommendations
    print("💡 Recommendations:")
    if total_cost > 100:
        print("   • Start with fewer samples (500-1000) for initial testing")
    if training_costs['estimated_training_hours'] > 3:
        print("   • Consider using a more powerful GPU to reduce training time")
    if storage['total_storage_gb'] > 20:
        print("   • Ensure sufficient disk space before starting")
    
    print("   • Monitor OpenAI API usage during dataset generation")
    print("   • Use rate limiting to prevent unexpected charges")
    print("   • Start with small batch sizes to test the pipeline")

def main():
    parser = argparse.ArgumentParser(description="Estimate RLAIF pipeline costs")
    parser.add_argument("--samples", type=int, default=1000, 
                       help="Number of samples for reward dataset")
    parser.add_argument("--api-model", default="gpt-4",
                       choices=["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
                       help="OpenAI model for evaluation")
    parser.add_argument("--gpu", default="rtx5090",
                       choices=["rtx4090", "rtx5090", "a100", "h100"],
                       help="GPU type for training")
    parser.add_argument("--json", action="store_true",
                       help="Output as JSON")
    
    args = parser.parse_args()
    
    # Calculate estimates
    api_costs = estimate_api_costs(args.samples, args.api_model)
    training_costs = estimate_training_costs(args.samples, args.gpu)
    storage = estimate_storage_requirements(args.samples)
    
    if args.json:
        # JSON output
        result = {
            "api_costs": api_costs,
            "training_costs": training_costs,
            "storage_requirements": storage,
            "total_estimated_cost": api_costs['total_cost'] + training_costs['total_training_cost']
        }
        print(json.dumps(result, indent=2))
    else:
        # Human-readable output
        print_cost_breakdown(api_costs, training_costs, storage)

if __name__ == "__main__":
    main()