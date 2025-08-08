#!/bin/bash

# WordPress SLM RLAIF Pipeline Runner
# Complete pipeline automation script

set -e  # Exit on any error

# Configuration
MODEL_PATH="./models/wp-slm-rtx5090"
DATASET_PATH="./datasets/wp_reward_data.json"
REWARD_MODEL_PATH="./models/wp-reward-model"
SAMPLES=1000
BATCH_SIZE=50
MAX_CONCURRENT=3

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_requirements() {
    log_info "Checking requirements..."
    
    # Check Python packages
    python -c "import torch, transformers, openai, aiohttp" 2>/dev/null || {
        log_error "Missing required Python packages"
        echo "Install with: pip install torch transformers openai aiohttp tqdm"
        exit 1
    }
    
    # Check OpenAI API key
    if [[ -z "${OPENAI_API_KEY}" ]]; then
        log_error "OPENAI_API_KEY environment variable not set"
        echo "Set with: export OPENAI_API_KEY='your-api-key-here'"
        exit 1
    fi
    
    # Check model exists
    if [[ ! -d "$MODEL_PATH" ]]; then
        log_error "Model not found at $MODEL_PATH"
        echo "Ensure your trained SFT model exists at this path"
        exit 1
    fi
    
    # Check for PEFT files
    if [[ ! -f "$MODEL_PATH/adapter_config.json" ]]; then
        log_error "PEFT adapter files not found in $MODEL_PATH"
        echo "Expected files: adapter_config.json, adapter_model.safetensors"
        exit 1
    fi
    
    log_success "All requirements satisfied"
}

estimate_costs() {
    log_info "Estimating costs for $SAMPLES samples..."
    
    # Rough cost estimation
    local estimated_cost=$(python -c "
import math
samples = $SAMPLES
# Rough estimate: ~$0.05-0.075 per sample for GPT-4 evaluation
cost_per_sample = 0.06
total_cost = samples * cost_per_sample
print(f'{total_cost:.2f}')
")
    
    log_warning "Estimated OpenAI API cost: \$$estimated_cost"
    echo -n "Continue with dataset generation? (y/N): "
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        log_info "Pipeline cancelled by user"
        exit 0
    fi
}

generate_reward_dataset() {
    log_info "Phase 1: Generating reward dataset..."
    log_info "Model: $MODEL_PATH"
    log_info "Samples: $SAMPLES"
    log_info "Output: $DATASET_PATH"
    
    python scripts/generate_reward_dataset.py \
        --model "$MODEL_PATH" \
        --output "$DATASET_PATH" \
        --samples "$SAMPLES" \
        --batch_size "$BATCH_SIZE" \
        --max_concurrent "$MAX_CONCURRENT" || {
        log_error "Dataset generation failed"
        exit 1
    }
    
    log_success "Reward dataset generated successfully"
    
    # Show dataset statistics
    if [[ -f "$DATASET_PATH" ]]; then
        local sample_count=$(python -c "
import json
with open('$DATASET_PATH', 'r') as f:
    data = json.load(f)
print(len(data['samples']))
")
        local avg_score=$(python -c "
import json
with open('$DATASET_PATH', 'r') as f:
    data = json.load(f)
scores = [s['overall_score'] for s in data['samples']]
print(f'{sum(scores)/len(scores):.3f}')
")
        log_info "Dataset statistics: $sample_count samples, average score: $avg_score"
    fi
}

train_reward_model() {
    log_info "Phase 2: Training reward model..."
    log_info "Dataset: $DATASET_PATH"
    log_info "Output: $REWARD_MODEL_PATH"
    
    python training/reward_model.py \
        --data "$DATASET_PATH" \
        --output "$REWARD_MODEL_PATH" \
        --epochs 3 \
        --batch_size 16 \
        --learning_rate 5e-5 || {
        log_error "Reward model training failed"
        exit 1
    }
    
    log_success "Reward model training completed"
    
    # Show training results
    if [[ -f "$REWARD_MODEL_PATH/training_results.json" ]]; then
        local final_accuracy=$(python -c "
import json
try:
    with open('$REWARD_MODEL_PATH/training_results.json', 'r') as f:
        results = json.load(f)
    print(f\"{results.get('final_eval_accuracy', 'N/A'):.3f}\")
except:
    print('N/A')
")
        log_info "Final validation accuracy: $final_accuracy"
    fi
}

show_next_steps() {
    log_success "RLAIF Pipeline Phase 1-2 Complete!"
    echo
    echo "📁 Generated Files:"
    echo "  - Reward Dataset: $DATASET_PATH"
    echo "  - Reward Model: $REWARD_MODEL_PATH"
    echo
    echo "🚀 Next Steps:"
    echo "  1. Review training results in $REWARD_MODEL_PATH"
    echo "  2. Wait for PPO training implementation (Phase 3)"
    echo "  3. Run complete RLAIF training with improved model"
    echo
    echo "💡 Tips:"
    echo "  - Check training logs for detailed metrics"
    echo "  - Consider generating larger dataset (5000+ samples) for better results"
    echo "  - Monitor OpenAI API usage in your dashboard"
}

# Main pipeline execution
main() {
    echo "🤖 WordPress SLM RLAIF Pipeline"
    echo "==============================="
    echo
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --samples)
                SAMPLES="$2"
                shift 2
                ;;
            --model)
                MODEL_PATH="$2"
                shift 2
                ;;
            --output)
                DATASET_PATH="$2"
                shift 2
                ;;
            --reward-model)
                REWARD_MODEL_PATH="$2"
                shift 2
                ;;
            --batch-size)
                BATCH_SIZE="$2"
                shift 2
                ;;
            --max-concurrent)
                MAX_CONCURRENT="$2"
                shift 2
                ;;
            --skip-cost-check)
                SKIP_COST_CHECK=1
                shift
                ;;
            -h|--help)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --samples N           Number of samples to generate (default: 1000)"
                echo "  --model PATH          Path to SFT model (default: ./models/wp-slm-rtx5090)"
                echo "  --output PATH         Output dataset path (default: ./datasets/wp_reward_data.json)"
                echo "  --reward-model PATH   Reward model output path (default: ./models/wp-reward-model)"
                echo "  --batch-size N        API batch size (default: 50)"
                echo "  --max-concurrent N    Max concurrent requests (default: 3)"
                echo "  --skip-cost-check     Skip cost estimation prompt"
                echo "  -h, --help           Show this help"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
    
    # Create directories
    mkdir -p "$(dirname "$DATASET_PATH")"
    mkdir -p "$(dirname "$REWARD_MODEL_PATH")"
    mkdir -p logs
    
    # Run pipeline
    check_requirements
    
    if [[ -z "$SKIP_COST_CHECK" ]]; then
        estimate_costs
    fi
    
    generate_reward_dataset
    train_reward_model
    show_next_steps
}

# Run main function with all arguments
main "$@"