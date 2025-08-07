#!/bin/bash

# RTX 5090 Optimized WordPress SLM Training Script
# 32GB VRAM Maximization - Expected 3-4 hour training
# Target: >98% improvement over base model performance

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CONFIG_FILE="training/config/rtx5090_optimized.yaml"
MODEL_OUTPUT_DIR="./models/wp-slm-rtx5090"
LOG_DIR="./logs/rtx5090"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create necessary directories
mkdir -p "$MODEL_OUTPUT_DIR" "$LOG_DIR" "data/sft"

echo -e "${BLUE}🚀 RTX 5090 WordPress SLM Training Started${NC}"
echo -e "${BLUE}Timestamp: $TIMESTAMP${NC}"
echo -e "${BLUE}Expected Duration: 3-4 hours${NC}"

# Function to log with timestamp
log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to check GPU status
check_gpu() {
    log "${YELLOW}🔍 Checking RTX 5090 Status...${NC}"
    
    if ! command -v nvidia-smi &> /dev/null; then
        log "${RED}❌ nvidia-smi not found. GPU monitoring disabled.${NC}"
        return 1
    fi
    
    # Check for RTX 5090
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
    log "${GREEN}GPU Detected: $GPU_NAME${NC}"
    
    # Check VRAM
    TOTAL_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    log "${GREEN}Total VRAM: ${TOTAL_VRAM}MB${NC}"
    
    if [ "$TOTAL_VRAM" -lt 30000 ]; then
        log "${YELLOW}⚠️  Warning: Expected 32GB VRAM, detected ${TOTAL_VRAM}MB${NC}"
    fi
    
    # Initial GPU status
    nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv > "$LOG_DIR/gpu_status_initial.csv"
    
    return 0
}

# Function to start GPU monitoring
start_gpu_monitoring() {
    if command -v nvidia-smi &> /dev/null; then
        log "${YELLOW}📊 Starting GPU monitoring...${NC}"
        # Monitor every 30 seconds during training
        nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv -l 30 > "$LOG_DIR/gpu_monitoring_$TIMESTAMP.csv" &
        GPU_MONITOR_PID=$!
        log "${GREEN}GPU monitoring started (PID: $GPU_MONITOR_PID)${NC}"
    fi
}

# Function to stop GPU monitoring
stop_gpu_monitoring() {
    if [ -n "${GPU_MONITOR_PID:-}" ]; then
        log "${YELLOW}⏹️  Stopping GPU monitoring...${NC}"
        kill $GPU_MONITOR_PID 2>/dev/null || true
        wait $GPU_MONITOR_PID 2>/dev/null || true
    fi
}

# Function to validate dataset
validate_dataset() {
    log "${YELLOW}🔍 Validating enhanced dataset...${NC}"
    
    TRAIN_FILE="data/sft/wp_enhanced_25k_train.jsonl"
    EVAL_FILE="data/sft/wp_enhanced_25k_eval.jsonl"
    
    if [ ! -f "$TRAIN_FILE" ]; then
        log "${RED}❌ Training file not found: $TRAIN_FILE${NC}"
        log "${YELLOW}Run: bash scripts/generate_enhanced_dataset.sh${NC}"
        exit 1
    fi
    
    if [ ! -f "$EVAL_FILE" ]; then
        log "${RED}❌ Evaluation file not found: $EVAL_FILE${NC}"
        log "${YELLOW}Run: bash scripts/generate_enhanced_dataset.sh${NC}"
        exit 1
    fi
    
    # Count samples
    TRAIN_COUNT=$(wc -l < "$TRAIN_FILE")
    EVAL_COUNT=$(wc -l < "$EVAL_FILE")
    TOTAL_COUNT=$((TRAIN_COUNT + EVAL_COUNT))
    
    log "${GREEN}✅ Dataset validated:${NC}"
    log "  Training samples: $TRAIN_COUNT"
    log "  Evaluation samples: $EVAL_COUNT"
    log "  Total samples: $TOTAL_COUNT"
    
    if [ "$TOTAL_COUNT" -lt 20000 ]; then
        log "${YELLOW}⚠️  Warning: Expected 25K samples, found $TOTAL_COUNT${NC}"
    fi
}

# Function to optimize CUDA settings
optimize_cuda() {
    log "${YELLOW}⚙️  Optimizing CUDA settings for RTX 5090...${NC}"
    
    # CUDA optimization environment variables
    export CUDA_LAUNCH_BLOCKING=0
    export CUDA_CACHE_DISABLE=0
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES=0
    
    # Memory optimization
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True
    export TORCH_CUDA_ARCH_LIST="8.9"  # RTX 5090 architecture
    
    # Performance optimization
    export OMP_NUM_THREADS=12
    export MKL_NUM_THREADS=12
    
    log "${GREEN}✅ CUDA settings optimized${NC}"
}

# Function to create training command
create_training_command() {
    TRAIN_FILE="data/sft/wp_enhanced_25k_train.jsonl"
    EVAL_FILE="data/sft/wp_enhanced_25k_eval.jsonl"
    
    # Only pass core arguments - configuration is handled by YAML file
    TRAINING_CMD="python3 training/sft_train.py"
    TRAINING_CMD+=" --config $CONFIG_FILE"
    TRAINING_CMD+=" --train_file $TRAIN_FILE"
    TRAINING_CMD+=" --eval_file $EVAL_FILE"
    
    echo "$TRAINING_CMD"
}

# Function to validate training output
validate_training() {
    log "${YELLOW}🔍 Validating training output...${NC}"
    
    # Check if PEFT/LoRA adapter files exist
    if [ -d "$MODEL_OUTPUT_DIR" ]; then
        # Look for PEFT-specific files
        ADAPTER_CONFIG="$MODEL_OUTPUT_DIR/adapter_config.json"
        ADAPTER_MODEL="$MODEL_OUTPUT_DIR/adapter_model.safetensors"
        
        if [ -f "$ADAPTER_CONFIG" ] && [ -f "$ADAPTER_MODEL" ]; then
            MODEL_SIZE=$(du -sh "$ADAPTER_MODEL" | cut -f1)
            log "${GREEN}✅ PEFT adapter model created successfully${NC}"
            log "  - adapter_config.json: $(stat -f%z "$ADAPTER_CONFIG" 2>/dev/null || stat -c%s "$ADAPTER_CONFIG" 2>/dev/null) bytes"
            log "  - adapter_model.safetensors: $MODEL_SIZE"
        else
            log "${RED}❌ PEFT adapter files not found${NC}"
            log "  Expected: $ADAPTER_CONFIG"
            log "  Expected: $ADAPTER_MODEL"
            return 1
        fi
    else
        log "${RED}❌ Output directory not found${NC}"
        return 1
    fi
    
    return 0
}

# Function to generate training report
generate_report() {
    log "${YELLOW}📋 Generating training report...${NC}"
    
    REPORT_FILE="$LOG_DIR/training_report_$TIMESTAMP.txt"
    
    {
        echo "RTX 5090 WordPress SLM Training Report"
        echo "======================================"
        echo "Timestamp: $TIMESTAMP"
        echo "Configuration: $CONFIG_FILE"
        echo "Output Directory: $MODEL_OUTPUT_DIR"
        echo ""
        
        if command -v nvidia-smi &> /dev/null; then
            echo "GPU Information:"
            nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
            echo ""
        fi
        
        echo "Dataset Information:"
        if [ -f "data/sft/wp_enhanced_25k_train.jsonl" ]; then
            echo "Training samples: $(wc -l < data/sft/wp_enhanced_25k_train.jsonl)"
        fi
        if [ -f "data/sft/wp_enhanced_25k_eval.jsonl" ]; then
            echo "Evaluation samples: $(wc -l < data/sft/wp_enhanced_25k_eval.jsonl)"
        fi
        echo ""
        
        echo "Training Configuration:"
        echo "- Batch size: 16"
        echo "- Gradient accumulation: 2 (effective batch size: 32)"
        echo "- LoRA rank: 32"
        echo "- Learning rate: 2e-4"
        echo "- Epochs: 5"
        echo "- Sequence length: 2048"
        echo ""
        
        # Check for PEFT adapter files specifically
        if [ -f "$MODEL_OUTPUT_DIR/adapter_model.safetensors" ] && [ -f "$MODEL_OUTPUT_DIR/adapter_config.json" ]; then
            echo "✅ PEFT Training completed successfully"
            MODEL_SIZE=$(du -sh "$MODEL_OUTPUT_DIR/adapter_model.safetensors" | cut -f1)
            echo "Adapter model size: $MODEL_SIZE"
            echo "Training method: LoRA/PEFT (Parameter Efficient Fine-Tuning)"
        else
            echo "❌ PEFT adapter files not found"
        fi
        
    } > "$REPORT_FILE"
    
    log "${GREEN}📋 Report saved: $REPORT_FILE${NC}"
    cat "$REPORT_FILE"
}

# Cleanup function
cleanup() {
    log "${YELLOW}🧹 Cleaning up...${NC}"
    stop_gpu_monitoring
    
    # Optional: Clear CUDA cache
    if command -v python3 &> /dev/null; then
        python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    fi
}

# Set up signal handlers
trap cleanup EXIT INT TERM

# Main execution
main() {
    log "${BLUE}🚀 Starting RTX 5090 WordPress SLM Training${NC}"
    
    # Pre-flight checks
    check_gpu
    validate_dataset
    optimize_cuda
    
    # Start monitoring
    start_gpu_monitoring
    
    # Create and execute training command
    TRAINING_CMD=$(create_training_command)
    log "${YELLOW}🎯 Training command:${NC}"
    log "$TRAINING_CMD"
    
    log "${GREEN}🚀 Starting training...${NC}"
    eval "$TRAINING_CMD" &
    TRAINING_PID=$!
    
    # Wait for training to complete
    wait $TRAINING_PID
    TRAINING_EXIT_CODE=$?
    
    # Stop monitoring
    stop_gpu_monitoring
    
    # Validate results
    if [ $TRAINING_EXIT_CODE -eq 0 ]; then
        log "${GREEN}🎉 Training completed successfully!${NC}"
        validate_training
        generate_report
        
        log "${GREEN}✅ RTX 5090 WordPress SLM training complete!${NC}"
        log "${BLUE}📁 Model saved to: $MODEL_OUTPUT_DIR${NC}"
        log "${BLUE}📊 Logs saved to: $LOG_DIR${NC}"
        log "${YELLOW}💡 Next steps:${NC}"
        log "  1. Run validation: bash scripts/verify_rtx5090_setup.sh"
        log "  2. Upload model: bash scripts/upload_model.sh"
        log "  3. Test inference: python3 inference/serve_vllm.py"
        
    else
        log "${RED}❌ Training failed with exit code: $TRAINING_EXIT_CODE${NC}"
        generate_report
        exit $TRAINING_EXIT_CODE
    fi
}

# Execute main function
main "$@"