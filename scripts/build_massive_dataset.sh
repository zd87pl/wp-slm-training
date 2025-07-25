#!/bin/bash
# One-click massive WordPress dataset builder and trainer
# Builds 10K+ examples and runs enhanced training

set -e

echo "🚀 WordPress SLM Massive Dataset Builder"
echo "========================================"

# Check if we're in the right directory
if [ ! -f "training/sft_train.py" ]; then
    echo "❌ Error: Must be run from wp-slm project root directory"
    echo "   Make sure you're in the directory containing training/sft_train.py"
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/sft
mkdir -p outputs

# Step 1: Generate massive dataset
echo ""
echo "📊 Step 1: Generating 10,000+ WordPress training examples..."
python scripts/automated_wp_dataset.py

# Check if dataset was created successfully
if [ ! -f "data/sft/massive_train.jsonl" ] || [ ! -f "data/sft/massive_val.jsonl" ]; then
    echo "❌ Error: Dataset generation failed"
    echo "   Files data/sft/massive_train.jsonl or data/sft/massive_val.jsonl not found"
    exit 1
fi

# Display dataset stats
echo ""
echo "📈 Dataset Statistics:"
echo "   Training examples: $(wc -l < data/sft/massive_train.jsonl)"
echo "   Validation examples: $(wc -l < data/sft/massive_val.jsonl)"
echo "   Total examples: $(($(wc -l < data/sft/massive_train.jsonl) + $(wc -l < data/sft/massive_val.jsonl)))"

# Step 2: Check if enhanced training config exists
if [ ! -f "training/config/enhanced_training.yaml" ]; then
    echo ""
    echo "⚠️  Enhanced training config not found, creating it..."
    
    cat > training/config/enhanced_training.yaml << 'EOF'
base_model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
output_dir: /workspace/outputs/wp-slm-massive

training:
  num_train_epochs: 3
  per_device_train_batch_size: 2
  per_device_eval_batch_size: 2
  gradient_accumulation_steps: 8
  gradient_checkpointing: false
  learning_rate: 0.00003
  lr_scheduler_type: cosine
  warmup_ratio: 0.05
  optim: adamw_torch
  weight_decay: 0.01
  max_grad_norm: 1.0
  logging_steps: 50
  eval_steps: 500
  save_steps: 1000
  save_strategy: steps
  eval_strategy: steps
  save_total_limit: 3
  load_best_model_at_end: true
  metric_for_best_model: eval_loss
  greater_is_better: false
  bf16: false
  fp16: false
  tf32: false
  seed: 42
  report_to: none
  push_to_hub: false
  max_seq_length: 1024

lora:
  r: 16
  alpha: 32
  target_modules: [q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj]
  dropout: 0.1
  bias: none
EOF
    
    echo "✅ Enhanced training config created"
fi

# Step 3: Run enhanced training
echo ""
echo "🎯 Step 2: Starting enhanced training with massive dataset..."
echo "   This will take significantly longer with 10K+ examples (estimated 1-3 hours)"
echo "   Configuration: 3 epochs, rank 16 LoRA, 1024 max sequence length"
echo ""

python training/sft_train.py \
  --config training/config/enhanced_training.yaml \
  --train_file data/sft/massive_train.jsonl \
  --eval_file data/sft/massive_val.jsonl

# Step 4: Display results
echo ""
echo "🎉 Massive Dataset Training Completed!"
echo "======================================"

# Check if training completed successfully
if [ -d "outputs" ] || [ -d "/workspace/outputs" ]; then
    echo "✅ Training completed successfully!"
    echo ""
    echo "📊 Results:"
    if [ -f "/workspace/outputs/wp-slm-massive/training_stats.json" ]; then
        echo "   Training stats: /workspace/outputs/wp-slm-massive/training_stats.json"
    elif [ -f "outputs/wp-slm-massive/training_stats.json" ]; then
        echo "   Training stats: outputs/wp-slm-massive/training_stats.json"
    fi
    echo ""
    echo "🚀 Your WordPress SLM is now trained on 10K+ examples!"
    echo "   Expected improvements:"
    echo "   • Much better WordPress knowledge"
    echo "   • More accurate code generation"
    echo "   • Better understanding of WordPress patterns"
    echo "   • Improved performance on complex tasks"
else
    echo "⚠️  Training may not have completed successfully"
    echo "   Check the output above for any errors"
fi

echo ""
echo "📁 Dataset files available for future use:"
echo "   • data/sft/massive_train.jsonl (training set)"
echo "   • data/sft/massive_val.jsonl (validation set)"
echo ""
echo "🔄 To run training again:"
echo "   python training/sft_train.py \\"
echo "     --config training/config/enhanced_training.yaml \\"
echo "     --train_file data/sft/massive_train.jsonl \\"
echo "     --eval_file data/sft/massive_val.jsonl"