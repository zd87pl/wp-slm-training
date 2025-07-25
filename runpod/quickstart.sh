#!/bin/bash
# WordPress SLM Quick Start for RunPod
# Run this script after SSHing into your RunPod instance

set -e

echo "🚀 WordPress SLM RunPod Quick Start"
echo "=================================="

# Check if running on RunPod
if [ ! -d "/workspace" ]; then
    echo "❌ Error: This script should be run on a RunPod instance"
    echo "   /workspace directory not found"
    exit 1
fi

# Check GPU
echo "🖥️  Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || {
    echo "❌ No GPU detected!"
    exit 1
}

# Navigate to workspace
cd /workspace

# Clone repository if not exists
if [ ! -d "wp-slm" ]; then
    echo "📦 Cloning WordPress SLM repository..."
    git clone https://github.com/your-repo/wp-slm.git
fi

cd wp-slm

# Install dependencies
echo "📚 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# Install vLLM for inference (optional, may fail on some GPUs)
echo "🔧 Installing vLLM (optional)..."
pip install vllm || echo "⚠️  vLLM installation failed, continuing without it"

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/{raw,processed,sft,prefs,eval}
mkdir -p outputs
mkdir -p models

# Download sample WordPress data (if available)
echo "📥 Setting up sample data..."
if [ ! -f "data/sft/train.jsonl" ]; then
    # Create minimal sample data for testing
    cat > data/sft/train.jsonl << 'EOF'
{"prompt": "How do I create a custom post type in WordPress?", "response": "To create a custom post type in WordPress, use the register_post_type() function..."}
{"prompt": "What is the WordPress hook execution order?", "response": "WordPress hooks execute in the following order: muplugins_loaded, plugins_loaded, setup_theme..."}
{"prompt": "How can I add a REST API endpoint?", "response": "To add a REST API endpoint in WordPress, use register_rest_route()..."}
EOF
    cp data/sft/train.jsonl data/sft/val.jsonl
    cp data/sft/train.jsonl data/sft/test.jsonl
    echo "✅ Sample data created"
fi

# Display options
echo ""
echo "✅ Setup complete! You can now:"
echo ""
echo "1. Run data collection pipeline:"
echo "   make data"
echo ""
echo "2. Start training (SFT):"
echo "   make sft"
echo ""
echo "3. Run DPO alignment (after SFT):"
echo "   make dpo"
echo ""
echo "4. Start inference server:"
echo "   python inference/serve_vllm.py --model outputs/wp-slm-merged"
echo ""
echo "5. Monitor GPU usage:"
echo "   watch -n 1 nvidia-smi"
echo ""
echo "📊 Current GPU Status:"
nvidia-smi

echo ""
echo "🎯 Quick test training (5 steps only):"
echo "accelerate launch training/sft_train.py \\"
echo "  --config training/config/sft_qlora.yaml \\"
echo "  --train_file data/sft/train.jsonl \\"
echo "  --eval_file data/sft/val.jsonl \\"
echo "  --max_steps 5"