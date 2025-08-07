#!/bin/bash

# RTX 5090 Training Setup Verification Script
# Validates environment, dependencies, and model files

echo "🔍 RTX 5090 Training Setup Verification"
echo "======================================"

# Check CUDA/GPU availability
echo "1. GPU and CUDA Status:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader
    echo "✅ NVIDIA driver detected"
else
    echo "❌ NVIDIA drivers not found"
fi

# Check Python and dependencies
echo -e "\n2. Python Environment:"
python --version
echo "✅ Python available"

# Check key packages
echo -e "\n3. Key Dependencies:"
python -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null && echo "✅ PyTorch" || echo "❌ PyTorch"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')" 2>/dev/null && echo "✅ Transformers" || echo "❌ Transformers"
python -c "import peft; print(f'PEFT: {peft.__version__}')" 2>/dev/null && echo "✅ PEFT" || echo "❌ PEFT"
python -c "import trl; print(f'TRL: {trl.__version__}')" 2>/dev/null && echo "✅ TRL" || echo "❌ TRL"

# Check CUDA in PyTorch
echo -e "\n4. PyTorch CUDA Status:"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA Devices: {torch.cuda.device_count()}')" 2>/dev/null

# Check training data
echo -e "\n5. Training Data:"
if [ -f "data/enhanced_training_data.json" ]; then
    lines=$(wc -l < "data/enhanced_training_data.json")
    echo "✅ Training data found: $lines samples"
else
    echo "❌ Training data not found (data/enhanced_training_data.json)"
fi

# Check configuration
echo -e "\n6. Training Configuration:"
if [ -f "training/config/rtx5090_optimized.yaml" ]; then
    echo "✅ RTX 5090 config found"
    cat training/config/rtx5090_optimized.yaml | head -10
else
    echo "❌ RTX 5090 config not found"
fi

# Check model directory
echo -e "\n7. Model Output Directory:"
if [ -d "models/wp-slm-rtx5090" ]; then
    echo "✅ Model directory exists"
    echo "Contents:"
    ls -la models/wp-slm-rtx5090/
else
    echo "⚠️  Model directory doesn't exist (will be created)"
fi

# Check disk space
echo -e "\n8. Disk Space:"
df -h . | head -2

echo -e "\n🎯 Setup Verification Complete"
echo "Ready to run: python training/sft_train.py --config training/config/rtx5090_optimized.yaml"