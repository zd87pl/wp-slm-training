# WordPress SLM on RunPod - Deployment Guide

This guide explains how to deploy and run the WordPress SLM pipeline on RunPod with GPU acceleration.

## Overview

RunPod provides cloud GPU instances perfect for:
- Training the WordPress SLM model
- Running inference servers
- Batch processing
- Development and experimentation

## Deployment Options

### Option 1: RunPod Pods (Interactive Development)

Best for: Development, debugging, and interactive training.

1. **Create a GPU Pod:**
   ```
   GPU: RTX 4090 or A5000 (24GB VRAM)
   Template: RunPod PyTorch 2.1
   Disk: 50GB+
   ```

2. **SSH into the pod and clone the repo:**
   ```bash
   git clone https://github.com/zd87pl/wp-slm-training.git wp-slm
   cd wp-slm
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

4. **Run training:**
   ```bash
   # Download/prepare data
   make data
   
   # Start training
   make sft
   ```

5. **Launch inference server:**
   ```bash
   python inference/serve_vllm.py --model outputs/wp-slm-merged
   ```

### Option 2: RunPod Serverless (Production Inference)

Best for: Scalable inference API.

1. **Build and push Docker image:**
   ```bash
   # Build image
   docker build -f runpod/Dockerfile -t wp-slm-runpod .
   
   # Tag for RunPod registry
   docker tag wp-slm-runpod runpod/wp-slm:latest
   
   # Push to registry (requires RunPod account)
   docker push runpod/wp-slm:latest
   ```

2. **Create Serverless Endpoint:**
   - Go to RunPod Console → Serverless → Create Endpoint
   - Container Image: `runpod/wp-slm:latest`
   - GPU: RTX 4090 or A5000
   - Max Workers: Based on load
   - Environment Variables:
     ```
     MODEL_PATH=/workspace/models/wp-slm
     ```

3. **Mount model volume:**
   - Create a network volume with your trained model
   - Mount at `/workspace/models`

4. **Test the endpoint:**
   ```python
   import requests
   
   response = requests.post(
       "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync",
       headers={"Authorization": "Bearer YOUR_API_KEY"},
       json={
           "input": {
               "prompt": "How do I create a custom post type in WordPress?",
               "temperature": 0.7
           }
       }
   )
   ```

### Option 3: Batch Training Job

Best for: Automated training runs.

1. **Create training script** (`runpod/train.sh`):
   ```bash
   #!/bin/bash
   cd /workspace/wp-slm
   
   # Prepare data
   python scripts/scrape_wp_docs.py
   python scripts/parse_wp_docs.py
   python scripts/build_sft_pairs.py
   python scripts/split_dataset.py
   
   # Train model
   accelerate launch training/sft_train.py \
     --config training/config/sft_qlora.yaml \
     --train_file data/sft/train.jsonl \
     --eval_file data/sft/val.jsonl
   
   # Save to network volume
   cp -r outputs/wp-sft-qlora /workspace/models/
   ```

2. **Submit as RunPod job:**
   ```python
   import runpod
   
   runpod.api_key = "YOUR_API_KEY"
   
   job = runpod.create_job(
       "wp-slm-training",
       "runpod/wp-slm:latest",
       "24GB",  # GPU memory
       command="bash /workspace/wp-slm/runpod/train.sh"
   )
   ```

## Docker Compose for RunPod

For complex setups with multiple services:

```yaml
# runpod/docker-compose.yml
version: '3.8'

services:
  wp-slm:
    build:
      context: ..
      dockerfile: runpod/Dockerfile
    image: wp-slm-runpod
    volumes:
      - models:/workspace/models
      - data:/workspace/data
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - MODEL_PATH=/workspace/models/wp-slm
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  inference:
    image: wp-slm-runpod
    command: python inference/serve_vllm.py --model /workspace/models/wp-slm
    ports:
      - "8000:8000"
    volumes:
      - models:/workspace/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  models:
  data:
```

## Cost Optimization

### GPU Selection for RunPod:
- **Training**: RTX 4090 ($0.44/hr) or A5000 ($0.34/hr)
- **Inference**: RTX 3090 ($0.22/hr) or A4000 ($0.20/hr)
- **Development**: RTX 3080 ($0.17/hr)

### Tips:
1. Use spot instances for training (up to 50% cheaper)
2. Use serverless for variable inference loads
3. Save models to network volumes to avoid re-downloading
4. Use smaller GPUs for development/testing

## Environment Variables

Set these in your RunPod configuration:

```bash
# Model configuration
MODEL_PATH=/workspace/models/wp-slm
BASE_MODEL=meta-llama/Llama-2-7b-hf

# Training configuration
BATCH_SIZE=8
LEARNING_RATE=2e-5
NUM_EPOCHS=1

# API configuration
API_PORT=8000
MAX_WORKERS=4
```

## Monitoring and Logs

### View logs in RunPod:
```bash
# Training logs
tail -f /workspace/wp-slm/outputs/wp-sft-qlora/trainer_state.json

# Inference logs
journalctl -u wp-slm-inference -f
```

### Monitor GPU usage:
```bash
nvidia-smi -l 1  # Update every second
nvtop            # Interactive GPU monitor
```

## Troubleshooting

### Common Issues & SOLUTIONS:

1. **CUDA Out of Memory:**
   - Reduce batch size in config
   - ⚠️ **DO NOT enable gradient checkpointing with LoRA** (causes gradient errors)
   - Use smaller sequence length

2. **LoRA Gradient Errors ("does not require grad"):**
   ```bash
   # SOLUTION: Update setup_peft method in sft_train.py
   # Add after get_peft_model():
   for name, param in self.model.named_parameters():
       if param.requires_grad:
           param.data = param.data.float()
           param.requires_grad_(True)
   ```

3. **KeyError: 'evaluation_strategy':**
   ```bash
   # SOLUTION: RunPod uses older Transformers - use 'eval_strategy'
   sed -i 's/evaluation_strategy:/eval_strategy:/' training/config/your_config.yaml
   sed -i 's/training_config\["evaluation_strategy"\]/training_config["eval_strategy"]/' training/sft_train.py
   ```

4. **TypeError: '<=' not supported between float and str:**
   ```bash
   # SOLUTION: Use decimal notation for learning rate
   # WRONG: learning_rate: 5e-5
   # RIGHT: learning_rate: 0.00005
   ```

5. **Docker Build Fails (exit code 100):**
   ```bash
   # SOLUTION: Remove problematic CUDA repos before apt-get
   rm -f /etc/apt/sources.list.d/cuda*
   ```

6. **Statistics Crash (NoneType format error):**
   - ✅ **FIXED in runpod_working.yaml** - Uses null-safe statistics handling

### Debug Commands:
```bash
# Check GPU and CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name()}')"

# Test LoRA setup
python -c "from peft import LoraConfig; print('PEFT OK')"

# Verify training config
python -c "import yaml; print(yaml.safe_load(open('training/config/runpod_working.yaml'))['training']['eval_strategy'])"

# Check for working files
ls -la training/config/runpod_working.yaml
grep -n "eval_strategy" training/sft_train.py
```

### WORKING COMMAND (Copy & Paste Ready):
```bash
# Complete working setup for RunPod
python training/sft_train.py \
  --config training/config/runpod_working.yaml \
  --train_file /workspace/data/sft/train.jsonl \
  --eval_file /workspace/data/sft/val.jsonl
```

**Expected Results with runpod_working.yaml:**
- ✅ Training completes without errors
- ✅ ~6.3M trainable parameters (0.57%)
- ✅ Final training loss: ~0.79
- ✅ Final evaluation loss: ~0.76
- ✅ Model saved to `/workspace/outputs/test-model/`

## Quick Start Script - TESTED & WORKING ✅

Save as `runpod/quickstart.sh` - This has been validated on RunPod RTX 4090:

```bash
#!/bin/bash
# WordPress SLM Training - Tested Working Setup for RunPod
# Last validated: 2025-01-25 on RTX 4090

echo "🚀 Setting up WordPress SLM Training Environment..."

# Install dependencies
cd /workspace
git clone https://github.com/zd87pl/wp-slm-training.git wp-slm
cd wp-slm
pip install -r requirements.txt

# CRITICAL FIX: Remove problematic CUDA repository sources
rm -f /etc/apt/sources.list.d/cuda*

# Create sample training data (replace with your data)
mkdir -p data/sft
cat > data/sft/train.jsonl << 'EOF'
{"prompt": "How do I create a custom post type in WordPress?", "response": "To create a custom post type in WordPress, use the register_post_type() function..."}
{"prompt": "What is the WordPress loop?", "response": "The WordPress loop is a PHP code block that displays posts..."}
EOF

cp data/sft/train.jsonl data/sft/val.jsonl

echo "✅ Environment setup complete!"
echo "🎯 Starting training with WORKING configuration..."

# Start training with VALIDATED config
python training/sft_train.py \
  --config training/config/runpod_working.yaml \
  --train_file data/sft/train.jsonl \
  --eval_file data/sft/val.jsonl

echo "🎉 Training completed! Check /workspace/outputs/test-model/ for results"
```

## CRITICAL FIXES APPLIED ⚠️

**These fixes are REQUIRED for RunPod compatibility:**

### 1. Docker Build Fix
```bash
# Remove problematic CUDA repository sources
rm -f /etc/apt/sources.list.d/cuda*
```

### 2. LoRA Gradient Fix
Update `training/sft_train.py` setup_peft method:
```python
# CRITICAL: Enable gradients for LoRA parameters
for name, param in self.model.named_parameters():
    if param.requires_grad:
        param.data = param.data.float()
        param.requires_grad_(True)
```

### 3. Configuration Compatibility
Use `training/config/runpod_working.yaml` with these CRITICAL settings:
- `eval_strategy: steps` (NOT `evaluation_strategy`)
- `learning_rate: 0.00005` (decimal, NOT scientific notation)
- `gradient_checkpointing: false` (MUST be disabled for LoRA)
- All precision flags disabled: `bf16: false, fp16: false, tf32: false`

### 4. Statistics Crash Fix
Update `_save_training_stats` method to handle None values safely.

## Next Steps

1. **Start with a Pod** for development and testing
2. **Train your model** using the provided scripts
3. **Deploy to Serverless** for production inference
4. **Monitor costs** and optimize GPU usage

RunPod provides excellent GPU resources for WordPress SLM development at competitive prices!