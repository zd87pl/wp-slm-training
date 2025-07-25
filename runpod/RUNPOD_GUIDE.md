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

### Common Issues:

1. **CUDA Out of Memory:**
   - Reduce batch size in config
   - Enable gradient checkpointing
   - Use smaller sequence length

2. **Model not loading:**
   - Check MODEL_PATH environment variable
   - Verify model files exist in volume
   - Check CUDA compatibility

3. **Slow training:**
   - Enable mixed precision (bf16)
   - Use flash attention if supported
   - Check GPU utilization with nvidia-smi

### Debug Commands:
```bash
# Check GPU
python -c "import torch; print(torch.cuda.is_available())"

# Test model loading
python -c "from transformers import AutoModel; print('OK')"

# Verify paths
ls -la /workspace/models/
```

## Quick Start Script

Save as `runpod/quickstart.sh`:

```bash
#!/bin/bash
# Quick setup for RunPod

# Install dependencies
cd /workspace
git clone https://github.com/zd87pl/wp-slm-training.git wp-slm
cd wp-slm
pip install -r requirements.txt

# Download sample data
mkdir -p data/sft
wget https://example.com/sample-wp-data.jsonl -O data/sft/train.jsonl

# Start training
accelerate launch training/sft_train.py \
  --config training/config/sft_qlora.yaml \
  --train_file data/sft/train.jsonl \
  --eval_file data/sft/train.jsonl

echo "Training started! Monitor with: tail -f outputs/*/trainer_state.json"
```

## Next Steps

1. **Start with a Pod** for development and testing
2. **Train your model** using the provided scripts
3. **Deploy to Serverless** for production inference
4. **Monitor costs** and optimize GPU usage

RunPod provides excellent GPU resources for WordPress SLM development at competitive prices!