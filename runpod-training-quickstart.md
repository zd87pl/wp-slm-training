# WordPress SLM Training on RunPod RTX 4090 - Complete Guide

## Quick Start (TL;DR)
```bash
# 1. Clone and setup
git clone https://github.com/your-username/wp-slm-development.git
cd wp-slm-development
chmod +x runpod/quickstart_working.sh
./runpod/quickstart_working.sh

# 2. Generate dataset and train
chmod +x scripts/build_massive_dataset.sh
./scripts/build_massive_dataset.sh
```

## Detailed Step-by-Step Instructions

### Step 1: Clone Repository and Setup Environment

```bash
# Clone your WordPress SLM repository
git clone https://github.com/your-username/wp-slm-development.git
cd wp-slm-development

# Verify you're in the right directory
ls -la
# You should see: training/, scripts/, runpod/, data/, etc.
```

### Step 2: Run the Automated Setup Script

```bash
# Make the quickstart script executable
chmod +x runpod/quickstart_working.sh

# Run the automated setup (this installs all dependencies)
./runpod/quickstart_working.sh
```

**What this script does:**
- Updates system packages
- Installs Python dependencies (transformers, peft, trl, datasets, etc.)
- Sets up the training environment
- Downloads and caches the base model (TinyLlama-1.1B-Chat-v1.0)
- Verifies GPU availability

### Step 3: Generate Training Dataset

```bash
# Make the dataset generation script executable
chmod +x scripts/build_massive_dataset.sh

# Generate 10K+ WordPress training examples
./scripts/build_massive_dataset.sh
```

**What this creates:**
- `data/sft/train_dataset.jsonl` (8000+ examples)
- `data/sft/eval_dataset.jsonl` (2000+ examples)
- Complete WordPress knowledge coverage (plugins, themes, security, etc.)

### Step 4: Verify Setup Before Training

```bash
# Check GPU availability
nvidia-smi

# Verify Python environment
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Check training data
ls -la data/sft/
wc -l data/sft/*.jsonl
```

**Expected output:**
```
CUDA available: True
GPU: NVIDIA GeForce RTX 4090
train_dataset.jsonl: ~8000 lines
eval_dataset.jsonl: ~2000 lines
```

### Step 5: Start Training

```bash
# Option A: Quick training (uses working config - 1 epoch)
python training/sft_train.py \
  --config training/config/runpod_working.yaml \
  --output_dir /workspace/outputs/wp-slm-quick

# Option B: Enhanced training (3 epochs, better performance)
python training/sft_train.py \
  --config training/config/enhanced_training.yaml \
  --output_dir /workspace/outputs/wp-slm-enhanced
```

### Step 6: Monitor Training Progress

```bash
# In another terminal, monitor GPU usage
watch -n 2 nvidia-smi

# Check training logs
tail -f /workspace/outputs/wp-slm-*/training_log.txt

# Monitor training progress (if available)
ls -la /workspace/outputs/wp-slm-*/checkpoint-*
```

**Training Timeline:**
- **Quick config**: ~30-45 minutes
- **Enhanced config**: ~2-3 hours
- **Checkpoints**: Saved every 500 steps

### Step 7: Verify Training Completion

```bash
# Check final output directory
ls -la /workspace/outputs/wp-slm-*/

# Expected files:
# - adapter_model.safetensors (your trained LoRA adapter)
# - adapter_config.json
# - tokenizer files
# - training_stats.json
# - README.md

# Check training metrics
cat /workspace/outputs/wp-slm-*/training_stats.json
```

### Step 8: Test Your Trained Model

```bash
# Create quick test script
cat > test_model.py << 'EOF'
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def test_wordpress_slm(model_path):
    print("🔄 Loading your trained WordPress SLM...")
    
    # Load tokenizer and models
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, model_path).to("cuda")
    
    # Test questions
    questions = [
        "How do I create a WordPress plugin?",
        "What are WordPress hooks?",
        "How to secure WordPress?"
    ]
    
    print("\n🧪 Testing WordPress SLM:")
    print("=" * 50)
    
    for i, question in enumerate(questions, 1):
        print(f"\n📝 Test {i}: {question}")
        print("-" * 40)
        
        # Format for TinyLlama
        prompt = f"<|system|>\nYou are a WordPress expert.</s>\n<|user|>\n{question}</s>\n<|assistant|>\n"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7, do_sample=True)
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = response.split("<|assistant|>")[-1].strip()
        
        print(f"🤖 WordPress SLM: {answer}")
        print("=" * 50)

if __name__ == "__main__":
    # Test your trained model
    model_path = "/workspace/outputs/wp-slm-enhanced"  # or wp-slm-quick
    test_wordpress_slm(model_path)
EOF

# Run the test
python test_model.py
```

## Alternative: Manual Step-by-Step Training

If you prefer manual control over each step:

### Manual Dataset Generation

```bash
# Generate WordPress dataset manually
python scripts/automated_wp_dataset.py --output_dir data/sft --num_examples 10000

# Verify dataset
head -5 data/sft/train_dataset.jsonl
echo "Total training examples:"
wc -l data/sft/train_dataset.jsonl
```

### Manual Training Execution

```bash
# Install specific versions if needed
pip install transformers==4.36.0 peft==0.7.1 trl==0.7.4

# Run training with custom parameters
python training/sft_train.py \
  --train_dataset data/sft/train_dataset.jsonl \
  --eval_dataset data/sft/eval_dataset.jsonl \
  --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --output_dir /workspace/outputs/custom-wp-slm \
  --num_epochs 3 \
  --batch_size 4 \
  --learning_rate 5e-5 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --max_seq_length 1024
```

## Configuration Options

### Quick Training (runpod_working.yaml)
```yaml
model_name: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
num_epochs: 1
batch_size: 4
learning_rate: 0.00005
lora_rank: 8
max_seq_length: 512
```

### Enhanced Training (enhanced_training.yaml)
```yaml
model_name: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
num_epochs: 3
batch_size: 4
learning_rate: 0.00005
lora_rank: 16
lora_alpha: 32
max_seq_length: 1024
lr_scheduler_type: "cosine"
```

## Troubleshooting

### Common Issues and Solutions

**1. CUDA Out of Memory**
```bash
# Reduce batch size in config
sed -i 's/batch_size: 4/batch_size: 2/' training/config/runpod_working.yaml
```

**2. Dependencies Issues**
```bash
# Reinstall with specific versions
pip install --upgrade transformers==4.36.0 peft==0.7.1 trl==0.7.4 datasets==2.15.0
```

**3. Training Hangs/Stops**
```bash
# Check GPU memory
nvidia-smi
# Check disk space
df -h
# Restart training from last checkpoint
python training/sft_train.py --config training/config/runpod_working.yaml --resume_from_checkpoint /workspace/outputs/wp-slm-*/checkpoint-500
```

**4. Dataset Generation Fails**
```bash
# Generate smaller dataset first
python scripts/automated_wp_dataset.py --output_dir data/sft --num_examples 1000
```

## Expected Performance Metrics

Based on previous successful training:

### Training Progress
- **Initial Loss**: ~2.5-3.0
- **Target Training Loss**: <0.05
- **Target Eval Loss**: <0.01
- **Best Achievement**: Training: 0.0140, Eval: 0.0009

### Resource Usage
- **GPU Memory**: ~18-20GB (RTX 4090 has 24GB)
- **Training Time**: 30min (quick) to 3hrs (enhanced)
- **Disk Usage**: ~5-10GB for datasets and model

## Next Steps After Training

### 1. Upload to HuggingFace
```bash
# Install HuggingFace CLI
pip install huggingface_hub

# Login and upload
huggingface-cli login
huggingface-cli upload your-username/wordpress-slm-v2 /workspace/outputs/wp-slm-enhanced/
```

### 2. Test Performance
```bash
# Run comprehensive evaluation
python eval/run_eval.py --model_path /workspace/outputs/wp-slm-enhanced --test_file data/eval/wp_test_set.jsonl
```

### 3. Deploy to Production
- Use the Vertex AI architecture plan we created
- Or deploy to RunPod Serverless for API access

## Success Indicators

✅ **Training Completes** without errors  
✅ **GPU Utilization** stays high (80-95%)  
✅ **Loss Decreases** consistently  
✅ **Eval Loss < 0.1** (good performance)  
✅ **Model Responds** coherently to WordPress questions  
✅ **Files Created** in output directory  

Your WordPress SLM should achieve exceptional performance similar to the previous training (98-99% improvement) and provide expert-level WordPress assistance!