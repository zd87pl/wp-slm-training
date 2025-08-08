# WordPress SLM RLAIF Pipeline Usage Guide

This guide walks you through running the complete RLAIF (Reinforcement Learning from AI Feedback) pipeline to improve your WordPress SLM model quality.

## Prerequisites

### Required Dependencies
```bash
pip install torch transformers datasets peft accelerate openai aiohttp tqdm numpy pandas scikit-learn
```

### OpenAI API Key Setup

#### Step 1: Create OpenAI Account
1. Go to [https://platform.openai.com](https://platform.openai.com)
2. Sign up for an account or log in if you already have one
3. Add a payment method in your account settings (required for API access)

#### Step 2: Generate API Key
1. Navigate to [API Keys page](https://platform.openai.com/api-keys)
2. Click **"Create new secret key"**
3. Give it a name (e.g., "WordPress RLAIF Pipeline")
4. Copy the generated key (it starts with `sk-...`)
5. **Important**: Save this key securely - you won't be able to see it again

#### Step 3: Set Environment Variable

**On Linux/macOS (Terminal):**
```bash
export OPENAI_API_KEY='your-api-key-here'
```

**To make it permanent, add to your shell profile:**
```bash
echo 'export OPENAI_API_KEY="your-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

**On Windows (Command Prompt):**
```cmd
set OPENAI_API_KEY=your-api-key-here
```

**On Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY="your-api-key-here"
```

**To make it permanent on Windows:**
1. Search for "Environment Variables" in Start Menu
2. Click "Edit the system environment variables"
3. Click "Environment Variables..." button
4. Under "User variables", click "New..."
5. Variable name: `OPENAI_API_KEY`
6. Variable value: `your-api-key-here`

#### Step 4: Verify API Key Setup
```bash
python -c "import os; print('✓ API Key set' if os.getenv('OPENAI_API_KEY') else '✗ API Key not found')"
```

#### Step 5: Test API Connection
```bash
python -c "
import openai
import os
client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
try:
    response = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[{'role': 'user', 'content': 'Hello!'}],
        max_tokens=5
    )
    print('✓ OpenAI API working correctly')
except Exception as e:
    print(f'✗ API Error: {e}')
"
```

### Required Resources
- **OpenAI API Key**: Set as environment variable `OPENAI_API_KEY` (see setup above)
- **GPU Memory**: Minimum 12GB VRAM for training (RTX 4090/RTX 5090 recommended)
- **Storage**: ~10GB for datasets and models
- **API Budget**: $50-200 for 1000-5000 samples

### API Usage and Billing
- **Pricing**: ~$0.05-0.075 per sample with GPT-4
- **Rate Limits**: 500 requests per minute (tier 1), 5000 per minute (tier 2+)
- **Monitoring**: Check usage at [https://platform.openai.com/usage](https://platform.openai.com/usage)
- **Budget Controls**: Set spending limits in [Billing Settings](https://platform.openai.com/account/billing/limits)

### File Structure
```
wp-slm-development/
├── models/
│   └── wp-slm-rtx5090/          # Your trained SFT model
├── datasets/
│   └── wp_reward_data.json      # Generated reward dataset
├── training/
│   ├── ai_judge.py              # AI judge system
│   ├── reward_model.py          # Reward model implementation
│   └── rlaif_pipeline.md        # Architecture documentation
└── scripts/
    └── generate_reward_dataset.py # Dataset generation script
```

## Phase 1: Generate Reward Dataset

### Step 1: Basic Dataset Generation
Generate a reward dataset using your SFT model and AI judge:

```bash
python scripts/generate_reward_dataset.py \
  --model ./models/wp-slm-rtx5090 \
  --output ./datasets/wp_reward_data.json \
  --samples 1000 \
  --batch_size 50 \
  --max_concurrent 3
```

### Step 2: Large-Scale Dataset (Optional)
For better reward model performance:

```bash
python scripts/generate_reward_dataset.py \
  --model ./models/wp-slm-rtx5090 \
  --output ./datasets/wp_reward_data_large.json \
  --samples 5000 \
  --batch_size 100 \
  --max_concurrent 5
```

### Expected Output
```json
{
  "samples": [
    {
      "prompt": "How do I create a custom WordPress widget?",
      "response": "To create a custom WordPress widget, extend the WP_Widget class...",
      "scores": {
        "code_quality": 0.85,
        "wp_accuracy": 0.90,
        "security": 0.75,
        "clarity": 0.80,
        "completeness": 0.85
      },
      "overall_score": 0.83,
      "feedback": "Good implementation with proper WordPress APIs..."
    }
  ],
  "metadata": {
    "total_samples": 1000,
    "generation_time": "2024-01-15T10:30:00Z",
    "model_used": "./models/wp-slm-rtx5090",
    "average_score": 0.78
  }
}
```

## Phase 2: Train Reward Model

### Step 1: Basic Reward Model Training
Train the reward model on your generated dataset:

```bash
python training/reward_model.py \
  --data ./datasets/wp_reward_data.json \
  --output ./models/wp-reward-model \
  --epochs 3 \
  --batch_size 16 \
  --learning_rate 5e-5
```

### Step 2: Advanced Training Configuration
For better performance with larger datasets:

```bash
python training/reward_model.py \
  --data ./datasets/wp_reward_data_large.json \
  --output ./models/wp-reward-model-v2 \
  --epochs 5 \
  --batch_size 32 \
  --learning_rate 3e-5 \
  --warmup_steps 100 \
  --eval_steps 250 \
  --save_steps 500
```

### Expected Training Output
```
Epoch 1/3: 100%|██████████| 125/125 [02:15<00:00, 1.85it/s]
Train Loss: 0.0245, Eval Loss: 0.0156, Eval Accuracy: 0.847
Epoch 2/3: 100%|██████████| 125/125 [02:14<00:00, 1.86it/s]
Train Loss: 0.0198, Eval Loss: 0.0142, Eval Accuracy: 0.863
Epoch 3/3: 100%|██████████| 125/125 [02:13<00:00, 1.87it/s]
Train Loss: 0.0176, Eval Loss: 0.0138, Eval Accuracy: 0.871

Model saved to ./models/wp-reward-model
```

## Phase 3: PPO Training (Coming Soon)

The PPO training implementation is planned for the next phase. It will use the trained reward model to improve your SFT model through reinforcement learning.

### Planned Usage:
```bash
python training/ppo_train.py \
  --base_model ./models/wp-slm-rtx5090 \
  --reward_model ./models/wp-reward-model \
  --output ./models/wp-slm-rlaif-v1 \
  --epochs 1 \
  --batch_size 8 \
  --learning_rate 1e-6
```

## Cost Estimation

### API Costs (OpenAI GPT-4)
- **1000 samples**: ~$50-75
- **5000 samples**: ~$200-300
- **Rate limiting**: Built-in to prevent overages

### GPU Training Costs
- **Reward Model**: 30-60 minutes on RTX 5090
- **PPO Training**: 2-4 hours on RTX 5090
- **Total VRAM**: 12-16GB peak usage

## Quality Validation

### Reward Model Performance Metrics
```python
# Automatic evaluation included in training
Validation Accuracy: 0.871
Mean Squared Error: 0.0138
Pearson Correlation: 0.923
```

### Score Distribution Analysis
```python
# Generated in training logs
Code Quality: μ=0.78, σ=0.15
WordPress Accuracy: μ=0.82, σ=0.12
Security: μ=0.71, σ=0.18
Clarity: μ=0.75, σ=0.14
Completeness: μ=0.77, σ=0.13
```

## Troubleshooting

### Common Issues

#### 1. OpenAI API Rate Limits
```bash
# Error: Rate limit exceeded
# Solution: Reduce max_concurrent parameter
python scripts/generate_reward_dataset.py --max_concurrent 2
```

#### 2. GPU Memory Issues
```bash
# Error: CUDA out of memory
# Solution: Reduce batch size
python training/reward_model.py --batch_size 8
```

#### 3. Model Loading Errors
```bash
# Error: Model not found
# Solution: Check model path and ensure PEFT adapters exist
ls -la ./models/wp-slm-rtx5090/
# Should contain: adapter_config.json, adapter_model.safetensors
```

### Performance Optimization

#### 1. Dataset Generation Speed
- Use `--max_concurrent 5` for faster generation
- Set `--batch_size 100` for API efficiency
- Monitor costs with `--estimate_cost_only` flag

#### 2. Training Speed
- Use gradient accumulation for larger effective batch sizes
- Enable mixed precision training (FP16)
- Use multiple GPUs if available

## Next Steps

After completing Phases 1-2:

1. **Validate Reward Model**: Check correlation with human preferences
2. **Implement PPO Training**: Complete the RLAIF pipeline
3. **Multi-Round Training**: Iteratively improve model quality
4. **Production Deployment**: Deploy improved model with inference server

## File Locations

All generated files:
- **Reward Dataset**: `./datasets/wp_reward_data.json`
- **Reward Model**: `./models/wp-reward-model/`
- **Training Logs**: `./logs/reward_model_training.log`
- **Evaluation Results**: `./models/wp-reward-model/eval_results.json`

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review training logs in `./logs/`
3. Validate model files exist in expected locations
4. Ensure all dependencies are installed with correct versions

---

*RLAIF Pipeline v1.0 | WordPress SLM Enhancement | AI-Powered Code Quality*