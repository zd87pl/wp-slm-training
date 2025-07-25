# WordPress SLM Dataset Scripts

This directory contains automated scripts for building large-scale WordPress training datasets.

## Scripts

### `automated_wp_dataset.py`
- **Purpose**: Generates 10K+ high-quality WordPress training examples
- **Features**:
  - Creates variations of core WordPress code examples
  - Covers themes, plugins, security, AJAX, custom post types, etc.
  - Automatically splits into train/validation sets (80/20)
  - Generates linguistically diverse prompts
- **Usage**: `python scripts/automated_wp_dataset.py`
- **Output**: 
  - `data/sft/massive_train.jsonl` (8K examples)
  - `data/sft/massive_val.jsonl` (2K examples)

### `build_massive_dataset.sh`
- **Purpose**: One-click solution for dataset generation + training
- **Features**:
  - Generates 10K+ examples using `automated_wp_dataset.py`
  - Creates enhanced training configuration if needed
  - Runs training with massive dataset
  - Provides progress updates and error handling
- **Usage**: `./scripts/build_massive_dataset.sh`
- **Requirements**: Must be run from project root directory

## Quick Start

### Generate Dataset Only
```bash
python scripts/automated_wp_dataset.py
```

### Complete Pipeline (Dataset + Training)
```bash
./scripts/build_massive_dataset.sh
```

### Manual Training with Generated Dataset
```bash
python training/sft_train.py \
  --config training/config/enhanced_training.yaml \
  --train_file data/sft/massive_train.jsonl \
  --eval_file data/sft/massive_val.jsonl
```

## Dataset Quality

The generated datasets include:

- **WordPress Core Functions**: `wp_enqueue_script`, `add_action`, `register_post_type`, etc.
- **Security Practices**: Nonce validation, input sanitization, capability checks
- **AJAX Handling**: Secure AJAX requests with proper validation
- **Custom Post Types**: Complete implementation examples
- **Meta Boxes**: Custom fields with validation and saving
- **Widgets**: Custom widget development
- **Shortcodes**: Attribute handling and security
- **Theme Development**: Best practices and patterns

## Performance Expectations

With 10K+ examples vs standard ~10 examples:

- **Training Loss**: Significant improvement (expect <0.60 vs 0.74+)
- **WordPress Knowledge**: Much more comprehensive
- **Code Quality**: More accurate and detailed
- **Pattern Recognition**: Better understanding of WordPress conventions

## Configuration

The scripts use `training/config/enhanced_training.yaml` with optimized settings:

- **3 epochs** for thorough learning
- **Rank 16 LoRA** for higher capacity
- **1024 max sequence length** for complex code
- **Cosine learning rate schedule** for better convergence

## Troubleshooting

### Common Issues

1. **Permission Errors**:
   ```bash
   chmod +x scripts/*.sh
   ```

2. **Directory Errors**:
   ```bash
   # Must run from project root
   cd /path/to/wp-slm-development
   ./scripts/build_massive_dataset.sh
   ```

3. **Memory Issues**:
   - Reduce `per_device_train_batch_size` in config
   - Ensure sufficient GPU memory (8GB+ recommended)

4. **Dataset Generation Fails**:
   ```bash
   # Check Python dependencies
   pip install -r requirements.txt
   
   # Run manually
   python scripts/automated_wp_dataset.py
   ```

## Extending the Dataset

To add more examples or topics:

1. Edit `automated_wp_dataset.py`
2. Add new examples to `get_core_wp_examples()`
3. Adjust `target_size` parameter for larger datasets
4. Add new variation patterns in `get_prompt_variations()`

Example:
```python
{
    "prompt": "How do I create a WordPress REST API endpoint?",
    "completion": "To create a custom REST API endpoint...[detailed code]"
}
```

## Integration with RunPod

For RunPod deployment, use the enhanced workflow:

```bash
# On RunPod instance
cd /workspace/wp-slm
./scripts/build_massive_dataset.sh
```

The scripts automatically handle RunPod-specific paths and configurations.