# WordPress SLM Setup Guide

This guide explains how to set up the WordPress SLM development environment.

## Environment Setup Options

You have three options for setting up the environment:

### Option 1: Conda (Recommended)

Conda is recommended because it handles CUDA dependencies automatically.

```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate wp-slm

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Option 2: pip with venv

If you prefer pip, ensure you have CUDA installed on your system first.

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch based on your CUDA version
# For CUDA 11.8:
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt

# For development (includes all extras):
pip install -e ".[all]"
```

### Option 3: Docker (Isolation)

For complete isolation, use the provided Docker setup:

```bash
# Build the Docker image
docker build -t wp-slm .

# Run with GPU support
docker run --gpus all -it wp-slm
```

## GPU Setup Verification

After installation, verify your GPU setup:

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

Expected output for RTX 4090:
```
PyTorch version: 2.1.0+cu118
CUDA available: True
CUDA version: 11.8
GPU: NVIDIA GeForce RTX 4090
```

## vLLM Installation (For Inference)

vLLM requires specific CUDA versions and may need separate installation:

```bash
# Install vLLM (requires CUDA 11.8 or 12.1)
pip install vllm

# If you encounter issues, try:
pip install vllm --no-build-isolation
```

## Common Issues

### 1. CUDA Version Mismatch
If you get CUDA errors, ensure your PyTorch CUDA version matches your system CUDA:
```bash
nvidia-smi  # Check system CUDA version
python -c "import torch; print(torch.version.cuda)"  # Check PyTorch CUDA
```

### 2. Out of Memory Errors
The 4-bit quantization should work on RTX 4090 (24GB). If you still get OOM:
- Reduce batch size in training configs
- Enable gradient checkpointing (already enabled by default)
- Use smaller sequence lengths

### 3. bitsandbytes Installation
If bitsandbytes fails to install:
```bash
# Linux
pip install bitsandbytes

# Windows (use pre-compiled wheel)
pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.0-py3-none-win_amd64.whl
```

### 4. Missing System Dependencies
Some packages require system libraries:
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y python3-dev build-essential

# For Docker support
sudo apt-get install -y docker.io docker-compose
```

## Development Workflow

1. **Always activate the environment first:**
   ```bash
   conda activate wp-slm  # or source venv/bin/activate
   ```

2. **Run tests to verify setup:**
   ```bash
   python tests/run_tests.py
   # or
   make test
   ```

3. **Check data pipeline:**
   ```bash
   wp-slm-validate --data-dir data
   ```

## Next Steps

Once your environment is set up:

1. Start data collection: `make data`
2. Begin training: `make sft`
3. Launch inference server: `make serve`
4. Install WordPress plugin for testing

See the main README.md for detailed usage instructions.