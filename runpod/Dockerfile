# RunPod-optimized Dockerfile for WordPress SLM
# Based on RunPod's PyTorch template for better compatibility
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0"

# Install system dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git \
        wget \
        curl \
        vim \
        tmux \
        htop \
        build-essential \
        libssl-dev \
        libffi-dev \
        python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /workspace

# Copy project files
COPY requirements.txt .

# Install Python dependencies
# PyTorch is already installed in the base image
RUN pip install -r requirements.txt

# Install vLLM for inference (may fail on some GPUs, that's ok)
RUN pip install vllm || echo "vLLM installation failed, continuing..."

# Install flash-attention for better performance
RUN pip install flash-attn --no-build-isolation || echo "Flash attention not supported"

# Copy the entire project
COPY . /workspace/wp-slm/

# Install the package
RUN cd /workspace/wp-slm && pip install -e .

# Create necessary directories
RUN mkdir -p /workspace/data /workspace/models /workspace/outputs

# RunPod handler script
COPY runpod/handler.py /workspace/handler.py

# Expose ports
EXPOSE 8000 8888

# RunPod expects a handler.py by default
# But we'll make it flexible to run different commands
CMD ["python", "-u", "/workspace/handler.py"]