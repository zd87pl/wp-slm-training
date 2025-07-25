# WordPress SLM Docker Image
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    vim \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip
RUN python -m pip install --upgrade pip

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install PyTorch with CUDA support
RUN pip install torch --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
RUN pip install -r requirements.txt

# Install additional packages that might fail in requirements.txt
RUN pip install vllm || echo "vLLM installation failed, continuing..."

# Copy the entire project
COPY . .

# Install the package in development mode
RUN pip install -e .

# Create directories for data and models
RUN mkdir -p data/raw data/processed data/sft data/prefs data/eval outputs

# Expose ports for inference server and WordPress
EXPOSE 8000 8080

# Default command
CMD ["/bin/bash"]