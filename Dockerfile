# Multi-platform build for YOLO tracking project
# Supports CPU, MPS (falls back to CPU in Docker), and CUDA devices
# Built with Docker Buildx for amd64 platform
# Note: CUDA base image works for both CPU and GPU modes

# Build arguments for multi-platform support (buildx)
ARG TARGETPLATFORM=linux/amd64
ARG BUILDPLATFORM

# Base image with CUDA support (Latest: CUDA 12.6.2 + cuDNN 9)
# Explicitly set platform for buildx compatibility
FROM --platform=${TARGETPLATFORM} nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04 AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    ca-certificates \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p data/inputs data/outputs weights runs logs

# Set permissions
RUN chmod +x train.py || true

# Expose any ports if needed (for monitoring/API)
EXPOSE 8000

# Default command (can be overridden)
CMD ["python", "-m", "src.main", "--help"]

