# Docker Deployment Guide

## Prerequisites

### On the Server (Ubuntu VM with A100)

1. **NVIDIA Driver** (Version 525.x or higher recommended for A100)
```bash
# Check if driver is installed
nvidia-smi

# If not installed, install NVIDIA driver
sudo apt-get update
sudo apt-get install -y nvidia-driver-525
sudo reboot
```

2. **Docker** (Version 20.10+)
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Verify installation
docker --version
```

3. **NVIDIA Container Toolkit**
```bash
# Add NVIDIA GPG key and repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install nvidia-container-toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Test GPU access in Docker
docker run --rm --gpus all nvidia/cuda:12.6.2-base-ubuntu22.04 nvidia-smi
```

---

## Installation

### Transfer Project to Server

**Option 1: Using Git**
```bash
# Clone the repository on the server
git clone <your-repo-url> yolo_tracking_demo
cd yolo_tracking_demo
```

**Option 2: Using SCP**
```bash
# From your local machine, transfer the project
scp -r /path/to/yolo_deepsort_demo user@server-ip:/home/user/yolo_tracking_demo

# SSH into server
ssh user@server-ip
cd yolo_tracking_demo
```

**Option 3: Using rsync**
```bash
# From your local machine (recommended for large projects)
rsync -avz --progress /path/to/yolo_deepsort_demo user@server-ip:/home/user/yolo_tracking_demo
```

---

## Building the Docker Image

### Build the Image (with Buildx)

The project uses Docker Buildx for optimized builds on Linux/amd64 (required for A100 GPU support):

```bash
# Navigate to project directory
cd yolo_tracking_demo

# Build using the shell script (uses buildx automatically)
./docker_run.sh build

# Or setup buildx first (one-time setup)
./docker_run.sh buildx-setup

# Or build manually with buildx
docker buildx build --platform linux/amd64 -t yolo-tracking:latest --load .
```

### Buildx Setup (One-Time)

If buildx is not initialized, run:

```bash
# Setup buildx builder
./docker_run.sh buildx-setup

# Or manually
docker buildx create --name multiarch --use
docker buildx inspect --bootstrap
```

### Verify the Build
```bash
# List Docker images
docker images | grep yolo-tracking

# Check image size
docker images yolo-tracking:latest --format "{{.Size}}"

# Verify platform
docker inspect yolo-tracking:latest | grep Architecture
```

### Build Options

**Standard build (amd64 for GPU):**
```bash
./docker_run.sh build
```

**Explicit platform build:**
```bash
./docker_run.sh buildx-build linux/amd64
```

**Note:** GPU support (A100) requires `linux/amd64` platform. Other platforms will not have CUDA support.

---

## Running the Container

### Using the Helper Script (Recommended)

The `docker_run.sh` script provides easy access to common operations:

```bash
# Show help
./docker_run.sh help

# Open interactive shell
./docker_run.sh bash

# Run inference/tracking
./docker_run.sh inference --source video.mp4 --show

# Run training
./docker_run.sh train --data data/dataset.yaml --epochs 100
```

### Using Docker Compose

```bash
# Start the container
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the container
docker-compose down

# Run commands in the container
docker-compose exec yolo-tracking python -m src.main --source 0
```

### Manual Docker Run Commands

**Interactive Shell:**
```bash
docker run --rm -it \
    --gpus all \
    --name yolo_tracking \
    --shm-size=8g \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/weights:/app/weights \
    -v $(pwd)/runs:/app/runs \
    -v $(pwd)/configs:/app/configs \
    yolo-tracking:latest \
    /bin/bash
```

**Run Inference:**
```bash
docker run --rm \
    --gpus all \
    --shm-size=8g \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/weights:/app/weights \
    -v $(pwd)/runs:/app/runs \
    yolo-tracking:latest \
    python -m src.main --source data/inputs/video.mp4 --device cuda
```

**Run Training:**
```bash
docker run --rm \
    --gpus all \
    --shm-size=8g \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/weights:/app/weights \
    yolo-tracking:latest \
    python train.py --data data/custom_dataset.yaml --epochs 100 --batch 32
```

---

## Usage Examples

### 1. Object Tracking on Video File

```bash
# Place your video in data/inputs/
cp /path/to/video.mp4 data/inputs/

# Run tracking
./docker_run.sh inference \
    --source data/inputs/video.mp4 \
    --device cuda \
    --img-size 640 \
    --conf 0.5 \
    --save-video
```

### 2. Real-time Tracking from Webcam

```bash
./docker_run.sh inference \
    --source 0 \
    --device cuda \
    --show
```

### 3. Training Custom Model

```bash
# Prepare your dataset in data/
# datasets/
#   ├── train/
#   ├── val/
#   └── data.yaml

./docker_run.sh train \
    --data data/custom_dataset/data.yaml \
    --epochs 100 \
    --batch 32 \
    --img-size 640 \
    --device cuda
```

### 4. Batch Processing Multiple Videos

```bash
# Create a script inside the container
./docker_run.sh bash

# Inside container:
for video in /app/data/inputs/*.mp4; do
    python -m src.main \
        --source "$video" \
        --device cuda \
        --save-video
done
```

### 5. Export Model to ONNX

```bash
./docker_run.sh bash

# Inside container:
python -c "
from src.models.yolo_detector import YOLODetector
from src.utils.config import load_config

config = load_config()
detector = YOLODetector(config['detector'], use_tracking=False)
detector.export_model(format='onnx', half=True)
"
```


## Performance Optimization for A100

### 1. Use Mixed Precision Training
```bash
# Edit configs/detector.yaml
amp: true

# Or via CLI
./docker_run.sh train --data data.yaml --amp
```

### 2. Optimize Batch Size for A100 (80GB VRAM)
```bash
# For YOLOv12s on A100, you can use large batches
./docker_run.sh train \
    --data data.yaml \
    --batch 128 \
    --img-size 640 \
    --device cuda
```

### 3. Enable TensorFloat-32 (TF32) for A100
```bash
# Inside container, add to your training script
export NVIDIA_TF32_OVERRIDE=1

# Or modify Dockerfile to add:
# ENV NVIDIA_TF32_OVERRIDE=1
```

### 4. Monitor GPU Usage
```bash
# In a separate terminal, watch GPU utilization
watch -n 1 nvidia-smi

# Or inside container
python -c "
import torch
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
"
```

---

## Troubleshooting

### Issue: GPU Not Detected in Container

**Check NVIDIA runtime:**
```bash
docker run --rm --gpus all nvidia/cuda:12.6.2-base-ubuntu22.04 nvidia-smi
```

**If it fails, check Docker daemon config:**
```bash
sudo cat /etc/docker/daemon.json

# Restart Docker
sudo systemctl restart docker
```

### Issue: Out of Memory

**Reduce batch size:**
```bash
# Edit configs/detector.yaml
training:
  batch_size: 16  # Reduce from 32 or higher
```

**Increase shared memory:**
```bash
docker run --shm-size=16g ...  # Increase from 8g
```

### Issue: Permission Denied on Volume Mounts

```bash
# Fix permissions on host
sudo chown -R $USER:$USER data/ weights/ runs/ logs/

# Or run container with user
docker run --user $(id -u):$(id -g) ...
```

### Issue: Slow Training/Inference

**Check GPU utilization:**
```bash
nvidia-smi dmon -s u -i 0
```

**Enable cuDNN benchmarking:**
```yaml
# configs/default.yaml
cudnn_benchmark: true
```

**Use FP16 inference:**
```bash
./docker_run.sh inference --source video.mp4 --half
```

---

## Monitoring and Logging

### View Container Logs
```bash
# Docker Compose
docker-compose logs -f yolo-tracking

# Docker run (in separate terminal)
docker logs -f yolo_tracking
```

### Monitor Resource Usage
```bash
# CPU/Memory
docker stats yolo_tracking

# GPU
nvidia-smi dmon -s ucm -i 0
```

### Export Logs
```bash
docker logs yolo_tracking > training.log 2>&1
```

---

## Cleanup

### Remove Containers
```bash
# Stop and remove all containers
docker-compose down

# Remove specific container
docker rm -f yolo_tracking
```

### Remove Images
```bash
# Remove project image
docker rmi yolo-tracking:latest

# Remove dangling images
docker image prune
```

### Clean Build Cache
```bash
docker builder prune -a
```