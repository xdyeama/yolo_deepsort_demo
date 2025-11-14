# Docker Deployment Guide

This guide explains how to build and run the YOLO tracking project in Docker containers on an Ubuntu VM with NVIDIA A100 GPU.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Building the Docker Image](#building-the-docker-image)
- [Running the Container](#running-the-container)
- [Usage Examples](#usage-examples)
- [Troubleshooting](#troubleshooting)

---

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

### Build the Image

```bash
# Navigate to project directory
cd yolo_tracking_demo

# Build using the shell script
./docker_run.sh build

# Or build manually
docker build -t yolo-tracking:latest .
```

### Verify the Build
```bash
# List Docker images
docker images | grep yolo-tracking

# Check image size
docker images yolo-tracking:latest --format "{{.Size}}"
```

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

---

## Volume Mounts

The Docker setup uses the following volume mounts:

| Host Directory | Container Directory | Purpose |
|---------------|---------------------|---------|
| `./data` | `/app/data` | Input/output data |
| `./weights` | `/app/weights` | Model weights |
| `./runs` | `/app/runs` | Training/inference runs |
| `./logs` | `/app/logs` | Application logs |
| `./configs` | `/app/configs` | Configuration files |

---

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
# Should contain:
# {
#     "runtimes": {
#         "nvidia": {
#             "path": "nvidia-container-runtime",
#             "runtimeArgs": []
#         }
#     },
#     "default-runtime": "nvidia"
# }

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

---

## Security Considerations

1. **Don't run as root** (use `--user` flag if needed)
2. **Limit GPU access** (use `CUDA_VISIBLE_DEVICES=0` to limit to specific GPU)
3. **Use secrets** for any API keys (don't bake into image)
4. **Update base image regularly** for security patches

---

## Additional Resources

- [NVIDIA Container Toolkit Documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html)
- [Docker GPU Support](https://docs.docker.com/config/containers/resource_constraints/#gpu)
- [YOLO Ultralytics Documentation](https://docs.ultralytics.com/)
- [A100 Performance Guide](https://www.nvidia.com/en-us/data-center/a100/)

---

## Support

For issues or questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review Docker and NVIDIA logs
3. Verify GPU drivers and CUDA compatibility

