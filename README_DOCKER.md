# YOLO Tracking - Docker Edition

Docker containerization for YOLO object detection and tracking with GPU support.

## 🏗️ Architecture

```
yolo_tracking_demo/
├── Dockerfile                  # Multi-stage Docker build
├── docker-compose.yml          # Docker Compose configuration
├── .dockerignore              # Files to exclude from image
├── docker_run.sh              # Helper script for running containers
├── setup_server.sh            # Automated server setup
├── DOCKER_DEPLOYMENT.md       # Comprehensive deployment guide
├── DEPLOYMENT_QUICKSTART.md   # Quick reference guide
└── ... (project files)
```

## Quick Start (3 Steps)

### 1. Transfer to Server
```bash
scp -r yolo_deepsort_demo user@server-ip:/home/user/yolo_tracking
# or use git clone
```

### 2. Setup Server
```bash
ssh user@server-ip
cd yolo_tracking
./setup_server.sh
```

### 3. Run!
```bash
./docker_run.sh inference --source data/inputs/video.mp4 --device cuda
```

## What's Included

### Docker Image Contents
- **Base**: NVIDIA CUDA 12.6.2 + cuDNN 9
- **OS**: Ubuntu 22.04
- **Python**: 3.10
- **GPU**: Full NVIDIA GPU support
- **Dependencies**: All Python packages from requirements.txt
- **Project**: Complete YOLO tracking codebase

### Image Size
- Approximately 6-8 GB (with all dependencies)
- Optimized for fast startup and execution

## Use Cases

### 1. Production Inference
```bash
docker run --gpus all -v $(pwd)/data:/app/data yolo-tracking:latest \
    python -m src.main --source /app/data/inputs/video.mp4 --device cuda
```

### 2. Model Training
```bash
./docker_run.sh train \
    --data data/custom_dataset/data.yaml \
    --epochs 100 \
    --batch 32 \
    --device cuda
```

### 3. Batch Processing
```bash
./docker_run.sh bash
# Inside container, process multiple files
for video in /app/data/inputs/*.mp4; do
    python -m src.main --source "$video" --device cuda --save-video
done
```

### 4. Development
```bash
./docker_run.sh bash
# Interactive development environment with GPU access
```

## 🔧 Configuration

### Environment Variables
Set in `docker-compose.yml` or pass with `-e`:

```bash
CUDA_VISIBLE_DEVICES=0        # GPU selection
NVIDIA_VISIBLE_DEVICES=all    # Make all GPUs visible
```

### Volume Mounts
Default mounts (can be customized):

```yaml
volumes:
  - ./data:/app/data          # Input/output data
  - ./weights:/app/weights    # Model weights
  - ./runs:/app/runs          # Training runs
  - ./configs:/app/configs    # Configurations
```

### Resource Limits
Adjust in `docker-compose.yml`:

```yaml
shm_size: '8gb'               # Shared memory
deploy:
  resources:
    limits:
      cpus: '16'              # CPU limit
      memory: 32G             # RAM limit
```

## 🎛️ Command Reference

### Build & Setup
```bash
./docker_run.sh build          # Build Docker image
./setup_server.sh              # Setup server (one-time)
```

### Run Modes
```bash
./docker_run.sh inference [OPTIONS]   # Run tracking
./docker_run.sh train [OPTIONS]       # Train model
./docker_run.sh bash                  # Interactive shell
./docker_run.sh help                  # Show help
```

### Docker Compose
```bash
docker-compose up -d           # Start in background
docker-compose logs -f         # View logs
docker-compose exec yolo-tracking bash  # Enter container
docker-compose down            # Stop and remove
```

### Manual Docker
```bash
# Build
docker build -t yolo-tracking:latest .

# Run inference
docker run --rm --gpus all \
    -v $(pwd)/data:/app/data \
    yolo-tracking:latest \
    python -m src.main --source data/inputs/video.mp4

# Interactive
docker run -it --gpus all \
    -v $(pwd)/data:/app/data \
    yolo-tracking:latest bash
```

## 🔍 Monitoring

### GPU Monitoring
```bash
# Real-time GPU stats
watch -n 1 nvidia-smi

# Detailed monitoring
nvidia-smi dmon -s ucmt

# Inside container
docker exec yolo_tracking nvidia-smi
```

### Container Monitoring
```bash
# Resource usage
docker stats yolo_tracking

# Logs
docker logs -f yolo_tracking

# Processes
docker top yolo_tracking
```

## ⚡ Performance Tips

### For NVIDIA A100 (80GB VRAM)

1. **Large Batch Sizes**
```bash
# Training with large batches
./docker_run.sh train --data data.yaml --batch 128 --img-size 640
```

2. **Mixed Precision**
```yaml
# configs/detector.yaml
amp: true
half: false  # Use AMP instead of FP16 for training
```

3. **Multi-GPU Training**
```bash
docker run --gpus all -e CUDA_VISIBLE_DEVICES=0,1 ...
```

4. **Optimizations**
```yaml
# configs/default.yaml
cudnn_benchmark: true
torch_compile: true
deterministic: false  # Disable for speed
```

## 🐛 Troubleshooting

### GPU Not Accessible
```bash
# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.6.2-base-ubuntu22.04 nvidia-smi

# If fails:
sudo systemctl restart docker
sudo nvidia-ctk runtime configure --runtime=docker
```

### Build Fails
```bash
# Clean build cache
docker builder prune -a

# Rebuild without cache
docker build --no-cache -t yolo-tracking:latest .
```

### Container Crashes
```bash
# Check logs
docker logs yolo_tracking

# Increase shared memory
docker run --shm-size=16g ...

# Reduce batch size
# Edit configs/detector.yaml: batch_size: 8
```

### Permission Issues
```bash
# Run as current user
docker run --user $(id -u):$(id -g) ...

# Fix host permissions
sudo chown -R $USER:$USER data/ weights/ runs/
```

## 📊 Benchmarks (Expected Performance)

### NVIDIA A100 (80GB)

| Task | Model | Batch | FPS | Notes |
|------|-------|-------|-----|-------|
| Inference | YOLOv12s | 1 | ~120 | 640x640, FP32 |
| Inference | YOLOv12s | 1 | ~200 | 640x640, FP16 |
| Training | YOLOv12s | 32 | ~50 | 640x640, AMP |
| Training | YOLOv12s | 128 | ~180 | 640x640, AMP |

### NVIDIA T4 (16GB)

| Task | Model | Batch | FPS | Notes |
|------|-------|-------|-----|-------|
| Inference | YOLOv12s | 1 | ~60 | 640x640, FP32 |
| Inference | YOLOv12s | 1 | ~100 | 640x640, FP16 |
| Training | YOLOv12s | 16 | ~30 | 640x640, AMP |

## 🔒 Security Best Practices

1. **Don't run as root**
```bash
docker run --user $(id -u):$(id -g) ...
```

2. **Limit GPU access**
```bash
export CUDA_VISIBLE_DEVICES=0  # Only GPU 0
```

3. **Use secrets for sensitive data**
```bash
docker run -e API_KEY=$(cat secret.key) ...
```

4. **Regular updates**
```bash
docker pull nvidia/cuda:12.6.2-cudnn8-runtime-ubuntu22.04
docker build -t yolo-tracking:latest .
```

5. **Read-only mounts where possible**
```bash
docker run -v $(pwd)/configs:/app/configs:ro ...
```

## 📦 Registry & Distribution

### Save Image
```bash
# Save to tar
docker save yolo-tracking:latest | gzip > yolo-tracking.tar.gz

# Transfer to server
scp yolo-tracking.tar.gz user@server-ip:/home/user/

# Load on server
gunzip -c yolo-tracking.tar.gz | docker load
```

### Push to Registry
```bash
# Tag for registry
docker tag yolo-tracking:latest registry.example.com/yolo-tracking:latest

# Push
docker push registry.example.com/yolo-tracking:latest

# Pull on server
docker pull registry.example.com/yolo-tracking:latest
```

## 🔄 Updates & Maintenance

### Update Code
```bash
# Pull latest code
git pull origin main

# Rebuild image
./docker_run.sh build

# Or with Docker Compose
docker-compose build --no-cache
```

### Cleanup
```bash
# Remove old containers
docker container prune

# Remove old images
docker image prune -a

# Remove build cache
docker builder prune -a

# Full cleanup
docker system prune -a --volumes
```

## 📚 Additional Resources

- **[DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md)** - Full deployment guide
- **[DEPLOYMENT_QUICKSTART.md](DEPLOYMENT_QUICKSTART.md)** - Quick reference
- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - Training instructions
- **[setup_server.sh](setup_server.sh)** - Automated setup script
- **[docker_run.sh](docker_run.sh)** - Helper script

## 🆘 Support

### Check System Status
```bash
# NVIDIA Driver
nvidia-smi

# Docker
docker --version
docker info

# NVIDIA Docker
docker run --rm --gpus all nvidia/cuda:12.6.2-base-ubuntu22.04 nvidia-smi

# Project Image
docker images | grep yolo-tracking
```

### Common Issues

1. **"docker: command not found"** → Install Docker
2. **"nvidia-smi: command not found"** → Install NVIDIA driver
3. **"could not select device driver"** → Install nvidia-container-toolkit
4. **Permission denied** → Add user to docker group: `sudo usermod -aG docker $USER`
5. **Out of memory** → Reduce batch size or increase shared memory

## 📝 License

Same as main project.

## 🤝 Contributing

See main project README.

---

**For detailed instructions, see [DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md)**

