# Docker Deployment Quick Start

## 🚀 Transfer Project to Server

### Option 1: Git (Recommended)
```bash
ssh user@server-ip
git clone <your-repo-url> yolo_tracking
cd yolo_tracking
```

### Option 2: SCP
```bash
# From local machine
scp -r yolo_deepsort_demo user@server-ip:/home/user/yolo_tracking
```

### Option 3: rsync
```bash
# From local machine (best for large projects)
rsync -avz --progress yolo_deepsort_demo/ user@server-ip:/home/user/yolo_tracking/
```

---

## ⚙️ One-Command Setup on Server

```bash
# SSH into server
ssh user@server-ip
cd yolo_tracking

# Run automated setup (installs everything)
./setup_server.sh
```

This script will:
- ✅ Check/install NVIDIA drivers
- ✅ Install Docker
- ✅ Install NVIDIA Container Toolkit
- ✅ Test GPU access in Docker
- ✅ Build the Docker image
- ✅ Create required directories

**Note:** The script may require a reboot or re-login between steps. Just run it again after.

---

## Manual Setup

### 1. Install NVIDIA Driver
```bash
sudo apt-get update
sudo apt-get install -y nvidia-driver-525
sudo reboot
```

### 2. Install Docker
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
# Log out and back in
```

### 3. Install NVIDIA Container Toolkit
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 4. Build Docker Image
```bash
cd yolo_tracking
docker build -t yolo-tracking:latest .
```

---

## 🎯 Usage Examples

### Test Installation
```bash
./docker_run.sh bash
# Inside container:
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"
nvidia-smi
exit
```

### Run Tracking on Video
```bash
# Place video in data/inputs/
cp /path/to/video.mp4 data/inputs/

# Run tracking
./docker_run.sh inference \
    --source data/inputs/video.mp4 \
    --device cuda \
    --conf 0.5 \
    --save-video
```

### Train Custom Model
```bash
# Prepare dataset in data/custom_dataset/
# Structure:
#   data/custom_dataset/
#     ├── train/images/
#     ├── train/labels/
#     ├── val/images/
#     ├── val/labels/
#     └── data.yaml

# Train
./docker_run.sh train \
    --data data/custom_dataset/data.yaml \
    --epochs 100 \
    --batch 32 \
    --device cuda
```

### Real-time Webcam Tracking
```bash
./docker_run.sh inference --source 0 --device cuda --show
```

### Batch Process Multiple Videos
```bash
./docker_run.sh bash

# Inside container:
for video in /app/data/inputs/*.mp4; do
    python -m src.main --source "$video" --device cuda --save-video
done
```

---

## 📊 Monitor GPU Usage

### Real-time Monitoring
```bash
# In separate SSH session
watch -n 1 nvidia-smi
```

### Detailed Monitoring
```bash
# GPU utilization, memory, temperature
nvidia-smi dmon -s ucmt -i 0

# Container resource usage
docker stats yolo_tracking
```

---

## 🔧 Common Commands

```bash
# Build image
./docker_run.sh build

# Open shell
./docker_run.sh bash

# Run tracking
./docker_run.sh inference --source video.mp4 --device cuda

# Run training
./docker_run.sh train --data data.yaml --epochs 100

# Show help
./docker_run.sh help

# View logs
docker logs yolo_tracking

# Stop container
docker stop yolo_tracking

# Remove container
docker rm yolo_tracking
```

---

## 🐛 Quick Troubleshooting

### GPU Not Detected
```bash
# Test GPU in Docker
docker run --rm --gpus all nvidia/cuda:12.6.2-base-ubuntu22.04 nvidia-smi

# If fails, restart Docker
sudo systemctl restart docker
```

### Out of Memory
```bash
# Reduce batch size in configs/detector.yaml
training:
  batch_size: 16  # Reduce this

# Or increase shared memory
docker run --shm-size=16g ...
```

### Permission Denied
```bash
# Fix file permissions
sudo chown -R $USER:$USER data/ weights/ runs/ logs/
```

### Slow Performance
```bash
# Check GPU utilization
nvidia-smi

# Enable optimizations in configs/default.yaml
cudnn_benchmark: true
torch_compile: true
```

---

## 📈 A100 Optimization Tips

```bash
# Use large batches (A100 has 80GB VRAM)
./docker_run.sh train --data data.yaml --batch 128

# Enable mixed precision
# Edit configs/detector.yaml: amp: true

# Use FP16 for inference
./docker_run.sh inference --source video.mp4 --half

# Enable TF32 (A100 specific)
export NVIDIA_TF32_OVERRIDE=1
```

---

## 📚 Documentation Files

- `DOCKER_DEPLOYMENT.md` - Comprehensive deployment guide
- `TRAINING_GUIDE.md` - Model training instructions
- `README.md` - Project overview
- `docker_run.sh` - Helper script for common operations
- `setup_server.sh` - Automated server setup script

---

## 🆘 Need Help?

1. Check logs: `docker logs yolo_tracking`
2. Test GPU: `nvidia-smi`
3. Verify Docker: `docker --version`
4. See full guide: `DOCKER_DEPLOYMENT.md`

---

## ✅ Verify Installation

```bash
# Check all components
echo "=== NVIDIA Driver ==="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

echo "=== Docker ==="
docker --version

echo "=== GPU in Docker ==="
docker run --rm --gpus all nvidia/cuda:12.6.2-base-ubuntu22.04 nvidia-smi

echo "=== Project Image ==="
docker images | grep yolo-tracking

echo "=== Test Run ==="
./docker_run.sh bash -c "python --version && python -c 'import torch; print(f\"PyTorch: {torch.__version__}\"); print(f\"CUDA Available: {torch.cuda.is_available()}\")'"
```

---

**Ready to go! 🎉**

