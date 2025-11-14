#!/bin/bash
# Server setup script for Ubuntu VM with NVIDIA GPU
# Run this script on the server after transferring the project

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}YOLO Tracking Server Setup${NC}"
echo -e "${BLUE}Ubuntu + NVIDIA GPU + Docker${NC}"
echo -e "${BLUE}================================================${NC}\n"

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo -e "${RED}This script should NOT be run as root${NC}"
   echo "Run as regular user: ./setup_server.sh"
   exit 1
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Step 1: Check NVIDIA Driver
echo -e "\n${YELLOW}[1/5] Checking NVIDIA Driver...${NC}"
if command_exists nvidia-smi; then
    echo -e "${GREEN}✓ NVIDIA driver found${NC}"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    echo -e "${RED}✗ NVIDIA driver not found${NC}"
    echo "Installing NVIDIA driver (requires sudo)..."
    sudo apt-get update
    sudo apt-get install -y nvidia-driver-525
    echo -e "${YELLOW}⚠ System reboot required. Run 'sudo reboot' and then run this script again.${NC}"
    exit 0
fi

# Step 2: Install Docker
echo -e "\n${YELLOW}[2/5] Checking Docker...${NC}"
if command_exists docker; then
    echo -e "${GREEN}✓ Docker found: $(docker --version)${NC}"
else
    echo -e "${YELLOW}Installing Docker...${NC}"
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    rm get-docker.sh
    
    # Add user to docker group
    sudo usermod -aG docker $USER
    echo -e "${GREEN}✓ Docker installed${NC}"
    echo -e "${YELLOW}⚠ You need to log out and back in for docker group changes to take effect${NC}"
    echo "After logging back in, run this script again."
    exit 0
fi

# Check if user is in docker group
if ! groups | grep -q docker; then
    echo -e "${YELLOW}Adding user to docker group...${NC}"
    sudo usermod -aG docker $USER
    echo -e "${YELLOW}⚠ Log out and back in, then run this script again${NC}"
    exit 0
fi

# Step 3: Install NVIDIA Container Toolkit
echo -e "\n${YELLOW}[3/5] Checking NVIDIA Container Toolkit...${NC}"
if command_exists nvidia-ctk; then
    echo -e "${GREEN}✓ NVIDIA Container Toolkit found${NC}"
else
    echo -e "${YELLOW}Installing NVIDIA Container Toolkit...${NC}"
    
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
        sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    
    curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    
    # Configure Docker runtime
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    
    echo -e "${GREEN}✓ NVIDIA Container Toolkit installed${NC}"
fi

# Step 4: Test GPU Access in Docker
echo -e "\n${YELLOW}[4/5] Testing GPU access in Docker...${NC}"
if docker run --rm --gpus all nvidia/cuda:12.6.2-base-ubuntu22.04 nvidia-smi &>/dev/null; then
    echo -e "${GREEN}✓ GPU accessible in Docker${NC}"
    docker run --rm --gpus all nvidia/cuda:12.6.2-base-ubuntu22.04 nvidia-smi --query-gpu=name --format=csv,noheader
else
    echo -e "${RED}✗ GPU not accessible in Docker${NC}"
    echo "Troubleshooting..."
    
    # Check daemon.json
    if [ ! -f /etc/docker/daemon.json ]; then
        echo "Creating /etc/docker/daemon.json"
        echo '{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}' | sudo tee /etc/docker/daemon.json
        sudo systemctl restart docker
        sleep 3
    fi
    
    # Try again
    if docker run --rm --gpus all nvidia/cuda:12.6.2-base-ubuntu22.04 nvidia-smi &>/dev/null; then
        echo -e "${GREEN}✓ GPU now accessible${NC}"
    else
        echo -e "${RED}✗ Still cannot access GPU. Please check Docker and NVIDIA setup manually.${NC}"
        exit 1
    fi
fi

# Step 5: Build Docker Image
echo -e "\n${YELLOW}[5/5] Building YOLO Tracking Docker Image...${NC}"
if [ -f "Dockerfile" ]; then
    echo "Building image (this may take several minutes)..."
    docker build -t yolo-tracking:latest . || {
        echo -e "${RED}✗ Docker build failed${NC}"
        exit 1
    }
    echo -e "${GREEN}✓ Docker image built successfully${NC}"
else
    echo -e "${RED}✗ Dockerfile not found in current directory${NC}"
    exit 1
fi

# Create required directories
echo -e "\n${YELLOW}Creating project directories...${NC}"
mkdir -p data/inputs data/outputs weights runs logs
echo -e "${GREEN}✓ Directories created${NC}"

# Make scripts executable
chmod +x docker_run.sh 2>/dev/null || true
chmod +x train.py 2>/dev/null || true

# Summary
echo -e "\n${GREEN}================================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}================================================${NC}\n"

echo "System Information:"
echo "  • GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "  • Driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
echo "  • Docker: $(docker --version | cut -d' ' -f3 | tr -d ',')"
echo "  • CUDA: 12.6.2 (in container)"

echo -e "\n${BLUE}Quick Start:${NC}"
echo "  • Test installation:"
echo "    ./docker_run.sh bash"
echo ""
echo "  • Run tracking on video:"
echo "    ./docker_run.sh inference --source data/inputs/video.mp4 --device cuda"
echo ""
echo "  • Train custom model:"
echo "    ./docker_run.sh train --data data/dataset.yaml --epochs 100"
echo ""
echo "  • View help:"
echo "    ./docker_run.sh help"

echo -e "\n${YELLOW}Documentation:${NC}"
echo "  • See DOCKER_DEPLOYMENT.md for detailed usage"
echo "  • See TRAINING_GUIDE.md for training instructions"

echo -e "\n${GREEN}Happy tracking! 🚀${NC}\n"

