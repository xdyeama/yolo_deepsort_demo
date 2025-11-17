#!/bin/bash
# Script to run YOLO tracking in Docker container
# Supports CPU, MPS (Apple Silicon), and CUDA devices

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}YOLO Tracking Docker Runner${NC}"
echo -e "${GREEN}========================================${NC}"

# Default values
IMAGE_NAME="xdyeama/yolo-tracking"
CONTAINER_NAME="yolo_tracking_demo"
USE_GPU=false
DEVICE_TYPE="auto"

# Function to detect device type from arguments
detect_device_type() {
    local args=("$@")
    DEVICE_TYPE="auto"  # Reset to default
    
    for i in "${!args[@]}"; do
        if [[ "${args[i]}" == "--device" ]] && [[ -n "${args[i+1]}" ]]; then
            DEVICE_TYPE="${args[i+1]}"
            break
        fi
    done
    
    # If device is cuda, we need GPU
    if [[ "$DEVICE_TYPE" == "cuda" ]]; then
        USE_GPU=true
    elif [[ "$DEVICE_TYPE" == "cpu" ]]; then
        USE_GPU=false
    elif [[ "$DEVICE_TYPE" == "mps" ]]; then
        USE_GPU=false
        echo -e "${YELLOW}Note: MPS (Metal) is not available in Docker containers.${NC}"
        echo -e "${YELLOW}MPS requires native macOS. Will use CPU instead.${NC}"
    elif [[ "$DEVICE_TYPE" == "auto" ]]; then
        # Auto-detect: check if CUDA is available
        if command -v nvidia-smi &> /dev/null; then
            USE_GPU=true
            DEVICE_TYPE="cuda"
        else
            USE_GPU=false
            DEVICE_TYPE="cpu"
        fi
    fi
}

# Function to check NVIDIA GPU (only when CUDA is requested)
check_nvidia_gpu() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo -e "${RED}Error: nvidia-smi not found. Make sure NVIDIA drivers are installed.${NC}"
        echo -e "${YELLOW}For CPU mode, use --device cpu${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ NVIDIA GPU detected:${NC}"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
}

# Function to check GPU access in Docker (only for CUDA)
check_gpu_access() {
    if ! docker run --rm --gpus all nvidia/cuda:12.6.2-base-ubuntu22.04 nvidia-smi &>/dev/null; then
        echo -e "${RED}✗ GPU NOT accessible in Docker${NC}"
        echo -e "\n${YELLOW}This usually means NVIDIA Container Toolkit is not installed or configured.${NC}\n"
        echo "To fix this, install NVIDIA Container Toolkit:"
        echo ""
        echo "1. Add NVIDIA GPG key and repository:"
        echo -e "   ${GREEN}distribution=\$(. /etc/os-release;echo \$ID\$VERSION_ID)${NC}"
        echo -e "   ${GREEN}curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg${NC}"
        echo -e "   ${GREEN}curl -s -L https://nvidia.github.io/libnvidia-container/\$distribution/libnvidia-container.list | \\${NC}"
        echo -e "   ${GREEN}    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \\${NC}"
        echo -e "   ${GREEN}    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list${NC}"
        echo ""
        echo "2. Install and configure:"
        echo -e "   ${GREEN}sudo apt-get update${NC}"
        echo -e "   ${GREEN}sudo apt-get install -y nvidia-container-toolkit${NC}"
        echo -e "   ${GREEN}sudo nvidia-ctk runtime configure --runtime=docker${NC}"
        echo -e "   ${GREEN}sudo systemctl restart docker${NC}"
        echo ""
        echo "3. Test again:"
        echo -e "   ${GREEN}docker run --rm --gpus all nvidia/cuda:12.6.2-base-ubuntu22.04 nvidia-smi${NC}"
        echo ""
        echo "Or use CPU mode:"
        echo -e "   ${GREEN}./docker_run.sh train --data data.yaml --device cpu${NC}"
        return 1
    fi
    return 0
}

# Function to build Docker run command with appropriate GPU flags
build_docker_run_cmd() {
    local base_cmd="docker run --rm -it"
    
    # Add GPU support only for CUDA
    if [[ "$USE_GPU" == true ]]; then
        base_cmd="$base_cmd --gpus all"
    fi
    
    # Add container name
    base_cmd="$base_cmd --name $CONTAINER_NAME"
    
    # Add shared memory
    base_cmd="$base_cmd --shm-size=8g"
    
    # Add volume mounts
    base_cmd="$base_cmd -v $(pwd)/data:/app/data"
    base_cmd="$base_cmd -v $(pwd)/weights:/app/weights"
    base_cmd="$base_cmd -v $(pwd)/runs:/app/runs"
    base_cmd="$base_cmd -v $(pwd)/configs:/app/configs"
    
    # Add image name
    base_cmd="$base_cmd $IMAGE_NAME"
    
    echo "$base_cmd"
}

# Function to check if image exists
check_image_exists() {
    # Check if image exists locally (with or without tag)
    local image_found=false
    if [[ "$IMAGE_NAME" == *":"* ]]; then
        # Image has tag
        if docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "^${IMAGE_NAME}$"; then
            image_found=true
        fi
    else
        # Image without tag, check for 'latest' or any tag
        if docker images --format "{{.Repository}}" | grep -q "^${IMAGE_NAME}$"; then
            image_found=true
        fi
    fi
    
    if [ "$image_found" = false ]; then
        echo -e "\n${RED}✗ Error: Docker image '${IMAGE_NAME}' not found locally${NC}"
        echo -e "${YELLOW}Attempting to pull from Docker Hub...${NC}\n"
        
        # Try to pull the image
        if docker pull "$IMAGE_NAME" 2>/dev/null; then
            echo -e "${GREEN}✓ Image pulled successfully${NC}"
            return 0
        else
            echo -e "${RED}✗ Failed to pull image from Docker Hub${NC}"
            echo -e "${YELLOW}The image needs to be built locally.${NC}\n"
            echo "To build the image, run:"
            echo -e "  ${GREEN}./docker_run.sh build${NC}"
            echo ""
            echo "Or manually:"
            echo -e "  ${GREEN}docker buildx build --platform linux/amd64 -t ${IMAGE_NAME} --load .${NC}"
            echo ""
            echo "For automated setup (installs everything including building the image):"
            echo -e "  ${GREEN}./setup_server.sh${NC}"
            exit 1
        fi
    fi
}

# Parse command line arguments
MODE=${1:-"inference"}
shift

case $MODE in
    inference|infer|track)
        check_image_exists
        detect_device_type "$@"
        
        if [[ "$USE_GPU" == true ]]; then
            check_nvidia_gpu
            check_gpu_access || exit 1
        else
            echo -e "${BLUE}Device mode: ${DEVICE_TYPE}${NC}"
            if [[ "$DEVICE_TYPE" == "mps" ]]; then
                echo -e "${YELLOW}Note: MPS (Metal) is not available in Docker containers.${NC}"
                echo -e "${YELLOW}MPS requires native macOS. Using CPU instead.${NC}"
            fi
        fi
        
        echo -e "\n${GREEN}Running inference/tracking...${NC}"
        DOCKER_CMD=$(build_docker_run_cmd)
        eval "$DOCKER_CMD python -m src.main" "$@"
        ;;
    
    train)
        check_image_exists
        detect_device_type "$@"
        
        if [[ "$USE_GPU" == true ]]; then
            check_nvidia_gpu
            check_gpu_access || exit 1
        else
            echo -e "${BLUE}Device mode: ${DEVICE_TYPE}${NC}"
            if [[ "$DEVICE_TYPE" == "mps" ]]; then
                echo -e "${YELLOW}Note: MPS (Metal) is not available in Docker containers.${NC}"
                echo -e "${YELLOW}MPS requires native macOS. Using CPU instead.${NC}"
            fi
            echo -e "${YELLOW}Training on CPU will be slower. Consider using --device cuda for GPU acceleration.${NC}"
        fi
        
        echo -e "\n${GREEN}Running training...${NC}"
        DOCKER_CMD=$(build_docker_run_cmd)
        eval "$DOCKER_CMD python train.py" "$@"
        ;;
    
    bash|shell)
        check_image_exists
        detect_device_type "$@"
        
        if [[ "$USE_GPU" == true ]]; then
            check_nvidia_gpu
            check_gpu_access || exit 1
        else
            echo -e "${BLUE}Device mode: ${DEVICE_TYPE}${NC}"
        fi
        
        echo -e "\n${GREEN}Opening interactive shell...${NC}"
        DOCKER_CMD=$(build_docker_run_cmd)
        eval "$DOCKER_CMD /bin/bash"
        ;;
    
    test-gpu)
        echo -e "\n${GREEN}Testing GPU access in Docker...${NC}"
        if check_gpu_access; then
            echo -e "\n${GREEN}✓ GPU test successful!${NC}"
            echo "GPU information from container:"
            docker run --rm --gpus all nvidia/cuda:12.6.2-base-ubuntu22.04 nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
        fi
        ;;
    
    build)
        echo -e "\n${GREEN}Building Docker image with buildx (amd64)...${NC}"
        
        # Check if buildx is available
        if ! docker buildx version &>/dev/null; then
            echo -e "${YELLOW}⚠ buildx not found, using standard docker build${NC}"
            docker build -t $IMAGE_NAME .
        else
            # Initialize buildx if needed
            if ! docker buildx inspect multiarch &>/dev/null 2>&1; then
                echo -e "${YELLOW}Creating buildx builder...${NC}"
                docker buildx create --name multiarch --use 2>/dev/null || docker buildx use default
            fi
            
            # Build with buildx for amd64 platform
            docker buildx build \
                --platform linux/amd64 \
                --tag $IMAGE_NAME \
                --load \
                .
        fi
        echo -e "${GREEN}✓ Image built successfully: $IMAGE_NAME${NC}"
        ;;
    
    buildx-setup)
        echo -e "\n${GREEN}Setting up Docker buildx...${NC}"
        if ! docker buildx version &>/dev/null; then
            echo -e "${RED}Error: Docker buildx is not available${NC}"
            echo "Install buildx or update Docker to a version that includes buildx"
            exit 1
        fi
        
        # Create and use buildx builder
        docker buildx create --name multiarch --use 2>/dev/null || {
            echo -e "${YELLOW}Builder 'multiarch' already exists, using it...${NC}"
            docker buildx use multiarch
        }
        
        echo -e "${GREEN}✓ Buildx builder ready${NC}"
        docker buildx inspect --bootstrap
        ;;
    
    buildx-build)
        PLATFORM=${1:-"linux/amd64"}
        shift
        
        echo -e "\n${GREEN}Building multi-platform image for: $PLATFORM${NC}"
        
        if ! docker buildx version &>/dev/null; then
            echo -e "${RED}Error: Docker buildx is not available${NC}"
            exit 1
        fi
        
        # Ensure buildx builder exists
        if ! docker buildx inspect multiarch &>/dev/null 2>&1; then
            echo -e "${YELLOW}Creating buildx builder...${NC}"
            docker buildx create --name multiarch --use
        else
            docker buildx use multiarch
        fi
        
        # Build with specified platform
        docker buildx build \
            --platform $PLATFORM \
            --tag $IMAGE_NAME \
            --load \
            "$@"
        
        echo -e "${GREEN}✓ Image built successfully: $IMAGE_NAME${NC}"
        ;;
    
    help|--help|-h)
        echo ""
        echo "Usage: ./docker_run.sh [MODE] [OPTIONS]"
        echo ""
        echo "Modes:"
        echo "  inference, infer, track  Run object tracking (default)"
        echo "  train                    Run model training"
        echo "  bash, shell              Open interactive bash shell"
        echo "  build                    Build Docker image with buildx (amd64)"
        echo "  buildx-setup             Setup Docker buildx builder"
        echo "  buildx-build [PLATFORM]  Build for specific platform (default: linux/amd64)"
        echo "  test-gpu                 Test GPU access in Docker"
        echo "  help                     Show this help message"
        echo ""
        echo "Examples:"
        echo "  ./docker_run.sh test-gpu                 # Test GPU access"
        echo "  ./docker_run.sh build                    # Build for amd64"
        echo "  ./docker_run.sh buildx-setup             # Initialize buildx"
        echo "  ./docker_run.sh buildx-build linux/amd64 # Explicit platform build"
        echo "  ./docker_run.sh inference --source video.mp4 --device cpu"
        echo "  ./docker_run.sh inference --source video.mp4 --device cuda"
        echo "  ./docker_run.sh train --data data.yaml --epochs 10 --device cpu"
        echo "  ./docker_run.sh train --data data.yaml --epochs 100 --device cuda"
        echo "  ./docker_run.sh bash"
        echo ""
        echo "Device options:"
        echo "  --device cpu   Use CPU (no GPU required, slower)"
        echo "  --device cuda  Use CUDA GPU (requires NVIDIA GPU and drivers)"
        echo "  --device mps   Use MPS (not available in Docker, falls back to CPU)"
        echo "  --device auto  Auto-detect (default, prefers CUDA if available)"
        echo ""
        echo "Note: GPU support requires linux/amd64 platform and NVIDIA Container Toolkit"
        echo ""
        ;;
    
    *)
        echo -e "${RED}Unknown mode: $MODE${NC}"
        echo "Run './docker_run.sh help' for usage information"
        exit 1
        ;;
esac

