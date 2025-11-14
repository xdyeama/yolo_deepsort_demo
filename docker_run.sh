#!/bin/bash
# Script to run YOLO tracking in Docker container

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}YOLO Tracking Docker Runner${NC}"
echo -e "${GREEN}========================================${NC}"

# Check if nvidia-docker is available
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}Error: nvidia-smi not found. Make sure NVIDIA drivers are installed.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ NVIDIA GPU detected:${NC}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# Default values
IMAGE_NAME="yolo-tracking:latest"
CONTAINER_NAME="yolo_tracking_demo"

# Parse command line arguments
MODE=${1:-"inference"}
shift

case $MODE in
    inference|infer|track)
        echo -e "\n${GREEN}Running inference/tracking...${NC}"
        docker run --rm -it \
            --gpus all \
            --name $CONTAINER_NAME \
            --shm-size=8g \
            -v $(pwd)/data:/app/data \
            -v $(pwd)/weights:/app/weights \
            -v $(pwd)/runs:/app/runs \
            -v $(pwd)/configs:/app/configs \
            $IMAGE_NAME \
            python -m src.main "$@"
        ;;
    
    train)
        echo -e "\n${GREEN}Running training...${NC}"
        docker run --rm -it \
            --gpus all \
            --name $CONTAINER_NAME \
            --shm-size=8g \
            -v $(pwd)/data:/app/data \
            -v $(pwd)/weights:/app/weights \
            -v $(pwd)/configs:/app/configs \
            $IMAGE_NAME \
            python train.py "$@"
        ;;
    
    bash|shell)
        echo -e "\n${GREEN}Opening interactive shell...${NC}"
        docker run --rm -it \
            --gpus all \
            --name $CONTAINER_NAME \
            --shm-size=8g \
            -v $(pwd)/data:/app/data \
            -v $(pwd)/weights:/app/weights \
            -v $(pwd)/runs:/app/runs \
            -v $(pwd)/configs:/app/configs \
            $IMAGE_NAME \
            /bin/bash
        ;;
    
    build)
        echo -e "\n${GREEN}Building Docker image...${NC}"
        docker build -t $IMAGE_NAME .
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
        echo "  build                    Build Docker image"
        echo "  help                     Show this help message"
        echo ""
        echo "Examples:"
        echo "  ./docker_run.sh build"
        echo "  ./docker_run.sh inference --source video.mp4 --show"
        echo "  ./docker_run.sh train --data data/custom_dataset.yaml --epochs 100"
        echo "  ./docker_run.sh bash"
        echo ""
        ;;
    
    *)
        echo -e "${RED}Unknown mode: $MODE${NC}"
        echo "Run './docker_run.sh help' for usage information"
        exit 1
        ;;
esac

