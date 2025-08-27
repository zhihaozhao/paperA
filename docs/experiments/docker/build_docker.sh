#!/bin/bash
# Build Docker images for WiFi CSI HAR experiments
# Author: Claude 4.1
# Date: December 2024

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="wifi-csi-har"
VERSION=${1:-latest}
DOCKERFILE="Dockerfile_claude4.1"
BUILD_ARGS=""

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check Docker installation
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    print_status "Docker version: $(docker --version)"
}

# Check NVIDIA Docker for GPU support
check_nvidia_docker() {
    if docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
        print_status "NVIDIA Docker runtime detected. GPU support enabled."
        return 0
    else
        print_warning "NVIDIA Docker runtime not detected. Building CPU-only version."
        return 1
    fi
}

# Build Docker image
build_image() {
    local dockerfile=$1
    local tag=$2
    local gpu_support=$3
    
    print_status "Building Docker image: ${IMAGE_NAME}:${tag}"
    
    # Enable BuildKit for better caching
    export DOCKER_BUILDKIT=1
    
    # Set build arguments based on GPU support
    if [ "$gpu_support" = true ]; then
        BUILD_ARGS="--build-arg BASE_IMAGE=nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04"
    else
        BUILD_ARGS="--build-arg BASE_IMAGE=ubuntu:22.04"
    fi
    
    # Build the image
    docker build \
        ${BUILD_ARGS} \
        --cache-from ${IMAGE_NAME}:${tag} \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        -t ${IMAGE_NAME}:${tag} \
        -f ${dockerfile} \
        .. || {
            print_error "Docker build failed"
            exit 1
        }
    
    print_status "Successfully built ${IMAGE_NAME}:${tag}"
}

# Build lightweight image for edge deployment
build_lightweight() {
    print_status "Building lightweight image for edge deployment"
    
    cat > Dockerfile.lightweight << 'EOF'
FROM python:3.10-slim

WORKDIR /workspace

# Install minimal dependencies
RUN pip install --no-cache-dir \
    torch==2.0.1+cpu -f https://download.pytorch.org/whl/torch_stable.html \
    numpy scipy scikit-learn tqdm

# Copy only essential files
COPY exp4_mamba_efficiency/ ./exp4_mamba_efficiency/
COPY unified_experiment_runner_claude4.1.py .

ENTRYPOINT ["python3"]
EOF

    docker build \
        -t ${IMAGE_NAME}:lightweight \
        -f Dockerfile.lightweight \
        .. || {
            print_error "Lightweight build failed"
            return 1
        }
    
    print_status "Successfully built ${IMAGE_NAME}:lightweight"
    
    # Clean up temporary Dockerfile
    rm -f Dockerfile.lightweight
}

# Tag images
tag_images() {
    local version=$1
    
    print_status "Tagging images..."
    
    # Tag with version
    docker tag ${IMAGE_NAME}:latest ${IMAGE_NAME}:${version}
    
    # Tag with date
    docker tag ${IMAGE_NAME}:latest ${IMAGE_NAME}:$(date +%Y%m%d)
    
    print_status "Images tagged successfully"
}

# Show image information
show_info() {
    print_status "Image information:"
    docker images | grep ${IMAGE_NAME}
    
    echo ""
    print_status "Image size details:"
    docker image inspect ${IMAGE_NAME}:latest --format='Size: {{.Size}} bytes'
    
    echo ""
    print_status "To run the container:"
    echo "  CPU only:  docker run -it --rm -v \$(pwd):/workspace ${IMAGE_NAME}:latest"
    echo "  With GPU:  docker run --gpus all -it --rm -v \$(pwd):/workspace ${IMAGE_NAME}:latest"
}

# Clean up old images
cleanup() {
    print_warning "Cleaning up old images..."
    docker image prune -f
    print_status "Cleanup complete"
}

# Main execution
main() {
    print_status "Starting Docker build process..."
    
    # Check prerequisites
    check_docker
    
    # Check GPU support
    gpu_support=false
    if check_nvidia_docker; then
        gpu_support=true
    fi
    
    # Build main image
    build_image ${DOCKERFILE} ${VERSION} ${gpu_support}
    
    # Build lightweight image
    if [ "$2" = "--with-lightweight" ]; then
        build_lightweight
    fi
    
    # Tag images
    if [ "${VERSION}" != "latest" ]; then
        tag_images ${VERSION}
    fi
    
    # Clean up if requested
    if [ "$2" = "--cleanup" ] || [ "$3" = "--cleanup" ]; then
        cleanup
    fi
    
    # Show information
    show_info
    
    print_status "Build process completed successfully!"
}

# Parse arguments
case "$1" in
    --help|-h)
        echo "Usage: $0 [VERSION] [OPTIONS]"
        echo ""
        echo "VERSION:"
        echo "  latest (default)    Build with latest tag"
        echo "  v1.0, v2.0, etc    Build with specific version"
        echo ""
        echo "OPTIONS:"
        echo "  --with-lightweight  Also build lightweight image"
        echo "  --cleanup          Clean up old images after build"
        echo "  --help             Show this help message"
        exit 0
        ;;
    *)
        main "$@"
        ;;
esac