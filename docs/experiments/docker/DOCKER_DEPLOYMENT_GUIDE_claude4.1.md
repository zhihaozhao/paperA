# 🐳 Docker Environment Deployment and Usage Guide

## 📋 Table of Contents
1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Detailed Setup](#detailed-setup)
4. [Running Experiments](#running-experiments)
5. [GPU Support](#gpu-support)
6. [Data Management](#data-management)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Usage](#advanced-usage)

## 🔧 Prerequisites

### System Requirements
- Docker Engine 20.10+ 
- Docker Compose 2.0+ (optional)
- NVIDIA Docker Runtime (for GPU support)
- 16GB+ RAM
- 50GB+ free disk space

### Installation

#### Ubuntu/Debian
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# For GPU support - Install NVIDIA Docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

#### macOS
```bash
# Install Docker Desktop
# Download from: https://www.docker.com/products/docker-desktop/

# Verify installation
docker --version
docker-compose --version
```

#### Windows
```powershell
# Install Docker Desktop for Windows
# Download from: https://www.docker.com/products/docker-desktop/

# Enable WSL2 backend for better performance
wsl --set-default-version 2

# Verify installation
docker --version
docker-compose --version
```

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/paperA.git
cd paperA
```

### 2. Build Docker Image
```bash
# Build the main experiment image
cd docs/experiments/docker
docker build -t wifi-csi-har:latest -f Dockerfile_claude4.1 .

# Or use the build script
./build_docker.sh
```

### 3. Run Container
```bash
# CPU-only version
docker run -it --rm \
  -v $(pwd):/workspace \
  -v $(pwd)/data:/data \
  wifi-csi-har:latest

# GPU-enabled version
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  -v $(pwd)/data:/data \
  wifi-csi-har:latest
```

## 📦 Detailed Setup

### Docker Image Structure

Our Docker image includes:
- **Base**: Ubuntu 22.04 with CUDA 11.8
- **Python**: 3.10 with scientific computing stack
- **Deep Learning**: PyTorch 2.0, TensorFlow 2.13
- **WiFi CSI Tools**: Custom processing libraries
- **Experiment Code**: All 4 experiment models

### Building Custom Images

#### Standard Build
```dockerfile
# Dockerfile_claude4.1
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy requirements
COPY requirements_claude4.1.txt .
RUN pip3 install --no-cache-dir -r requirements_claude4.1.txt

# Copy experiment code
COPY exp1_enhanced_sim2real/ ./exp1_enhanced_sim2real/
COPY exp2_enhanced_pinn_loss/ ./exp2_enhanced_pinn_loss/
COPY exp3_pinn_lstm_causal/ ./exp3_pinn_lstm_causal/
COPY exp4_mamba_efficiency/ ./exp4_mamba_efficiency/

# Set entrypoint
ENTRYPOINT ["python3"]
```

#### Lightweight Build (for edge deployment)
```dockerfile
# Dockerfile.lightweight
FROM python:3.10-slim

WORKDIR /workspace

# Install minimal dependencies
COPY requirements_minimal.txt .
RUN pip install --no-cache-dir -r requirements_minimal.txt

# Copy only Exp4 lightweight model
COPY exp4_mamba_efficiency/ ./exp4_mamba_efficiency/

ENTRYPOINT ["python3"]
```

### Docker Compose Configuration

```yaml
# docker-compose.yml
version: '3.8'

services:
  experiment-runner:
    build:
      context: .
      dockerfile: Dockerfile_claude4.1
    image: wifi-csi-har:latest
    container_name: csi-experiments
    runtime: nvidia  # For GPU support
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - PYTHONUNBUFFERED=1
      - WANDB_API_KEY=${WANDB_API_KEY}
    volumes:
      - ./:/workspace
      - ./data:/data
      - ./results:/results
      - ./checkpoints:/checkpoints
    ports:
      - "8888:8888"  # Jupyter
      - "6006:6006"  # TensorBoard
    command: jupyter lab --ip=0.0.0.0 --allow-root

  tensorboard:
    image: tensorflow/tensorflow:latest
    container_name: csi-tensorboard
    volumes:
      - ./logs:/logs
    ports:
      - "6007:6006"
    command: tensorboard --logdir=/logs --host=0.0.0.0

  data-processor:
    build:
      context: .
      dockerfile: Dockerfile_claude4.1
    image: wifi-csi-har:latest
    container_name: csi-data-processor
    volumes:
      - ./data:/data
      - ./processed:/processed
    command: python3 preprocess_data.py
```

### Starting Services
```bash
# Start all services
docker-compose up -d

# Start specific service
docker-compose up experiment-runner

# View logs
docker-compose logs -f experiment-runner

# Stop services
docker-compose down
```

## 🧪 Running Experiments

### Interactive Mode
```bash
# Start interactive container
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  -v $(pwd)/data:/data \
  -v $(pwd)/results:/results \
  wifi-csi-har:latest bash

# Inside container - run experiments
cd /workspace
python unified_experiment_runner_claude4.1.py --model exp1_sim2real --epochs 20
```

### Batch Mode
```bash
# Run specific experiment
docker run --gpus all --rm \
  -v $(pwd):/workspace \
  -v $(pwd)/data:/data \
  wifi-csi-har:latest \
  unified_experiment_runner_claude4.1.py --model exp2_pinn_loss --epochs 50

# Run all experiments
docker run --gpus all --rm \
  -v $(pwd):/workspace \
  wifi-csi-har:latest \
  bash run_all_experiments.sh
```

### Jupyter Notebook
```bash
# Start Jupyter server
docker run --gpus all -p 8888:8888 --rm \
  -v $(pwd):/workspace \
  wifi-csi-har:latest \
  jupyter lab --ip=0.0.0.0 --allow-root --no-browser

# Access at http://localhost:8888
```

## 🎮 GPU Support

### Verify GPU Access
```bash
# Check NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Inside container
docker run --gpus all -it wifi-csi-har:latest bash
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
```

### Multi-GPU Training
```bash
# Use all GPUs
docker run --gpus all --rm \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  wifi-csi-har:latest \
  python train_distributed.py

# Use specific GPUs
docker run --gpus '"device=0,1"' --rm \
  wifi-csi-har:latest \
  python train.py --gpus 0,1
```

## 💾 Data Management

### Volume Mounting
```bash
# Mount data directory
docker run -v /path/to/local/data:/data wifi-csi-har:latest

# Mount with read-only
docker run -v /path/to/local/data:/data:ro wifi-csi-har:latest

# Named volumes for persistence
docker volume create csi-data
docker run -v csi-data:/data wifi-csi-har:latest
```

### Data Transfer
```bash
# Copy data into container
docker cp local_data.zip container_id:/workspace/

# Copy results from container
docker cp container_id:/workspace/results ./local_results

# Using docker-compose
docker-compose exec experiment-runner tar -czf /tmp/results.tar.gz /results
docker cp $(docker-compose ps -q experiment-runner):/tmp/results.tar.gz .
```

## 🔧 Troubleshooting

### Common Issues

#### 1. CUDA/GPU Not Available
```bash
# Check NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:11.8.0-base nvidia-smi

# If fails, reinstall nvidia-docker2
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Verify Docker daemon configuration
cat /etc/docker/daemon.json
# Should contain:
{
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
```

#### 2. Out of Memory
```bash
# Limit memory usage
docker run -m 8g --memory-swap 8g wifi-csi-har:latest

# Clear Docker cache
docker system prune -a
docker volume prune
```

#### 3. Permission Denied
```bash
# Run with user permissions
docker run --user $(id -u):$(id -g) \
  -v $(pwd):/workspace \
  wifi-csi-har:latest

# Fix file permissions
docker run --rm -v $(pwd):/workspace wifi-csi-har:latest \
  chown -R $(id -u):$(id -g) /workspace
```

#### 4. Slow Build
```bash
# Use BuildKit for faster builds
DOCKER_BUILDKIT=1 docker build -t wifi-csi-har:latest .

# Multi-stage build with cache
docker build --target base -t wifi-csi-har:base .
docker build --cache-from wifi-csi-har:base -t wifi-csi-har:latest .
```

## 🚀 Advanced Usage

### Custom Entrypoint Script
```bash
# entrypoint.sh
#!/bin/bash
set -e

# Activate virtual environment if exists
if [ -f /opt/venv/bin/activate ]; then
    source /opt/venv/bin/activate
fi

# Set Python path
export PYTHONPATH=/workspace:$PYTHONPATH

# Run command
exec "$@"
```

### Environment Variables
```bash
# .env file
CUDA_VISIBLE_DEVICES=0
WANDB_API_KEY=your_key_here
EXPERIMENT_NAME=exp1_sim2real
BATCH_SIZE=32
LEARNING_RATE=0.001

# Run with env file
docker run --env-file .env wifi-csi-har:latest
```

### Health Checks
```dockerfile
# Add to Dockerfile
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python3 -c "import torch; assert torch.cuda.is_available()" || exit 1
```

### Resource Limits
```bash
# Limit CPU and memory
docker run \
  --cpus="4.0" \
  --memory="16g" \
  --memory-swap="16g" \
  --shm-size="8g" \
  wifi-csi-har:latest

# With docker-compose
services:
  experiment:
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 16G
        reservations:
          devices:
            - driver: nvidia
              count: 2
              capabilities: [gpu]
```

### Monitoring
```bash
# Real-time stats
docker stats

# Container logs
docker logs -f container_id

# Inside container monitoring
docker exec -it container_id nvidia-smi -l 1
docker exec -it container_id htop
```

### Backup and Export
```bash
# Save image
docker save wifi-csi-har:latest | gzip > wifi-csi-har.tar.gz

# Load image
docker load < wifi-csi-har.tar.gz

# Export container
docker export container_id > container_backup.tar

# Push to registry
docker tag wifi-csi-har:latest your-registry/wifi-csi-har:latest
docker push your-registry/wifi-csi-har:latest
```

## 📝 Scripts

### build_docker.sh
```bash
#!/bin/bash
# Build Docker image with caching

set -e

IMAGE_NAME="wifi-csi-har"
VERSION="latest"

echo "Building Docker image: ${IMAGE_NAME}:${VERSION}"

# Enable BuildKit
export DOCKER_BUILDKIT=1

# Build with cache
docker build \
  --cache-from ${IMAGE_NAME}:${VERSION} \
  --build-arg BUILDKIT_INLINE_CACHE=1 \
  -t ${IMAGE_NAME}:${VERSION} \
  -f Dockerfile_claude4.1 \
  .

echo "Build complete!"
echo "Run with: docker run --gpus all -it ${IMAGE_NAME}:${VERSION}"
```

### run_experiment.sh
```bash
#!/bin/bash
# Run experiment in Docker

MODEL=${1:-exp1_sim2real}
EPOCHS=${2:-20}
GPU=${3:-0}

docker run --gpus all --rm \
  -e CUDA_VISIBLE_DEVICES=${GPU} \
  -v $(pwd):/workspace \
  -v $(pwd)/data:/data \
  -v $(pwd)/results:/results \
  wifi-csi-har:latest \
  unified_experiment_runner_claude4.1.py \
  --model ${MODEL} \
  --epochs ${EPOCHS}
```

### cleanup.sh
```bash
#!/bin/bash
# Clean up Docker resources

echo "Stopping all containers..."
docker stop $(docker ps -aq) 2>/dev/null

echo "Removing stopped containers..."
docker container prune -f

echo "Removing unused images..."
docker image prune -a -f

echo "Removing unused volumes..."
docker volume prune -f

echo "Removing build cache..."
docker builder prune -a -f

echo "Cleanup complete!"
docker system df
```

## 🌐 Cloud Deployment

### AWS ECR
```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  123456789.dkr.ecr.us-east-1.amazonaws.com

# Tag and push
docker tag wifi-csi-har:latest \
  123456789.dkr.ecr.us-east-1.amazonaws.com/wifi-csi-har:latest
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/wifi-csi-har:latest
```

### Google Cloud Run
```bash
# Build and push to GCR
gcloud builds submit --tag gcr.io/PROJECT_ID/wifi-csi-har

# Deploy
gcloud run deploy wifi-csi-har \
  --image gcr.io/PROJECT_ID/wifi-csi-har \
  --platform managed \
  --memory 8Gi \
  --cpu 4
```

### Kubernetes Deployment
```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: wifi-csi-experiments
spec:
  replicas: 1
  selector:
    matchLabels:
      app: wifi-csi
  template:
    metadata:
      labels:
        app: wifi-csi
    spec:
      containers:
      - name: experiment-runner
        image: wifi-csi-har:latest
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "16Gi"
            cpu: "4"
        volumeMounts:
        - name: data
          mountPath: /data
        - name: results
          mountPath: /results
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: csi-data-pvc
      - name: results
        persistentVolumeClaim:
          claimName: csi-results-pvc
```

## 📚 Best Practices

1. **Layer Caching**: Order Dockerfile commands from least to most frequently changing
2. **Multi-stage Builds**: Use for smaller production images
3. **Security**: Don't run as root in production, scan images for vulnerabilities
4. **Reproducibility**: Pin all dependency versions
5. **Documentation**: Keep Dockerfile well-commented
6. **Testing**: Include tests in CI/CD pipeline

## 🔗 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [NVIDIA Docker Guide](https://github.com/NVIDIA/nvidia-docker)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Container Security](https://docs.docker.com/engine/security/)

---

**Version**: 1.0
**Last Updated**: December 2024
**Maintainer**: Claude 4.1