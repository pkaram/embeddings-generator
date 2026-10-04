#!/bin/bash
# Script to run embeddings generator using Dockerfile

set -e

echo "🚀 Running Embeddings Generator (Dockerfile only)"
echo "================================================"

# Configuration
MODEL_NAME="${1:-sentence-transformers/all-MiniLM-L6-v2}"
MODELS_DIR="./models"
IMAGE_NAME="embeddings-generator"
CONTAINER_NAME="embeddings-generator-container"

echo "Model: ${MODEL_NAME}"
echo "Models directory: ${MODELS_DIR}"
echo ""

# Create models directory if it doesn't exist
mkdir -p "${MODELS_DIR}"

# Build the image
echo "🔨 Building Docker image..."
docker build -t "${IMAGE_NAME}" .

# Stop and remove existing container if it exists
echo "🧹 Cleaning up existing container..."
docker stop "${CONTAINER_NAME}" 2>/dev/null || true
docker rm "${CONTAINER_NAME}" 2>/dev/null || true

# Run the container
echo "🐳 Starting container..."
docker run -d \
    --name "${CONTAINER_NAME}" \
    -p 8000:8000 \
    -v "$(pwd)/${MODELS_DIR}:/app/models" \
    -e DEFAULT_MODEL_NAME="${MODEL_NAME}" \
    -e MODEL_CACHE_DIR="/app/models" \
    -e MAX_BATCH_SIZE=64 \
    -e MAX_SEQUENCE_LENGTH=512 \
    --restart unless-stopped \
    --memory=4g \
    "${IMAGE_NAME}"

echo ""
echo "✅ Container started successfully!"
echo ""
echo "🧪 STARTUP PROCESS:"
echo "   1. ✅ Container is now running"
echo "   2. 🧪 Tests are running automatically (10-30 seconds)"
echo "   3. 🌐 API server will start after tests pass"
echo ""
echo "👀 WATCH STARTUP PROGRESS:"
echo "   Follow logs in real-time: docker logs -f ${CONTAINER_NAME}"
echo ""
echo "💡 To see what's happening right now, run the command above!"
echo "   You'll see test progress and know exactly when the API is ready."
echo ""
echo "🔗 Once startup completes, access:"
echo "   • API: http://localhost:8000"
echo "   • Docs: http://localhost:8000/docs"
echo "   • Health: http://localhost:8000/health"
echo ""
echo "📋 Other useful commands:"
echo "   View all logs: docker logs ${CONTAINER_NAME}"
echo "   Stop: docker stop ${CONTAINER_NAME}"
echo "   Remove: docker rm ${CONTAINER_NAME}"
echo "   Shell access: docker exec -it ${CONTAINER_NAME} bash"
echo ""
echo "⚡ Quick start: Run this now to watch progress:"
echo "   docker logs -f ${CONTAINER_NAME}"
echo ""

# Ask if user wants to follow logs immediately
read -p "🤔 Would you like to follow the startup logs now? (y/N): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📺 Following container logs... (Press Ctrl+C to stop watching)"
    echo ""
    docker logs -f "${CONTAINER_NAME}"
fi
