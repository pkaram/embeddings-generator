# Embeddings Generator

A production-ready framework for generating text embeddings using Docker containers with CPU-only models from Hugging Face. Built with FastAPI and following best engineering practices.

## Features

- 🚀 **FastAPI-based REST API** with automatic OpenAPI documentation
- 🐳 **Docker-first approach** with CPU-only inference for cost efficiency
- 🤗 **Hugging Face integration** with support for sentence-transformers models
- 📊 **Comprehensive monitoring** with health checks and metrics
- 🔧 **Configurable** via environment variables
- 🛡️ **Production-ready** with proper logging, error handling, and security
- 📦 **Easy deployment** with Docker Compose

## Quick Start

### Using Docker Compose (Recommended)

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd embeddings-generator
   ```

2. **Make script executable and start the service:**
   ```bash
   chmod +x scripts/start_container.sh
   ./scripts/start_container.sh
   ```

3. **Test the API:**
   ```bash
   curl http://localhost:8000/health
   ```

4. **Generate embeddings:**
   ```bash
   curl -X POST "http://localhost:8000/embeddings" \
        -H "Content-Type: application/json" \
        -d '{
          "texts": ["Hello world", "This is a test"],
          "normalize": true
        }'
   ```

## API Endpoints

### Core Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `GET /docs` - Interactive API documentation (Swagger UI)
- `GET /redoc` - Alternative API documentation

### Embedding Endpoints

- `POST /embeddings` - Generate embeddings for texts
- `GET /model/info` - Get information about the loaded model
- `POST /model/load` - Load a specific model
- `POST /model/unload` - Unload the current model

### Example API Usage

#### Load a specific model

```bash
curl -X POST "http://localhost:8000/model/load?model_name=sentence-transformers/all-MiniLM-L6-v2"
```

#### Check model info

```bash
curl http://localhost:8000/model/info
```

#### Generate Embeddings
```bash
curl -X POST "http://localhost:8000/embeddings" \
     -H "Content-Type: application/json" \
     -d '{
       "texts": [
         "The quick brown fox jumps over the lazy dog",
         "Machine learning is fascinating",
         "Natural language processing with transformers"
       ],
       "model_name": "sentence-transformers/all-MiniLM-L6-v2",
       "normalize": true,
       "batch_size": 16
     }'
```
Note that depending on the model used in **model_name**, model will first be loaded (if not loaded already), therefore initial request with new model will be executed slower.

#### Response
```json
{
  "embeddings": [
    [0.1, 0.2, 0.3, ...],
    [0.4, 0.5, 0.6, ...],
    [0.7, 0.8, 0.9, ...]
  ],
  "model_name": "sentence-transformers/all-MiniLM-L6-v2",
  "dimensions": 384,
  "processing_time": 0.15,
  "total_texts": 3
}
```

## Configuration

The application can be configured by modifying MAX_BATCH_SIZE, MAX_SEQUENCE_LENGTH in start_container.sh 

### Supported Models

The framework supports any sentence-transformers model from Hugging Face. Popular options:

- `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, fast)
- `sentence-transformers/all-mpnet-base-v2` (768 dimensions, high quality)
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (multilingual)


### Resource Limits

The configuration includes:
- Memory limit: 4GB
- Memory reservation: 2GB
- Health checks with automatic restart

### Project Structure

```
embeddings-generator/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration management
│   ├── models.py            # Pydantic models
│   ├── embedding_service.py # Core embedding logic
│   └── logger.py            # Logging configuration
├── scripts/
│   └── start_container.sh     # Build image and start container
├── tests/
    ├── __init__.py
│   └── test_main.py        # Tests for endpoints
├── Dockerfile              # Docker image definition
├── requirements.txt        # Python dependencies
├── pyproject.toml         # Project configuration
└── README.md              # This file
```

## Monitoring and Health Checks

The application includes comprehensive monitoring:

- **Health endpoint** (`/health`) with model status
- **Docker health checks** with automatic restart
- **Structured logging** with configurable levels
- **Error handling** with detailed error responses

## Performance Considerations

- **CPU-only inference** for cost efficiency
- **Batch processing** for improved throughput
- **Model caching** to avoid repeated downloads
- **Memory management** with model unloading capabilities
- **Configurable batch sizes** for different hardware

## Security

- **Non-root user** in Docker container
- **Input validation** with Pydantic models
- **Error handling** without information leakage
- **CORS configuration** for cross-origin requests
