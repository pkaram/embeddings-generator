# Embeddings Generator

A production-ready framework for generating text embeddings using Docker containers with CPU-only models from Hugging Face. Built with FastAPI and following best engineering practices.

## Features

- 🚀 **FastAPI-based REST API** with automatic OpenAPI documentation
- 🐳 **Docker-first approach** with CPU-only inference for cost efficiency
- 🤗 **Hugging Face integration** with support for sentence-transformers models
- 📊 **Measurement suite** for latency percentiles, throughput, cold vs warm load, retrieval quality, and concurrency
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
The first request for a model that is not resident pays `load_time` as well as `processing_time`. Later requests for the same model report `load_time` of 0. A download from Hugging Face is included in `load_time` only when the model is not already in the cache.

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
  "load_time": 0.0,
  "batch_size": 16,
  "total_texts": 3
}
```

`processing_time` is only the forward pass. `load_time` is the time spent loading the model for this request, and it is `0` when that model is already resident. The client-observed HTTP time, including JSON, is measured by the benchmark tools rather than returned by the API.

## Configuration

Set `MAX_BATCH_SIZE`, `MAX_SEQUENCE_LENGTH`, and `MAX_TEXT_CHARACTERS` in `scripts/start_container.sh` or as environment variables.

- `MAX_BATCH_SIZE` (default 64) is forwarded to `SentenceTransformer.encode`. Larger requested values are clamped, and the response `batch_size` is the value actually used.
- `MAX_SEQUENCE_LENGTH` (default 512) is a token cap. The resident model uses the smaller of this value and its own positional limit, and it truncates longer inputs.
- `MAX_TEXT_CHARACTERS` (default 20000) rejects a single text that is large enough to be an abuse of the payload limit.

### Supported Models

The framework supports sentence-transformers models from Hugging Face. The comparison set is:

- `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, speed baseline)
- `BAAI/bge-small-en-v1.5` (384 dimensions, retrieval-tuned)
- `sentence-transformers/all-mpnet-base-v2` (768 dimensions, larger quality ceiling on CPU)


### Resource Limits

`scripts/start_container.sh` sets a 4GB memory limit on the container. The benchmark records process RSS against that limit.

### Project Structure

```
embeddings-generator/
├── app/
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration management
│   ├── models.py            # Pydantic models
│   ├── embedding_service.py # Load and encode
│   └── measurement.py       # Percentiles and retrieval metrics
├── benchmarks/              # Latency, retrieval, load, and comparison CLIs
├── docs/
│   └── architecture.md      # Request path and measurement clocks
├── scripts/
│   ├── start_container.sh   # Build image and start container
│   └── setup_benchmarks.sh  # Create the local .venv for the benchmark CLI
├── tests/
├── Dockerfile
├── requirements.txt         # API image
├── requirements-benchmark.txt
└── README.md
```

## Measurement

Start the API first (`./scripts/start_container.sh`). The benchmark commands run on the host, from the repository root, and talk to that API. They need a local virtualenv. `.venv` is not committed; create it once per machine:

```bash
./scripts/setup_benchmarks.sh
```

Each command prints a report and writes JSON and Markdown under `results/`.

```bash
# Cold load, warmup encode, steady-state p50/p95/p99, throughput, RSS, and CPU.
# Sweeps batch sizes 1, 8, 16, 32, and 64. Each steady request sends 64 texts.
.venv/bin/python -m benchmarks benchmark --base-url http://localhost:8000

# BEIR SciFact: Recall@10, MRR@10, and nDCG@10.
.venv/bin/python -m benchmarks retrieval --base-url http://localhost:8000

# Warm model, concurrency 1, 2, 4, 8, and 16.
.venv/bin/python -m benchmarks load --base-url http://localhost:8000

# The three comparison models, including retrieval.
.venv/bin/python -m benchmarks compare --base-url http://localhost:8000
```

`compare` is the long run: three models, the batch sweep, and the full SciFact corpus. `--requests` (default 30) and `--skip-retrieval` shorten it. `--max-docs` and `--max-queries` score a subset; the report marks that row as a subset. `--timeout` (default 3600 seconds) is how long one request may take, including a first-time model download. Each finished model is written to `results/benchmark-partial.md` before the next one starts.

`--cache-dir` defaults to `./models`, which is the volume mounted by `start_container.sh`. When that directory is visible and the model is not in it, the benchmark records a cache-miss load (download included) and then a cold load from disk. `--measure-cache-miss` deletes the cached model first.

The load test holds a fixed number of requests in flight. Saturation is the first concurrency whose texts/s failed to beat the best lower level by 10 percent. That point is expected to be low: encode blocks the event loop and the process is one worker. See [docs/architecture.md](docs/architecture.md) for the clocks.

## Health and resources

- `GET /health` reports status, version, whether a model is resident, and uptime
- `GET /system` reports the API process RSS and CPU percent since the previous sample
- Docker restarts the container when the health check fails
- Logs include model load time and encode time

## Performance

- CPU-only inference
- `batch_size` is passed through to the encoder
- The model cache is mounted at `/app/models` so restarts do not re-download
- `POST /model/unload` drops the resident model before a cold-load measurement
- One model is resident at a time

## Security

- **Non-root user** in Docker container
- **Input validation** with Pydantic models
- **Error handling** without information leakage
- **CORS configuration** for cross-origin requests
