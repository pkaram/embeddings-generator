"""Main FastAPI application for the embeddings generator."""

import time
from contextlib import asynccontextmanager
from typing import Dict, Any

import psutil
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.embedding_service import embedding_service
from app.logger import get_logger
from app.models import (
    EmbeddingRequest,
    EmbeddingResponse,
    HealthResponse,
    ModelInfo,
    ErrorResponse,
    SystemResponse,
)

logger = get_logger(__name__)
settings = get_settings()

# Track application start time and prime the process CPU counter.
app_start_time = time.time()
_process = psutil.Process()
_process.cpu_percent(interval=None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Embeddings Generator API")
    logger.info("Model will be loaded on first request to optimize startup time")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Embeddings Generator API")
    embedding_service.unload_model()


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="A framework for generating embeddings using Docker containers with CPU-only models from Hugging Face",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc) if settings.debug else None,
            status_code=500
        ).dict()
    )


@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Embeddings Generator API",
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    uptime = time.time() - app_start_time
    model_loaded = embedding_service.is_model_loaded()
    
    # Consider the service healthy even if model is not loaded yet
    # The model will be loaded on first request
    status = "healthy"
    
    return HealthResponse(
        status=status,
        version=settings.app_version,
        model_loaded=model_loaded,
        uptime=uptime
    )


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the currently loaded model."""
    try:
        model_info = embedding_service.get_model_info()
        
        # If no model is loaded, return 503 Service Unavailable
        if not model_info.get("is_loaded", False):
            raise HTTPException(
                status_code=503, 
                detail="No model is currently loaded. Model will be loaded on first embedding request."
            )
        
        return ModelInfo(**model_info)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embeddings", response_model=EmbeddingResponse)
async def generate_embeddings(request: EmbeddingRequest):
    """Generate embeddings for the provided texts."""
    try:
        # Character cap is an abuse guard. Token truncation happens inside the
        # model at max_seq_length, which is a token limit, not a character limit.
        for i, text in enumerate(request.texts):
            if len(text) > settings.max_text_characters:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Text at index {i} exceeds maximum length of "
                        f"{settings.max_text_characters} characters"
                    )
                )
        
        # Encode runs on the event loop. With the default single worker, concurrent
        # requests queue behind the forward pass. The load test measures that point.
        embeddings, processing_time, load_time, effective_batch_size = embedding_service.generate_embeddings(
            texts=request.texts,
            model_name=request.model_name,
            normalize=request.normalize,
            batch_size=request.batch_size
        )
        
        # Get model info for response
        model_info = embedding_service.get_model_info()
        
        return EmbeddingResponse(
            embeddings=embeddings,
            model_name=model_info["model_name"],
            dimensions=model_info["embedding_dimensions"],
            processing_time=processing_time,
            load_time=load_time,
            batch_size=effective_batch_size,
            total_texts=len(request.texts)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/system", response_model=SystemResponse)
async def system_metrics():
    """Resident memory and process CPU since the previous sample."""
    return SystemResponse(
        rss_bytes=_process.memory_info().rss,
        cpu_percent=_process.cpu_percent(interval=None),
    )


@app.post("/model/load")
async def load_model(model_name: str):
    """Load a specific model."""
    try:
        embedding_service.load_model(model_name)
        return {"message": f"Model {model_name} loaded successfully"}
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/model/unload")
async def unload_model():
    """Unload the current model."""
    try:
        embedding_service.unload_model()
        return {"message": "Model unloaded successfully"}
    except Exception as e:
        logger.error(f"Failed to unload model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=1 if settings.debug else settings.workers,
        log_level=settings.log_level.lower()
    )
