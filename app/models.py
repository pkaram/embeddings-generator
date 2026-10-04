"""Pydantic models for request/response schemas."""

from typing import List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class EmbeddingRequest(BaseModel):
    """Request model for generating embeddings."""

    model_config = ConfigDict(protected_namespaces=())

    texts: List[str] = Field(
        ..., 
        description="List of texts to generate embeddings for",
        min_items=1,
        max_items=100
    )
    model_name: Optional[str] = Field(
        default=None,
        description="Name of the Hugging Face model to use. If not provided, uses default model."
    )
    normalize: bool = Field(
        default=True,
        description="Whether to normalize the embeddings to unit vectors"
    )
    batch_size: Optional[int] = Field(
        default=None,
        ge=1,
        description="Batch size forwarded to the encoder. If omitted, the server default is used. Values above the configured maximum are clamped."
    )


class EmbeddingResponse(BaseModel):
    """Response model for embedding generation."""

    model_config = ConfigDict(protected_namespaces=())

    embeddings: List[List[float]] = Field(
        ...,
        description="List of embedding vectors, one for each input text"
    )
    model_name: str = Field(
        ...,
        description="Name of the model used to generate embeddings"
    )
    dimensions: int = Field(
        ...,
        description="Number of dimensions in each embedding vector"
    )
    processing_time: float = Field(
        ...,
        description="Seconds spent in model.encode. Excludes model load and JSON serialization."
    )
    load_time: float = Field(
        ...,
        description="Seconds spent loading the model for this request. Zero when the model was already resident."
    )
    batch_size: int = Field(
        ...,
        description="Batch size actually passed to the encoder after clamping to the server maximum."
    )
    total_texts: int = Field(
        ...,
        description="Total number of texts processed"
    )


class HealthResponse(BaseModel):
    """Health check response model."""

    model_config = ConfigDict(protected_namespaces=())

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Application version")
    model_loaded: bool = Field(..., description="Whether the default model is loaded")
    uptime: float = Field(..., description="Service uptime in seconds")


class ModelInfo(BaseModel):
    """Model information response model."""

    model_config = ConfigDict(protected_namespaces=())

    model_name: str = Field(..., description="Name of the model")
    model_type: str = Field(..., description="Type of the model")
    max_sequence_length: int = Field(..., description="Maximum sequence length supported")
    embedding_dimensions: int = Field(..., description="Number of embedding dimensions")
    model_size_bytes: Optional[int] = Field(
        default=None,
        description="On-disk size of the cached model in bytes, or null if the cache directory is not visible."
    )
    is_loaded: bool = Field(..., description="Whether the model is currently loaded")


class SystemResponse(BaseModel):
    """Process resource sample for benchmark clients."""

    rss_bytes: int = Field(..., description="Resident set size of the API process in bytes")
    cpu_percent: float = Field(
        ...,
        description="Process CPU percent since the previous sample. The first sample after startup is zero."
    )


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(default=None, description="Detailed error information")
    status_code: int = Field(..., description="HTTP status code")
