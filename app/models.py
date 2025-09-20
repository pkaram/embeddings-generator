"""Pydantic models for request/response schemas."""

from typing import List, Optional, Union

from pydantic import BaseModel, Field


class EmbeddingRequest(BaseModel):
    """Request model for generating embeddings."""

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
        description="Batch size for processing. If not provided, uses default batch size."
    )


class EmbeddingResponse(BaseModel):
    """Response model for embedding generation."""

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
        description="Time taken to process the request in seconds"
    )
    total_texts: int = Field(
        ...,
        description="Total number of texts processed"
    )


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Application version")
    model_loaded: bool = Field(..., description="Whether the default model is loaded")
    uptime: float = Field(..., description="Service uptime in seconds")


class ModelInfo(BaseModel):
    """Model information response model."""

    model_name: str = Field(..., description="Name of the model")
    model_type: str = Field(..., description="Type of the model")
    max_sequence_length: int = Field(..., description="Maximum sequence length supported")
    embedding_dimensions: int = Field(..., description="Number of embedding dimensions")
    is_loaded: bool = Field(..., description="Whether the model is currently loaded")


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(default=None, description="Detailed error information")
    status_code: int = Field(..., description="HTTP status code")
