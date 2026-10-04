"""Embedding generation service using Hugging Face models."""

import time
from typing import List, Optional, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

from app.config import get_settings
from app.logger import get_logger
from app.measurement import model_disk_size_bytes

logger = get_logger(__name__)


class EmbeddingService:
    """Service for generating embeddings using Hugging Face models."""

    def __init__(self):
        """Initialize the embedding service."""
        self.settings = get_settings()
        self.model: Optional[SentenceTransformer] = None
        self.model_name: Optional[str] = None
        self.device = "cpu"  # Force CPU usage
        
        # Ensure model cache directory exists
        import os
        os.makedirs(self.settings.model_cache_dir, exist_ok=True)

    def load_model(self, model_name: Optional[str] = None) -> float:
        """Load a Hugging Face model. Returns seconds spent loading, or 0 if already resident."""
        model_name = model_name or self.settings.default_model_name
        
        if self.model_name == model_name and self.model is not None:
            logger.info(f"Model {model_name} is already loaded")
            return 0.0
        
        try:
            logger.info(f"Loading model: {model_name}")
            start_time = time.perf_counter()
            
            # Load model with CPU-only configuration
            self.model = SentenceTransformer(
                model_name,
                cache_folder=self.settings.model_cache_dir,
                device=self.device
            )
            
            # Ensure model is in evaluation mode and on CPU
            self.model.eval()
            if hasattr(self.model, 'to'):
                self.model.to(self.device)

            # Cap token length at the model's own positional limit.
            native_limit = getattr(self.model, "max_seq_length", self.settings.max_sequence_length)
            self.model.max_seq_length = min(int(native_limit), int(self.settings.max_sequence_length))
            
            self.model_name = model_name
            
            load_time = time.perf_counter() - start_time
            logger.info(
                f"Model {model_name} loaded successfully in {load_time:.2f} seconds "
                f"(max_seq_length={self.model.max_seq_length} tokens)"
            )
            return load_time
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise RuntimeError(f"Failed to load model {model_name}: {str(e)}")

    def generate_embeddings(
        self,
        texts: List[str],
        model_name: Optional[str] = None,
        normalize: bool = True,
        batch_size: Optional[int] = None
    ) -> Tuple[List[List[float]], float, float, int]:
        """Generate embeddings for a list of texts.

        Returns ``(embeddings, encode_time_seconds, load_time_seconds, batch_size)``.
        ``encode_time_seconds`` is only the forward pass. Load time is zero
        when the requested model is already resident. ``batch_size`` is the
        value actually passed to the encoder after clamping.
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")
        
        # Load model if needed. This clock is separate from the encode clock.
        load_time = self.load_model(model_name)
        
        if self.model is None:
            raise RuntimeError("Model is not loaded")
        
        # The library batches internally. Clamping here keeps a request from
        # exceeding the configured memory budget.
        requested = self.settings.max_batch_size if batch_size is None else batch_size
        batch_size = max(1, min(int(requested), self.settings.max_batch_size))
        
        try:
            logger.info(
                f"Generating embeddings for {len(texts)} texts using model {self.model_name} "
                f"with batch_size={batch_size}"
            )
            start_time = time.perf_counter()
            encoded = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_tensor=False,
                normalize_embeddings=normalize,
                show_progress_bar=False
            )
            all_embeddings = encoded.tolist()
            
            encode_time = time.perf_counter() - start_time
            logger.info(
                f"Generated embeddings for {len(texts)} texts in {encode_time:.2f} seconds "
                f"(load_time={load_time:.2f}s)"
            )
            
            return all_embeddings, encode_time, load_time, batch_size
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            raise RuntimeError(f"Failed to generate embeddings: {str(e)}")

    def get_model_info(self) -> dict:
        """Get information about the currently loaded model."""
        if self.model is None:
            return {
                "model_name": None,
                "model_type": None,
                "max_sequence_length": None,
                "embedding_dimensions": None,
                "model_size_bytes": None,
                "is_loaded": False
            }
        
        try:
            # Get model dimensions
            if hasattr(self.model, 'get_sentence_embedding_dimension'):
                dimensions = self.model.get_sentence_embedding_dimension()
            else:
                # Fallback: generate a test embedding to get dimensions
                test_embedding = self.model.encode(["test"])
                dimensions = len(test_embedding[0])
            
            # Get max sequence length
            max_length = getattr(self.model, 'max_seq_length', self.settings.max_sequence_length)
            
            return {
                "model_name": self.model_name,
                "model_type": "sentence-transformer",
                "max_sequence_length": max_length,
                "embedding_dimensions": dimensions,
                "model_size_bytes": model_disk_size_bytes(
                    self.settings.model_cache_dir, self.model_name
                ),
                "is_loaded": True
            }
            
        except Exception as e:
            logger.error(f"Failed to get model info: {str(e)}")
            return {
                "model_name": self.model_name,
                "model_type": "sentence-transformer",
                "max_sequence_length": self.settings.max_sequence_length,
                "embedding_dimensions": None,
                "model_size_bytes": None,
                "is_loaded": False
            }

    def is_model_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self.model is not None

    def unload_model(self) -> None:
        """Unload the current model to free memory."""
        if self.model is not None:
            logger.info(f"Unloading model: {self.model_name}")
            del self.model
            self.model = None
            self.model_name = None
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# Global service instance
embedding_service = EmbeddingService()
