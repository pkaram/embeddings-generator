"""Embedding generation service using Hugging Face models."""

import time
from typing import List, Optional, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

from app.config import get_settings
from app.logger import get_logger

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

    def load_model(self, model_name: Optional[str] = None) -> None:
        """Load a Hugging Face model for embedding generation."""
        model_name = model_name or self.settings.default_model_name
        
        if self.model_name == model_name and self.model is not None:
            logger.info(f"Model {model_name} is already loaded")
            return
        
        try:
            logger.info(f"Loading model: {model_name}")
            start_time = time.time()
            
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
            
            self.model_name = model_name
            
            load_time = time.time() - start_time
            logger.info(f"Model {model_name} loaded successfully in {load_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise RuntimeError(f"Failed to load model {model_name}: {str(e)}")

    def generate_embeddings(
        self,
        texts: List[str],
        model_name: Optional[str] = None,
        normalize: bool = True,
        batch_size: Optional[int] = None
    ) -> Tuple[List[List[float]], float]:
        """Generate embeddings for a list of texts."""
        if not texts:
            raise ValueError("Texts list cannot be empty")
        
        # Load model if needed
        self.load_model(model_name)
        
        if self.model is None:
            raise RuntimeError("Model is not loaded")
        
        # Use provided batch size or default
        batch_size = batch_size or self.settings.max_batch_size
        
        try:
            logger.info(f"Generating embeddings for {len(texts)} texts using model {self.model_name}")
            start_time = time.time()
            
            # Generate embeddings in batches
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                logger.debug(f"Processing batch {i//batch_size + 1}: {len(batch_texts)} texts")
                
                # Generate embeddings for the batch
                batch_embeddings = self.model.encode(
                    batch_texts,
                    convert_to_tensor=False,
                    normalize_embeddings=normalize,
                    show_progress_bar=False
                )
                
                all_embeddings.extend(batch_embeddings.tolist())
            
            processing_time = time.time() - start_time
            logger.info(f"Generated embeddings for {len(texts)} texts in {processing_time:.2f} seconds")
            
            return all_embeddings, processing_time
            
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
                "is_loaded": True
            }
            
        except Exception as e:
            logger.error(f"Failed to get model info: {str(e)}")
            return {
                "model_name": self.model_name,
                "model_type": "sentence-transformer",
                "max_sequence_length": self.settings.max_sequence_length,
                "embedding_dimensions": None,
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
