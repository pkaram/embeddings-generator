"""Logging configuration for the embeddings generator."""

import logging
import sys
from typing import Optional

from app.config import get_settings


def setup_logging(log_level: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration."""
    settings = get_settings()
    
    # Use provided log level or default from settings
    level = log_level or settings.log_level
    
    # Create logger
    logger = logging.getLogger("embeddings_generator")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    # Prevent duplicate logs
    logger.propagate = False
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(f"embeddings_generator.{name}")


# Global logger instance
logger = setup_logging()
