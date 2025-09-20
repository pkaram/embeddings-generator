"""Configuration management for the embeddings generator."""

import os
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # Application Configuration
    app_name: str = Field(default="Embeddings Generator", env="APP_NAME")
    app_version: str = Field(default="0.1.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    # Server Configuration
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    workers: int = Field(default=1, env="WORKERS")

    # Model Configuration
    default_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", env="DEFAULT_MODEL_NAME"
    )
    model_cache_dir: str = Field(default="/app/models", env="MODEL_CACHE_DIR", alias="model_cache_dir")
    max_batch_size: int = Field(default=32, env="MAX_BATCH_SIZE")
    max_sequence_length: int = Field(default=512, env="MAX_SEQUENCE_LENGTH")

    # Docker Configuration
    docker_image_name: str = Field(default="embeddings-generator", env="DOCKER_IMAGE_NAME")
    docker_tag: str = Field(default="latest", env="DOCKER_TAG")

    # Security
    secret_key: str = Field(default="your-secret-key-here", env="SECRET_KEY")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")

    # Monitoring
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")

    class Config:
        """Pydantic config."""

        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        protected_namespaces = ('settings_',)


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings."""
    return settings
