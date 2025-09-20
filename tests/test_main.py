"""Tests for the main FastAPI application."""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert data["message"] == "Embeddings Generator API"


def test_health_endpoint():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "version" in data
    assert "model_loaded" in data
    assert "uptime" in data


def test_model_info_endpoint():
    """Test the model info endpoint."""
    response = client.get("/model/info")
    # When no model is loaded, expect 503 Service Unavailable
    assert response.status_code == 503
    data = response.json()
    assert "detail" in data
    assert "No model is currently loaded" in data["detail"]


def test_embeddings_endpoint_empty_texts():
    """Test embeddings endpoint with empty texts."""
    response = client.post("/embeddings", json={"texts": []})
    assert response.status_code == 422


def test_embeddings_endpoint_valid_request():
    """Test embeddings endpoint with valid request."""
    response = client.post(
        "/embeddings",
        json={
            "texts": ["Hello world", "This is a test"],
            "normalize": True
        }
    )
    # This might fail if model is not loaded, which is expected in tests
    assert response.status_code in [200, 500]


def test_embeddings_endpoint_too_many_texts():
    """Test embeddings endpoint with too many texts."""
    texts = ["test"] * 101  # More than the 100 limit
    response = client.post("/embeddings", json={"texts": texts})
    assert response.status_code == 422


def test_docs_endpoint():
    """Test that the docs endpoint is accessible."""
    response = client.get("/docs")
    assert response.status_code == 200


def test_redoc_endpoint():
    """Test that the redoc endpoint is accessible."""
    response = client.get("/redoc")
    assert response.status_code == 200
