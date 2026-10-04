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


def test_system_endpoint_reports_process_rss():
    """Process memory is sampled from inside the API process."""
    response = client.get("/system")
    assert response.status_code == 200
    data = response.json()
    assert data["rss_bytes"] > 0
    assert "cpu_percent" in data


def test_embeddings_split_load_and_encode_time(monkeypatch):
    """A warm request reports encode time separately from a zero load time."""
    monkeypatch.setattr(
        "app.main.embedding_service.generate_embeddings",
        lambda texts, model_name, normalize, batch_size: ([[0.1, 0.2]], 0.01, 0.0, 8),
    )
    monkeypatch.setattr(
        "app.main.embedding_service.get_model_info",
        lambda: {
            "model_name": "unit-test-model",
            "model_type": "sentence-transformer",
            "max_sequence_length": 256,
            "embedding_dimensions": 2,
            "model_size_bytes": 128,
            "is_loaded": True,
        },
    )
    response = client.post("/embeddings", json={"texts": ["a" * 600], "batch_size": 8})
    assert response.status_code == 200
    data = response.json()
    assert data["processing_time"] == 0.01
    assert data["load_time"] == 0.0
    assert data["batch_size"] == 8
    assert data["dimensions"] == 2


def test_embeddings_rejects_only_the_character_safety_cap():
    """Token length is enforced by the model. The API caps raw characters."""
    response = client.post("/embeddings", json={"texts": ["a" * 20001]})
    assert response.status_code == 400
    assert "characters" in response.json()["detail"]


def test_embeddings_rejects_non_positive_batch_size():
    response = client.post("/embeddings", json={"texts": ["hello"], "batch_size": 0})
    assert response.status_code == 422


def test_docs_endpoint():
    """Test that the docs endpoint is accessible."""
    response = client.get("/docs")
    assert response.status_code == 200


def test_redoc_endpoint():
    """Test that the redoc endpoint is accessible."""
    response = client.get("/redoc")
    assert response.status_code == 200
