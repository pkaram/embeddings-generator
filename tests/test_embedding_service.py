"""Tests for encode batching and the split load/encode clocks."""

import numpy as np

from app.embedding_service import embedding_service


class _FakeModel:
    def __init__(self):
        self.calls = []

    def encode(self, texts, batch_size, convert_to_tensor, normalize_embeddings, show_progress_bar):
        self.calls.append({
            "texts": list(texts),
            "batch_size": batch_size,
            "normalize_embeddings": normalize_embeddings,
        })
        return np.zeros((len(texts), 4))


def test_batch_size_is_forwarded_and_clamped():
    fake = _FakeModel()
    embedding_service.model = fake
    embedding_service.model_name = "unit-test-model"
    try:
        embeddings, encode_time, load_time, used = embedding_service.generate_embeddings(
            ["a", "b", "c"],
            model_name="unit-test-model",
            batch_size=4,
            normalize=True,
        )
        clamped_embeddings, _, clamped_load, clamped = embedding_service.generate_embeddings(
            ["a", "b", "c"],
            model_name="unit-test-model",
            batch_size=10_000,
        )
    finally:
        embedding_service.model = None
        embedding_service.model_name = None

    assert fake.calls[0]["texts"] == ["a", "b", "c"]
    assert fake.calls[0]["batch_size"] == 4
    assert fake.calls[0]["normalize_embeddings"] is True
    assert used == 4
    assert load_time == 0.0
    assert encode_time >= 0
    assert len(embeddings) == 3

    assert clamped == embedding_service.settings.max_batch_size
    assert fake.calls[1]["batch_size"] == embedding_service.settings.max_batch_size
    assert clamped_load == 0.0
    assert len(clamped_embeddings) == 3
