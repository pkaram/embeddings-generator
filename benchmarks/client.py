"""HTTP client for the embeddings API used by the measurement tools."""

import time

import httpx


# A first load downloads the model inside the request. all-mpnet-base-v2 took
# 612s on a cold cache, which is past a 10-minute read timeout.
DEFAULT_EMBED_TIMEOUT_S = 3600.0


class EmbeddingClient:
    """Blocking client. One instance is not shared across threads."""

    def __init__(self, base_url: str, timeout_s: float = DEFAULT_EMBED_TIMEOUT_S):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.http = httpx.Client(timeout=httpx.Timeout(30.0, connect=10.0))
        self._embed_timeout = httpx.Timeout(timeout_s, connect=10.0)

    def close(self) -> None:
        self.http.close()

    def health(self) -> dict:
        return self._json("GET", "/health")

    def system(self) -> dict:
        return self._json("GET", "/system")

    def model_info(self) -> dict:
        return self._json("GET", "/model/info")

    def unload(self) -> None:
        self._json("POST", "/model/unload")

    def embed(self, texts: list, model_name: str, batch_size: int, normalize: bool = True) -> dict:
        """POST /embeddings and attach client-observed latency as ``client_e2e_s``."""
        started = time.perf_counter()
        body = self._json(
            "POST",
            "/embeddings",
            json={
                "texts": texts,
                "model_name": model_name,
                "normalize": normalize,
                "batch_size": batch_size,
            },
            timeout=self._embed_timeout,
        )
        body["client_e2e_s"] = time.perf_counter() - started
        return body

    def _json(self, method: str, path: str, json: dict = None, timeout: httpx.Timeout = None) -> dict:
        response = self.http.request(method, f"{self.base_url}{path}", json=json, timeout=timeout)
        response.raise_for_status()
        return response.json()
