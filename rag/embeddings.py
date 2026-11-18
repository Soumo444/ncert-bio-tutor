"""Cloud embeddings via the Gemini API - nothing is downloaded to disk.

`gemini-embedding-001` is free-tier friendly and supports asymmetric retrieval:
documents are embedded with task type RETRIEVAL_DOCUMENT and the user question
with RETRIEVAL_QUERY, which measurably improves matching.

Requests are batched (default 30 texts) and paced below the free tier's
100-texts-per-minute cap, with exponential backoff on any remaining rate limit.
"""

from __future__ import annotations

import random
import time

import httpx
import numpy as np

from . import config

API_ROOT = "https://generativelanguage.googleapis.com/v1beta"
RETRY_STATUS = {429, 500, 502, 503, 504}


class EmbeddingError(RuntimeError):
    pass


def _require_key() -> str:
    if not config.GEMINI_API_KEY:
        raise EmbeddingError(
            "GEMINI_API_KEY is missing. Embeddings run in the cloud, so you need a "
            "free key from https://aistudio.google.com/apikey - paste it into .env"
        )
    return config.GEMINI_API_KEY


def _normalize(vector: list[float]) -> list[float]:
    array = np.asarray(vector, dtype="float32")
    norm = float(np.linalg.norm(array))
    if norm == 0:
        return vector
    return (array / norm).tolist()


def _batch(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def embed_texts(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    """Embed a list of strings. Returns one L2-normalised vector per input."""
    if not texts:
        return []

    api_key = _require_key()
    url = f"{API_ROOT}/models/{config.EMBED_MODEL}:batchEmbedContents"
    vectors: list[list[float]] = []

    with httpx.Client(timeout=config.REQUEST_TIMEOUT) as client:
        for batch in _batch(texts, min(config.EMBED_BATCH_SIZE, 100)):
            payload = {
                "requests": [
                    {
                        "model": f"models/{config.EMBED_MODEL}",
                        "content": {"parts": [{"text": text[:8000]}]},
                        "taskType": task_type,
                        "outputDimensionality": config.EMBED_DIM,
                    }
                    for text in batch
                ]
            }
            data = None
            for attempt in range(6):
                try:
                    response = client.post(url, params={"key": api_key}, json=payload)
                except httpx.HTTPError as exc:
                    if attempt == 5:
                        raise EmbeddingError(f"Embedding request failed: {exc}") from exc
                    time.sleep(min(60, 2**attempt) + random.random())
                    continue

                if response.status_code == 200:
                    data = response.json()
                    break
                if response.status_code in RETRY_STATUS and attempt < 5:
                    # 429 = per-minute text quota; wait long enough for the
                    # window to roll over before trying this batch again.
                    wait = 30 + attempt * 12 if response.status_code == 429 else min(30, 2**attempt)
                    time.sleep(wait + random.random())
                    continue
                raise EmbeddingError(
                    f"Embedding API error {response.status_code}: {response.text[:400]}"
                )

            if data is None:
                raise EmbeddingError("Embedding API did not return a response")
            embeddings = data.get("embeddings", [])
            if len(embeddings) != len(batch):
                raise EmbeddingError(
                    f"Embedding API returned {len(embeddings)} vectors for {len(batch)} inputs"
                )
            vectors.extend(_normalize(item["values"]) for item in embeddings)

    return vectors


def embed_query(question: str) -> list[float]:
    return embed_texts([question], task_type="RETRIEVAL_QUERY")[0]
