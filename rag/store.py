"""ChromaDB persistence layer.

A single persistent collection holds every chunk. We always pass embeddings
computed by the Gemini API, so Chroma never tries to embed anything itself.
Metadata filters let us re-ingest or delete one document at a time.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import chromadb

from . import config
from .chunking import Chunk

_client: chromadb.ClientAPI | None = None
_collection = None


def _get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return _client


def get_collection():
    global _collection
    if _collection is not None:
        return _collection

    client = _get_client()
    try:
        _collection = client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            configuration={"hnsw": {"space": "cosine"}},
            embedding_function=None,
        )
    except (TypeError, ValueError):
        _collection = client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def count() -> int:
    try:
        return get_collection().count()
    except Exception:
        return 0


def add_chunks(chunks: list[Chunk], embeddings: list[list[float]]) -> None:
    if not chunks:
        return
    collection = get_collection()
    batch = 500
    for start in range(0, len(chunks), batch):
        slice_chunks = chunks[start : start + batch]
        slice_embeddings = embeddings[start : start + batch]
        collection.add(
            ids=[c.id for c in slice_chunks],
            documents=[c.text for c in slice_chunks],
            embeddings=slice_embeddings,
            metadatas=[c.metadata for c in slice_chunks],
        )


def delete_source(source: str) -> None:
    collection = get_collection()
    try:
        collection.delete(where={"source": source})
    except Exception:
        existing = collection.get(where={"source": source}, include=[])
        if existing.get("ids"):
            collection.delete(ids=existing["ids"])


def list_documents() -> list[dict[str, Any]]:
    collection = get_collection()
    if collection.count() == 0:
        return []
    data = collection.get(include=["metadatas"])
    grouped: dict[str, dict[str, Any]] = {}
    for meta in data.get("metadatas") or []:
        if not meta:
            continue
        source = meta.get("source") or "unknown"
        entry = grouped.setdefault(
            source,
            {
                "source": source,
                "chunks": 0,
                "pages": set(),
                "path": meta.get("path", ""),
            },
        )
        entry["chunks"] += 1
        if meta.get("page"):
            entry["pages"].add(int(meta["page"]))

    documents = []
    for entry in grouped.values():
        entry["pages"] = len(entry["pages"])
        documents.append(entry)
    return sorted(documents, key=lambda d: d["source"].lower())


def query(
    embedding: list[float],
    k: int = 6,
    where: dict | None = None,
    include_embeddings: bool = False,
) -> list[dict[str, Any]]:
    collection = get_collection()
    if collection.count() == 0:
        return []

    include = ["documents", "metadatas", "distances"]
    if include_embeddings:
        include.append("embeddings")

    result = collection.query(
        query_embeddings=[embedding],
        n_results=min(k, max(1, collection.count())),
        where=where,
        include=include,
    )

    hits: list[dict[str, Any]] = []
    ids = (result.get("ids") or [[]])[0]
    docs = (result.get("documents") or [[]])[0]
    metas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    embeddings = (result.get("embeddings") or [[]])[0] if include_embeddings else []

    for index, chunk_id in enumerate(ids):
        distance = float(distances[index]) if index < len(distances) else 1.0
        hits.append(
            {
                "id": chunk_id,
                "text": docs[index] if index < len(docs) else "",
                "metadata": metas[index] or {} if index < len(metas) else {},
                "score": max(0.0, 1.0 - distance),  # cosine distance -> similarity
                "embedding": embeddings[index] if index < len(embeddings) else None,
            }
        )
    return hits


def reset() -> None:
    global _collection
    client = _get_client()
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass
    _collection = None
    get_collection()


def source_hashes() -> dict[str, list[str]]:
    """Chunk ids per source - used by ingest to skip unchanged files."""
    collection = get_collection()
    if collection.count() == 0:
        return {}
    data = collection.get(include=["metadatas"])
    grouped: dict[str, list[str]] = defaultdict(list)
    for chunk_id, meta in zip(data.get("ids") or [], data.get("metadatas") or []):
        if meta:
            grouped[meta.get("source", "unknown")].append(chunk_id)
    return dict(grouped)


def iter_metadata(where: dict | None = None) -> Iterable[dict]:
    collection = get_collection()
    data = collection.get(where=where, include=["metadatas"])
    yield from (data.get("metadatas") or [])
