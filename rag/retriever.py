"""Retrieval: vector search + optional MMR diversification + score cutoff.

Why MMR? With a plain top-k you often get five near-duplicate chunks from the
same page. Maximal Marginal Relevance re-ranks candidates so they are both
relevant to the question and different from each other, which improves the
answer and reduces wasted context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import config, embeddings, store


@dataclass
class RetrievedChunk:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    embedding: list[float] | None = None

    def citation(self, number: int) -> str:
        bits = [self.metadata.get("source", "unknown")]
        if self.metadata.get("page"):
            bits.append(f"p.{self.metadata['page']}")
        if self.metadata.get("section"):
            bits.append(str(self.metadata["section"]))
        if self.metadata.get("row_start"):
            bits.append(f"rows {self.metadata['row_start']}-{self.metadata.get('row_end')}")
        return f"[{number}] " + ", ".join(bits)


def _mmr(
    query_vector: list[float],
    candidates: list[RetrievedChunk],
    k: int,
    lam: float,
) -> list[RetrievedChunk]:
    if len(candidates) <= 1:
        return candidates[:k]

    query = np.asarray(query_vector, dtype="float32")
    matrix = np.asarray([c.embedding for c in candidates], dtype="float32")
    # normalise defensively - embeddings should already be normalised
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms
    query = query / (np.linalg.norm(query) or 1.0)

    relevance = matrix @ query
    selected: list[int] = []
    remaining = list(range(len(candidates)))

    while remaining and len(selected) < k:
        if not selected:
            best = max(remaining, key=lambda i: relevance[i])
        else:
            selected_matrix = matrix[selected]
            best = None
            best_value = -1e9
            for i in remaining:
                redundancy = float(np.max(selected_matrix @ matrix[i]))
                value = lam * float(relevance[i]) - (1 - lam) * redundancy
                if value > best_value:
                    best_value = value
                    best = i
        selected.append(best)
        remaining.remove(best)

    return [candidates[i] for i in selected]


def retrieve(
    question: str,
    k: int | None = None,
    min_score: float | None = None,
    use_mmr: bool | None = None,
    lam: float | None = None,
    where: dict | None = None,
) -> list[RetrievedChunk]:
    k = k or config.TOP_K
    min_score = config.MIN_SCORE if min_score is None else min_score
    use_mmr = config.USE_MMR if use_mmr is None else use_mmr
    lam = config.MMR_LAMBDA if lam is None else lam

    vector = embeddings.embed_query(question)
    # pull a wider candidate pool when diversifying
    pool = max(k * 4, k) if use_mmr else k
    hits = store.query(vector, k=pool, where=where, include_embeddings=use_mmr)

    chunks = [
        RetrievedChunk(
            id=hit["id"],
            text=hit["text"],
            metadata=hit["metadata"],
            score=hit["score"],
            embedding=hit.get("embedding"),
        )
        for hit in hits
        if hit["score"] >= min_score
    ]

    if not use_mmr:
        return chunks[:k]
    if any(c.embedding is None for c in chunks):
        return chunks[:k]
    return _mmr(vector, chunks, k, lam)
