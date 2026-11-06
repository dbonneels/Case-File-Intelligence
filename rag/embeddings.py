"""Embedding helpers based on BAAI/bge-m3."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "BAAI/bge-m3"
_embedder: SentenceTransformer | None = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def embed_texts(texts: Sequence[str]) -> np.ndarray:
    """Embed and L2-normalize a list of texts."""
    if not texts:
        return np.zeros((0, 1024), dtype="float32")
    vectors = get_embedder().encode(
        list(texts),
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype="float32")