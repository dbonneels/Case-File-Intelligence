"""Combined FAISS vector store for all claim chunks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from rag.embeddings import EMBEDDING_MODEL, embed_texts

STORAGE_DIR = Path("storage")
VECTOR_DIR = STORAGE_DIR / "vectorstore"
INDEX_PATH = VECTOR_DIR / "claims.index"
CHUNKS_PATH = VECTOR_DIR / "chunks.json"


def ensure_storage() -> None:
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)


def build_index(chunks: list[dict[str, Any]]) -> tuple[Any, list[dict[str, Any]]]:
    """Build and persist a FAISS index from chunk dictionaries."""
    ensure_storage()
    texts = [chunk["text"] for chunk in chunks]
    vectors = embed_texts(texts)
    if vectors.shape[0] == 0:
        raise ValueError("Cannot build an empty FAISS index")
    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)
    faiss.write_index(index, str(INDEX_PATH))
    with CHUNKS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(chunks, handle, ensure_ascii=False, indent=2)
    return index, chunks


def load_index() -> tuple[Any | None, list[dict[str, Any]]]:
    if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
        return None, []
    index = faiss.read_index(str(INDEX_PATH))
    with CHUNKS_PATH.open("r", encoding="utf-8") as handle:
        chunks = json.load(handle)
    return index, chunks


def search(
    query: str,
    claim_id: str | None = None,
    k: int = 5,
) -> list[dict[str, Any]]:
    """Return the most relevant chunks for a question."""
    index, chunks = load_index()
    if index is None or not chunks:
        return []
    # Fetch a larger candidate set when filtering by claim.
    fetch_k = min(index.ntotal, k * 4 if claim_id else k)
    query_vector = embed_texts([query])
    scores, ids = index.search(query_vector, fetch_k)
    results: list[dict[str, Any]] = []
    for rank, idx in enumerate(ids[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        chunk = dict(chunks[idx])
        chunk["score"] = float(scores[0][rank])
        if claim_id and chunk.get("claim_id") != claim_id:
            continue
        results.append(chunk)
        if len(results) >= k:
            break
    return results


def embedding_model_name() -> str:
    return EMBEDDING_MODEL