"""Page-aware chunking for retrieved citations."""

from __future__ import annotations

import re
from typing import Any

DEFAULT_TARGET_CHARS = 900
DEFAULT_MAX_CHARS = 1200


def _split_long_text(text: str, max_chars: int) -> list[str]:
    """Split an oversized block into readable pieces."""
    if len(text) <= max_chars:
        return [text]
    pieces: list[str] = []
    current = ""
    # Prefer paragraph boundaries, then sentences, then hard cuts.
    separators = [r"\n\n", r"(?<=[.!?])\s+"]
    units = [text]
    for separator in separators:
        next_units: list[str] = []
        for unit in units:
            if len(unit) <= max_chars:
                next_units.append(unit)
            else:
                next_units.extend(re.split(separator, unit))
        units = [unit for unit in next_units if unit.strip()]
        if all(len(unit) <= max_chars for unit in units):
            break

    for unit in units:
        if len(current) + len(unit) + 1 <= max_chars:
            current = (current + " " + unit).strip()
        else:
            if current:
                pieces.append(current)
            current = unit.strip()
    if current:
        pieces.append(current)
    return pieces or [text[:max_chars]]


def chunk_blocks(
    claim_id: str,
    file_name: str,
    doc_type: str,
    blocks: list[dict[str, Any]],
    *,
    target_chars: int = DEFAULT_TARGET_CHARS,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> list[dict[str, Any]]:
    """Build page-aware chunks from ordered Docling blocks."""
    chunks: list[dict[str, Any]] = []
    current_texts: list[str] = []
    current_pages: set[int] = set()
    current_headings: list[str] = []

    def flush() -> None:
        nonlocal current_texts, current_pages, current_headings
        text = "\n\n".join(part for part in current_texts if part.strip()).strip()
        if not text:
            current_texts = []
            current_pages = set()
            return
        pages = sorted(current_pages) if current_pages else [None]
        # Prefix heading for retrieval context, without duplicating it as content.
        heading_prefix = current_headings[-1] if current_headings else ""
        chunk_text = text
        if heading_prefix and heading_prefix not in text[:120]:
            chunk_text = f"{heading_prefix}\n{text}"
        chunk_id = f"{claim_id}_{file_name}_{len(chunks) + 1:04d}"
        chunks.append(
            {
                "chunk_id": chunk_id,
                "claim_id": claim_id,
                "file_name": file_name,
                "doc_type": doc_type,
                "page_start": pages[0],
                "page_end": pages[-1],
                "heading": heading_prefix,
                "text": chunk_text,
            }
        )
        current_texts = []
        current_pages = set()

    for block in blocks:
        text = (block.get("text") or "").strip()
        if not text:
            continue
        page = block.get("page")
        block_type = block.get("block_type", "")
        heading = block.get("heading", "") or ""

        if heading and heading not in current_headings:
            current_headings.append(heading)

        # Headings and short labels can be absorbed; long blocks must be split.
        for piece in _split_long_text(text, max_chars):
            estimated = sum(len(item) for item in current_texts) + len(piece)
            page_changed = bool(current_pages and page not in current_pages)
            if current_texts and (estimated > target_chars or page_changed):
                flush()
            current_texts.append(piece)
            if page is not None:
                current_pages.add(page)

    flush()
    return chunks