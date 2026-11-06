"""Build the complete search index and validation data for all claim files."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from app.database import (
    init_schema,
    replace_chunks,
    replace_documents,
    replace_findings,
    replace_mappings,
    upsert_claim,
)
from ingestion.classify_documents import classify_document, load_manifest
from ingestion.extract_metadata import extract_document_metadata, write_json
from ingestion.parse_documents import PdfParser, blocks_to_json
from rag.chunking import chunk_blocks
from rag.retrieval import build_index
from validation.check_inconsistencies import find_inconsistencies
from validation.check_missing_documents import find_missing_documents

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

DATA_ROOT = Path("data/claims")
STORAGE_ROOT = Path("storage")


def claim_id_from_manifest(manifest: dict[str, Any], folder_name: str) -> str:
    return manifest.get("claim_id") or manifest.get("case_id") or folder_name


def claim_type_from_manifest(manifest: dict[str, Any]) -> str:
    return manifest.get("claim_type") or "automobile_collision"


def process_claim(claim_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = load_manifest(claim_dir)
    claim_id = claim_id_from_manifest(manifest, claim_dir.name)
    claim_type = claim_type_from_manifest(manifest)
    LOGGER.info("Processing claim %s", claim_id)
    upsert_claim(claim_id, claim_type, str(claim_dir))

    parser = PdfParser()
    all_chunks: list[dict[str, Any]] = []
    document_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []

    for pdf_path in sorted(claim_dir.glob("*.pdf")):
        LOGGER.info("  Parsing %s", pdf_path.name)
        document = parser.convert(pdf_path)
        raw_blocks = parser.blocks_from_document(
            document=document,
            claim_id=claim_id,
            file_name=pdf_path.name,
            doc_type="unknown",
        )
        block_dicts = blocks_to_json(raw_blocks)
        full_text = "\n".join(block["text"] for block in block_dicts)
        doc_type = classify_document(pdf_path.name, full_text)

        # Rewrite block doc_type after classification.
        for block in block_dicts:
            block["doc_type"] = doc_type

        # Persist parsed blocks for auditing/debugging.
        parsed_path = STORAGE_ROOT / "parsed" / claim_id / f"{pdf_path.stem}.json"
        write_json(parsed_path, block_dicts)

        # Metadata.
        metadata = extract_document_metadata(claim_id, pdf_path.name, doc_type, block_dicts)
        metadata_path = STORAGE_ROOT / "metadata" / claim_id / f"{pdf_path.stem}.json"
        write_json(metadata_path, metadata)

        # Chunks.
        doc_chunks = chunk_blocks(claim_id, pdf_path.name, doc_type, block_dicts)
        all_chunks.extend(doc_chunks)

        document_rows.append(
            {
                "file_name": pdf_path.name,
                "doc_type": doc_type,
                "page_count": metadata["page_count"],
                "char_count": len(full_text),
                "metadata": metadata["metadata"],
            }
        )
        mapping_rows.append({"file_name": pdf_path.name, "canonical_type": doc_type})

    replace_documents(claim_id, document_rows)
    replace_mappings(claim_id, mapping_rows)
    replace_chunks(claim_id, all_chunks)

    # Validation results.
    missing_result = find_missing_documents(claim_id, claim_dir, manifest)
    inconsistencies = find_inconsistencies(claim_id, claim_dir, manifest)
    findings = inconsistencies + [
        {
            "category": "completeness",
            "severity": "high" if missing_result["missing"] else "low",
            "field": "missing_documents",
            "summary": (
                "Pièces manquantes : " + ", ".join(missing_result["missing"])
                if missing_result["missing"]
                else "Aucune pièce attendue manquante selon le manifeste."
            ),
            "details": missing_result,
        }
    ]
    replace_findings(claim_id, findings)

    return document_rows, all_chunks


def main() -> None:
    init_schema()
    all_chunks: list[dict[str, Any]] = []
    for claim_dir in sorted(DATA_ROOT.iterdir()):
        if not claim_dir.is_dir():
            continue
        if not (claim_dir / "manifest.json").exists():
            continue
        _, chunks = process_claim(claim_dir)
        all_chunks.extend(chunks)

    if not all_chunks:
        raise RuntimeError("No chunks were produced. Check data/claims.")

    index, persisted_chunks = build_index(all_chunks)
    LOGGER.info("Built FAISS index with %d chunks.", index.ntotal)
    LOGGER.info("Persisted %d chunks to storage/vectorstore/chunks.json", len(persisted_chunks))


if __name__ == "__main__":
    main()