"""Build the complete search index and validation data for all claim files.

Ce script automatise les étapes suivantes :
1. Découvrir les dossiers de sinistres.
2. Lire les documents PDF.
3. Classifier les documents.
4. Extraire les métadonnées.
5. Découper les documents en chunks.
6. Enregistrer les données dans la base de données.
7. Détecter les incohérences.
8. Détecter les documents manquants.
9. Construire l'index FAISS.
"""

# imports de Python
import logging
from pathlib import Path
from typing import Any

# import de mes propres modules
from app.database import (
    init_schema,
    replace_chunks,
    replace_documents,
    replace_findings,
    replace_mappings,
    upsert_claim,
)
from rag.chunking import chunk_blocks
from rag.retrieval import build_index
from validation.check_inconsistencies import find_inconsistencies
from validation.check_missing_documents import find_missing_documents

from ingestion.classify_documents import classify_document, load_manifest
from ingestion.extract_metadata import extract_document_metadata, write_json
from ingestion.parse_documents import PdfParser, blocks_to_json

"""L'application est organisée en trois grandes parties :
- ingestion : ingestion et extraction documentaire.
- rag : préparation des données pour la recherche vectorielle.
- validation : contrôle de cohérence et de complétude."""


LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Location of source claim folders and local storage artifacts.
DATA_ROOT = Path("data/claims")  # Répertoire contenant les dossiers d'assurance.
STORAGE_ROOT = Path(
    "storage"
)  # Répertoire dans lequel l'application conserve les résultats intermédiaires.


def claim_id_from_manifest(manifest: dict[str, Any], folder_name: str) -> str:
    """Resolve the claim id from manifest fields, falling back to folder name.
    1. Chercher claim_id.
    2. Si absent ou vide, chercher case_id.
    3. Si absent ou vide, utiliser le nom du dossier."""
    return manifest.get("claim_id") or manifest.get("case_id") or folder_name


def claim_type_from_manifest(manifest: dict[str, Any]) -> str:
    """Resolve the claim type from manifest fields."""
    return manifest.get("claim_type")


def process_claim(claim_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Parse, classify, index, and validate all documents for one claim folder.
    Retourne deux listes :
    - document_rows : les informations sur les documents.
    - all_chunks : les fragments de texte destinés au RAG."""
    # The JSON manifest is read-only input describing expected/received documents.
    manifest = load_manifest(claim_dir)
    claim_id = claim_id_from_manifest(manifest, claim_dir.name)
    claim_type = claim_type_from_manifest(manifest)
    LOGGER.info("Processing claim %s", claim_id)
    upsert_claim(
        claim_id, claim_type, str(claim_dir)
    )  # on enregistre le sinistre dans la base de données.

    parser = (
        PdfParser()
    )  # crée une instance de la classe PdfParser et la stocke dans la variable parser
    all_chunks: list[dict[str, Any]] = []
    document_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    """On crée trois listes vides.
    all_chunks	Tous les chunks du sinistre
    document_rows	Les informations sur les PDF
    mapping_rows	Les correspondances entre noms et types de documents"""

    # Process each PDF in deterministic filename order.
    for pdf_path in sorted(claim_dir.glob("*.pdf")):
        LOGGER.info("  Parsing %s", pdf_path.name)
        document = parser.convert(pdf_path)
        raw_blocks = parser.blocks_from_document(
            document=document,
            claim_id=claim_id,
            file_name=pdf_path.name,
            doc_type="unknown",  # on ne connaît pas encore la classification définitive du document.
        )

        block_dicts = blocks_to_json(
            raw_blocks
        )  # conversion des blocs extraits en dictionnaires Python compatibles avec une sérialisation JSON.
        full_text = "\n".join(
            block["text"] for block in block_dicts
        )  # récupère le texte de chaque bloc et rassemble les textes dans une seule chaîne de caractères.
        doc_type = classify_document(pdf_path.name, full_text)

        # Rewrite block doc_type after classification.
        for block in block_dicts:
            block["doc_type"] = doc_type

        # Persist parsed blocks for auditing/debugging.
        parsed_path = STORAGE_ROOT / "parsed" / claim_id / f"{pdf_path.stem}.json"
        write_json(parsed_path, block_dicts)

        # Extract and persist lightweight document metadata.
        metadata = extract_document_metadata(
            claim_id, pdf_path.name, doc_type, block_dicts
        )
        metadata_path = STORAGE_ROOT / "metadata" / claim_id / f"{pdf_path.stem}.json"
        write_json(metadata_path, metadata)

        # Build vector-search chunks from the classified blocks.
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

    # Replace persisted claim data with the freshly computed rows.
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
    # Persist deterministic validation findings for later querying.
    replace_findings(claim_id, findings)

    return document_rows, all_chunks


def main() -> None:
    """Initialize storage schema and build data for every claim folder."""
    init_schema()
    all_chunks: list[dict[str, Any]] = []
    # Only directories containing a manifest.json are treated as claim folders.
    for claim_dir in sorted(DATA_ROOT.iterdir()):
        if not claim_dir.is_dir():
            continue
        if not (claim_dir / "manifest.json").exists():
            continue
        _, chunks = process_claim(claim_dir)
        all_chunks.extend(chunks)

    if not all_chunks:
        raise RuntimeError("No chunks were produced. Check data/claims.")

    # Build and persist the FAISS vector index from all collected chunks.
    index, persisted_chunks = build_index(all_chunks)
    LOGGER.info("Built FAISS index with %d chunks.", index.ntotal)
    LOGGER.info(
        "Persisted %d chunks to storage/vectorstore/chunks.json", len(persisted_chunks)
    )


if __name__ == "__main__":
    main()
