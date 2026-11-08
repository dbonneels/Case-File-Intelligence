"""Document classification for insurance claim files."""

import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

CANONICAL_TYPE_KEYWORDS: list[tuple[str, list[str]]] = [
    ("declaration", ["déclaration", "declaration", "accident"]),
    ("insurance_policy", ["conditions particulières", "contrat", "police"]),
    ("premium_payment", ["cotisation", "relevé", "prime"]),
    ("expert_report", ["rapport d'expertise", "expertise", "expert"]),
    ("repair_quote", ["devis", "réparation"]),
    ("plumber_intervention", ["plombier", "intervention"]),
    ("correspondence", ["courriel", "échange", "email", "mail"]),
    ("internal_note", ["note interne", "note d'instruction", "instruction"]),
    ("insurance_attestation", ["attestation"]),
    ("accident_report", ["constat amiable", "constat"]),
    ("driver_statement", ["compte rendu", "conducteur"]),
    ("assistance", ["assistance", "remorquage"]),
]

FILENAME_HINTS: list[tuple[str, str]] = [
    ("declaration", "declaration"),
    ("conditions_particulieres", "insurance_policy"),
    ("releve_cotisations", "premium_payment"),
    ("rapport_expertise", "expert_report"),
    ("devis", "repair_quote"),
    ("compte_rendu_plombier", "plumber_intervention"),
    ("echanges_courriels", "correspondence"),
    ("note_instruction", "internal_note"),
    ("attestation", "insurance_attestation"),
    ("constat_amiable", "accident_report"),
    ("compte_rendu_conducteur", "driver_statement"),
    ("assistance_remorquage", "assistance"),
]


def _normalize_name(name: str) -> str:
    name = unicodedata.normalize("NFKD", name)
    name = "".join(char for char in name if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def classify_by_filename(file_name: str) -> str | None:
    normalized = _normalize_name(file_name)
    for hint, doc_type in FILENAME_HINTS:
        if hint in normalized:
            return doc_type
    return None


def classify_by_text(text: str) -> dict[str, Any]:
    lowered = text.lower()

    scores: dict[str, int] = {}
    for doc_type, keywords in CANONICAL_TYPE_KEYWORDS:
        score = sum(1 for keyword in keywords if keyword in lowered)
        if score > 0:
            scores[doc_type] = score

    if not scores:
        return {
            "doc_type": "unknown",
            "classification_status": "unknown",
            "alternative_types": [],
            "classification_scores": {},
        }

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_type, best_score = ranked[0]
    alternative_types = [doc_type for doc_type, score in ranked[1:] if score >= best_score - 1]
    status = "ambiguous" if alternative_types else "clear"

    if status == "ambiguous":
        LOGGER.warning(
            "Ambiguous classification: %s vs %s", best_type, ", ".join(alternative_types)
        )

    return {
        "doc_type": best_type,
        "classification_status": status,
        "alternative_types": alternative_types,
        "classification_scores": scores,
    }


def classify_document_details(file_name: str, text: str = "") -> dict[str, Any]:
    by_name = classify_by_filename(file_name)
    if by_name:
        return {
            "doc_type": by_name,
            "classification_status": "clear",
            "alternative_types": [],
            "classification_scores": {},
            "method": "filename",
        }
    result = classify_by_text(text)
    result["method"] = "text"
    return result


def classify_document(file_name: str, text: str = "") -> str:
    return classify_document_details(file_name, text)["doc_type"]


def build_claim_inventory(claim_id: str, claim_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pdf_path in sorted(claim_dir.glob("*.pdf")):
        rows.append(
            {
                "claim_id": claim_id,
                "file_name": pdf_path.name,
                "file_path": str(pdf_path),
            }
        )
    return rows


def load_manifest(claim_dir: Path) -> dict[str, Any]:
    manifest_path = claim_dir / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
