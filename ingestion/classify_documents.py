"""Document classification for insurance claim files."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# Canonical document types shared by manifests and validations.
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
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def classify_by_filename(file_name: str) -> str | None:
    normalized = _normalize_name(file_name)
    for hint, doc_type in FILENAME_HINTS:
        if hint in normalized:
            return doc_type
    return None


def classify_by_text(text: str) -> str:
    lowered = text.lower()
    best_type = "unknown"
    best_score = 0
    for doc_type, keywords in CANONICAL_TYPE_KEYWORDS:
        score = sum(1 for keyword in keywords if keyword in lowered)
        if score > best_score:
            best_score = score
            best_type = doc_type
    return best_type


def classify_document(file_name: str, text: str = "") -> str:
    """Classify a document using filename first, then content fallback."""
    by_name = classify_by_filename(file_name)
    if by_name:
        return by_name
    return classify_by_text(text)


def build_claim_inventory(claim_id: str, claim_dir: Path) -> list[dict[str, Any]]:
    """Return one inventory row per received PDF in a claim folder."""
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