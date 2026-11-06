"""Completeness checks for insurance claim files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ingestion.classify_documents import load_manifest

# Additional pieces that AUTO manifest lists as missing but which are not in
# its received document list. Canonical names are used for reporting.
AUTO_MISSING_CANONICAL = {
    "constat_amiable_officiel_signe": "signed_official_constat",
    "photographies_originales": "original_photos",
    "conditions_generales_CG-AUTO-2025-v2": "general_policy_conditions",
    "ordre_reparation_et_facture_finale": "repair_order_and_final_invoice",
}


def _manifest_expected(manifest: dict[str, Any]) -> list[str]:
    expected = list(manifest.get("expected_documents", []))
    # AUTO manifest does not expose expected_documents; derive from received
    # types plus declared missing pieces.
    if not expected:
        received_count = len(manifest.get("received_documents", []))
        expected = []
        # Received implied types are unknown here, but the canonical missing list
        # is explicit. We keep the manifest's missing names as expected keys.
        for missing_name in manifest.get("missing_documents", []):
            expected.append(missing_name)
        # Preserve a stable received baseline by number; not a canonical type.
        expected = [f"received_{index}" for index in range(received_count)] + expected
    return expected


def _received_types(claim_id: str, claim_dir: Path) -> set[str]:
    # Read parsed document classifications from SQLite if available.
    try:
        from app.database import get_mappings

        mappings = get_mappings(claim_id)
        return {row["canonical_type"] for row in mappings}
    except Exception:
        return set()


def find_missing_documents(claim_id: str, claim_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    manifest_expected = manifest.get("expected_documents", [])
    received_types = _received_types(claim_id, claim_dir)

    # AUTO manifest lists missing pieces explicitly instead of expected_documents.
    if not manifest_expected:
        missing = list(manifest.get("missing_documents", []))
    else:
        missing = sorted(set(manifest_expected) - received_types)

    return {
        "claim_id": claim_id,
        "expected": manifest_expected or sorted(set(received_types)),
        "received_types": sorted(received_types),
        "missing": missing,
        "received_files": sorted(path.name for path in claim_dir.glob("*.pdf")),
        "missing_from_manifest": manifest.get("missing_documents", []),
    }