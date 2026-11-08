"""Completeness checks for insurance claim files."""


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
EXPECTED_DOCUMENT_TYPES = {
    "auto_accident": {
        "declaration",
        "insurance_policy",
        "accident_report",
        "expert_report",
        "repair_quote",
        "final_invoice",
        "repair_order",
    },
    "home_water_damage": {
        "declaration",
        "insurance_policy",
        "expert_report",
        "repair_quote",
        "final_invoice",
    },
}

OPTIONAL_DOCUMENT_TYPES = {
    "auto_accident": {
        "original_photos",
    },
    "home_water_damage": {
        "damage_photos",
    },
}

REQUIRED_DOCUMENT_TYPES = EXPECTED_DOCUMENT_TYPES
FACULTATIVE_DOCUMENT_TYPES = OPTIONAL_DOCUMENT_TYPES
OPTIONAL_DOCUMENTS = OPTIONAL_DOCUMENT_TYPES


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

    optional_types = optional_document_types_for(claim_id=claim_id, manifest=manifest)
    optional_missing = sorted(optional_types - received_types)

    return {
        "claim_id": claim_id,
        "expected": manifest_expected or sorted(set(received_types)),
        "received_types": sorted(received_types),
        "missing": missing,
        "optional": sorted(optional_types),
        "optional_documents": sorted(optional_types),
        "optional_types": sorted(optional_types),
        "optional_missing": optional_missing,
        "optional_missing_documents": optional_missing,
        "received_files": sorted(path.name for path in claim_dir.glob("*.pdf")),
        "missing_from_manifest": manifest.get("missing_documents", []),
    }

def claim_type_for(claim_id: Any = None, manifest: dict[str, Any] | None = None) -> str | None:
    """Resolve a canonical claim type from a claim id and/or manifest."""
    if manifest:
        claim_type = manifest.get("claim_type")
        if claim_type == "water_damage":
            return "home_water_damage"
        if claim_type in EXPECTED_DOCUMENT_TYPES:
            return str(claim_type)
    if claim_id is not None:
        claim_id_text = str(claim_id).upper()
        if claim_id_text.startswith("AUTO"):
            return "auto_accident"
        if claim_id_text.startswith(("SIN", "HOME", "HAB")):
            return "home_water_damage"
    if manifest and manifest.get("claim_type") is not None:
        return str(manifest.get("claim_type"))
    return None


def expected_document_types_for(
    claim_id: Any = None,
    manifest: dict[str, Any] | None = None,
    claim_type: str | None = None,
) -> set[str]:
    """Return the required document types for a claim."""
    resolved_claim_type = claim_type or claim_type_for(claim_id, manifest)
    if resolved_claim_type is None:
        return set()
    return set(EXPECTED_DOCUMENT_TYPES.get(resolved_claim_type, set()))


def optional_document_types_for(
    claim_id: Any = None,
    manifest: dict[str, Any] | None = None,
    claim_type: str | None = None,
) -> set[str]:
    """Return the useful but optional document types for a claim."""
    resolved_claim_type = claim_type or claim_type_for(claim_id, manifest)
    if resolved_claim_type is None:
        return set()
    return set(OPTIONAL_DOCUMENT_TYPES.get(resolved_claim_type, set()))


def missing_expected_documents(
    received_types: Any,
    claim_id: Any = None,
    manifest: dict[str, Any] | None = None,
    claim_type: str | None = None,
) -> list[str]:
    """Return required document types not present among ``received_types``."""
    received = set(received_types or [])
    expected = expected_document_types_for(claim_id=claim_id, manifest=manifest, claim_type=claim_type)
    return sorted(expected - received)
