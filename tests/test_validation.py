from pathlib import Path

from validation.check_missing_documents import find_missing_documents
from validation.check_inconsistencies import find_inconsistencies


def test_water_claim_missing_and_conflicts() -> None:
    claim_id = "SIN-2026-0042"
    directory = Path("data/claims") / claim_id
    manifest = __import__("json").loads((directory / "manifest.json").read_text(encoding="utf-8"))
    missing = find_missing_documents(claim_id, directory, manifest)
    assert "final_repair_invoice" in missing["missing"]
    assert "general_policy_conditions" in missing["missing"]

    findings = find_inconsistencies(claim_id, directory, manifest)
    categories = {item["category"] for item in findings}
    assert "date_conflict" in categories
    assert "premium_status" in categories
    assert "amount_conflict" in categories


def test_auto_claim_conflicts() -> None:
    claim_id = "AUTO-2026-0087"
    directory = Path("data/claims") / claim_id
    manifest = __import__("json").loads((directory / "manifest.json").read_text(encoding="utf-8"))
    findings = find_inconsistencies(claim_id, directory, manifest)
    categories = {item["category"] for item in findings}
    assert "amount_conflict" in categories
    assert "statement_conflict" in categories
    assert "repair_scope" in categories
