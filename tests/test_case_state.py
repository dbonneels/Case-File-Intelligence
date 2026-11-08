from pathlib import Path

from validation.case_state import compute_case_state


def test_complete() -> None:
    assert compute_case_state([], []) == "COMPLETE"


def test_incomplete() -> None:
    assert compute_case_state(["missing_doc"], []) == "INCOMPLETE"


def test_conflicting() -> None:
    assert compute_case_state([], [{"category": "date_conflict", "severity": "high"}]) == "CONFLICTING"


def test_incomplete_and_conflicting() -> None:
    assert compute_case_state(["missing_doc"], [{"category": "date_conflict", "severity": "high"}]) == "INCOMPLETE_AND_CONFLICTING"


def test_review_required_for_ambiguous_classification() -> None:
    docs = [{"metadata": {"classification": {"classification_status": "ambiguous"}}}]
    assert compute_case_state([], [], documents=docs) == "REVIEW_REQUIRED"


def test_review_required_status_list() -> None:
    assert compute_case_state([], [], documents=["ambiguous"]) == "REVIEW_REQUIRED"


def test_unknown_without_inputs() -> None:
    assert compute_case_state() == "UNKNOWN"


def test_validation_payload() -> None:
    payload = {
        "missing_documents": ["a"],
        "inconsistencies": [{"category": "date_conflict", "severity": "high"}],
    }
    assert compute_case_state(payload) == "INCOMPLETE_AND_CONFLICTING"


def test_claim_context() -> None:
    claim_id = "SIN-2026-0042"
    directory = Path("data/claims") / claim_id
    import json

    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert compute_case_state(claim_id, directory, manifest) == "INCOMPLETE_AND_CONFLICTING"