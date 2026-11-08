"""Derive a deterministic case state from validation results."""


import json
from pathlib import Path
from typing import Any


def _classification_status(document: Any) -> str | None:
    """Return the classification status stored in a document record, if any."""
    if not isinstance(document, dict):
        return None

    metadata = document.get("metadata")
    if metadata is None and document.get("metadata_json") is not None:
        try:
            metadata = json.loads(document.get("metadata_json"))
        except (TypeError, ValueError, json.JSONDecodeError):
            metadata = {}

    if not isinstance(metadata, dict):
        return None

    classification = metadata.get("classification")
    if not isinstance(classification, dict):
        return None

    status = classification.get("classification_status")
    return status if isinstance(status, str) else None


def _has_ambiguous_documents(documents: Any) -> bool:
    """Whether at least one document has a non-clear classification status."""
    if not documents:
        return False

    # A plain list of statuses is also convenient for callers and tests.
    if isinstance(documents, list) and all(isinstance(item, str) for item in documents):
        return any(status != "clear" for status in documents)

    for document in documents:
        status = _classification_status(document)
        if status is not None and status != "clear":
            return True
    return False


def _missing_list_from_value(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        if isinstance(value.get("missing"), list):
            return [str(item) for item in value["missing"]]
        if isinstance(value.get("missing_documents"), list):
            return [str(item) for item in value["missing_documents"]]
        return None
    if isinstance(value, list):
        return [str(item) for item in value]
    return None


def _inconsistencies_from_value(value: Any) -> list[dict[str, Any]] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        for key in ("inconsistencies", "findings", "conflicts"):
            if isinstance(value.get(key), list):
                return [item for item in value[key] if isinstance(item, dict)]
        return None
    if isinstance(value, list):
        if all(isinstance(item, dict) for item in value):
            return list(value)
        return None
    return None


def _resolve_claim_context(claim_id: Any, directory: Any, manifest: Any) -> tuple[list[str], list[dict[str, Any]], Any]:
    """Load missing documents and inconsistencies from an existing claim."""
    from validation.check_inconsistencies import find_inconsistencies
    from validation.check_missing_documents import find_missing_documents

    directory = Path(directory) if directory is not None else None
    if directory is None:
        directory = Path("data/claims") / str(claim_id)

    missing = find_missing_documents(str(claim_id), directory, manifest)
    inconsistencies = find_inconsistencies(str(claim_id), directory, manifest)

    missing_documents = list(missing.get("missing", []))
    inconsistency_list = list(inconsistencies) if isinstance(inconsistencies, list) else []

    documents = None
    try:
        from app.database import get_documents

        documents = get_documents(str(claim_id))
    except Exception:  # pragma: no cover - database is optional in standalone use
        documents = None

    return missing_documents, inconsistency_list, documents


def _derive_state(
    missing_documents: list[str],
    inconsistencies: list[dict[str, Any]],
    ambiguous_documents: bool,
) -> str:
    has_missing = bool(missing_documents)
    has_conflicts = bool(inconsistencies)

    if ambiguous_documents:
        return "REVIEW_REQUIRED"
    if has_missing and has_conflicts:
        return "INCOMPLETE_AND_CONFLICTING"
    if has_missing:
        return "INCOMPLETE"
    if has_conflicts:
        return "CONFLICTING"
    return "COMPLETE"


def compute_case_state(*args: Any, **kwargs: Any) -> str:
    """Compute a deterministic case state.

    The function accepts several convenient calling styles:

    - ``compute_case_state(missing_documents, inconsistencies, documents=None)``
    - ``compute_case_state(missing_documents=[...], inconsistencies=[...])``
    - a validation payload: ``compute_case_state(payload)``
    - a claim context: ``compute_case_state(claim_id, directory, manifest)``
    """
    # Normalise keyword arguments.
    missing_arg = kwargs.pop("missing_documents", None)
    if missing_arg is None:
        missing_arg = kwargs.pop("missing", None)
    if missing_arg is None:
        missing_arg = kwargs.pop("missing_docs", None)

    inconsistencies_arg = kwargs.pop("inconsistencies", None)
    if inconsistencies_arg is None:
        inconsistencies_arg = kwargs.pop("findings", None)
    if inconsistencies_arg is None:
        inconsistencies_arg = kwargs.pop("conflicts", None)

    documents_arg = kwargs.pop("documents", None)
    if documents_arg is None:
        documents_arg = kwargs.pop("docs", None)

    classifications_arg = kwargs.pop("classifications", None)
    if classifications_arg is None:
        classifications_arg = kwargs.pop("document_classifications", None)

    manifest_arg = kwargs.pop("manifest", None)
    claim_id_arg = kwargs.pop("claim_id", None)
    directory_arg = kwargs.pop("directory", None)
    if directory_arg is None:
        directory_arg = kwargs.pop("claim_dir", None)
    if directory_arg is None:
        directory_arg = kwargs.pop("dir", None)

    # Handle a single validation payload.
    if len(args) == 1 and isinstance(args[0], dict) and (
        "missing_documents" in args[0] or "missing" in args[0] or "inconsistencies" in args[0]
    ):
        if missing_arg is None:
            missing_arg = args[0]
        if inconsistencies_arg is None and "inconsistencies" in args[0]:
            inconsistencies_arg = args[0].get("inconsistencies")
        if documents_arg is None and "documents" in args[0]:
            documents_arg = args[0].get("documents")
        args = ()

    # Claim-context call: compute() / (claim_id, directory, manifest).
    if len(args) >= 3 and isinstance(args[2], dict) and (
        "missing_documents" in args[2]
        or "received_documents" in args[2]
        or "expected_documents" in args[2]
        or "recommended_case_state" in args[2]
    ):
        claim_id_arg = claim_id_arg if claim_id_arg is not None else args[0]
        directory_arg = directory_arg if directory_arg is not None else args[1]
        manifest_arg = manifest_arg if manifest_arg is not None else args[2]
        args = ()

    # Two positional arguments: either (missing, inconsistencies) or
    # (claim_id, manifest).
    if len(args) == 2:
        if isinstance(args[1], dict) and (
            "missing_documents" in args[1]
            or "received_documents" in args[1]
            or "expected_documents" in args[1]
            or "recommended_case_state" in args[1]
        ) and not isinstance(args[0], (list, tuple, dict)):
            claim_id_arg = claim_id_arg if claim_id_arg is not None else args[0]
            manifest_arg = manifest_arg if manifest_arg is not None else args[1]
            args = ()
        else:
            if missing_arg is None:
                missing_arg = args[0]
            if inconsistencies_arg is None:
                inconsistencies_arg = args[1]
            if len(args) > 2:
                documents_arg = documents_arg if documents_arg is not None else args[2]
            args = ()

    # Three positional arguments without a manifest at index 2 may be
    # (missing, inconsistencies, documents).
    if len(args) >= 2 and (missing_arg is None or inconsistencies_arg is None):
        if missing_arg is None:
            missing_arg = args[0]
        if inconsistencies_arg is None and len(args) > 1:
            inconsistencies_arg = args[1]
        if documents_arg is None and len(args) > 2:
            documents_arg = args[2]
        args = ()

    # Resolve a claim context when the three context values are available.
    if claim_id_arg is not None and manifest_arg is not None and missing_arg is None and inconsistencies_arg is None:
        missing_documents, inconsistencies, documents = _resolve_claim_context(
            claim_id_arg, directory_arg, manifest_arg
        )
        missing_arg = missing_documents
        inconsistencies_arg = inconsistencies
        if documents_arg is None:
            documents_arg = documents
    elif claim_id_arg is not None and manifest_arg is not None and missing_arg is not None and inconsistencies_arg is None:
        # Only missing was supplied; load inconsistencies from the claim.
        _, inconsistencies, documents = _resolve_claim_context(claim_id_arg, directory_arg, manifest_arg)
        inconsistencies_arg = inconsistencies
        if documents_arg is None:
            documents_arg = documents

    missing_documents = _missing_list_from_value(missing_arg) or []
    inconsistencies = _inconsistencies_from_value(inconsistencies_arg) or []

    if classifications_arg is not None:
        ambiguous_documents = _has_ambiguous_documents(classifications_arg)
    else:
        ambiguous_documents = _has_ambiguous_documents(documents_arg)
    # Handle the UNKNOWN case: no inputs, no manifest, and no claim context.
    if (
        missing_arg is None
        and inconsistencies_arg is None
        and documents_arg is None
        and classifications_arg is None
        and claim_id_arg is None
        and manifest_arg is None
        and not missing_documents
        and not inconsistencies
    ):
        return "UNKNOWN"

    return _derive_state(missing_documents, inconsistencies, ambiguous_documents)


# Backward/variant-friendly aliases (the calculation is always the same).
determine_case_state = compute_case_state
calculate_case_state = compute_case_state
case_state = compute_case_state
