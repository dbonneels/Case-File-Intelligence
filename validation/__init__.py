"""Validation helpers for insurance claim files."""


from validation.case_state import (
    calculate_case_state,
    compute_case_state,
    determine_case_state,
)
from validation.check_missing_documents import (
    EXPECTED_DOCUMENT_TYPES,
    FACULTATIVE_DOCUMENT_TYPES,
    OPTIONAL_DOCUMENTS,
    OPTIONAL_DOCUMENT_TYPES,
    REQUIRED_DOCUMENT_TYPES,
    claim_type_for,
    expected_document_types_for,
    missing_expected_documents,
    optional_document_types_for,
)

# Convenient short alias for callers that expect a ``case_state`` callable.
case_state = compute_case_state

__all__ = [
    "calculate_case_state",
    "case_state",
    "claim_type_for",
    "compute_case_state",
    "determine_case_state",
    "expected_document_types_for",
    "EXPECTED_DOCUMENT_TYPES",
    "FACULTATIVE_DOCUMENT_TYPES",
    "missing_expected_documents",
    "optional_document_types_for",
    "OPTIONAL_DOCUMENTS",
    "OPTIONAL_DOCUMENT_TYPES",
    "REQUIRED_DOCUMENT_TYPES",
]
