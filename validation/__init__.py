"""Validation helpers for insurance claim files."""


from validation.case_state import (
    calculate_case_state,
    compute_case_state,
    determine_case_state,
)

# Convenient short alias for callers that expect a ``case_state`` callable.
case_state = compute_case_state

__all__ = [
    "calculate_case_state",
    "case_state",
    "compute_case_state",
    "determine_case_state",
]
