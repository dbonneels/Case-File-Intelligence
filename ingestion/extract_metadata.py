"""Lightweight structured metadata extraction from claim documents."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

DATE_PATTERNS = [
    re.compile(r"\b(\d{4}-\d{2}-\d{2})\b"),
    re.compile(r"\b(\d{2}[/.]\d{2}[/.]\d{4})\b"),
]
AMOUNT_PATTERN = re.compile(r"(\d[\d\s]*(?:,\d{1,2})?)\s*€")
POLICY_PATTERN = re.compile(r"\b([A-Z]{2,6}-?\d{4}-?\d{3,6})\b")
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_PATTERN = re.compile(r"\b(?:\+33|0)\s?(?:\d[\s.-]?){8,9}\d\b")


def _parse_date(value: str) -> str:
    value = value.strip()
    if re.fullmatch(r"\d{2}[/.]\d{2}[/.]\d{4}", value):
        day, month, year = re.split(r"[/.]", value)
        return f"{year}-{month}-{day}"
    return value


def extract_dates(text: str) -> list[str]:
    found: list[str] = []
    for pattern in DATE_PATTERNS:
        found.extend(match.group(1) for match in pattern.finditer(text))
    normalized = [_parse_date(value) for value in found]
    return sorted(set(normalized))


def extract_amounts(text: str) -> list[float]:
    amounts: list[float] = []
    for match in AMOUNT_PATTERN.finditer(text):
        raw = match.group(1).replace(" ", "").replace("\u00a0", "")
        raw = raw.replace(",", ".")
        try:
            amounts.append(float(raw))
        except ValueError:
            continue
    return amounts


def extract_keys(text: str, doc_type: str) -> dict[str, Any]:
    """Return the most useful scalar fields for a document type."""
    lower = text.lower()
    amounts = extract_amounts(text)
    dates = extract_dates(text)
    facts: dict[str, Any] = {
        "document_type": doc_type,
        "dates": dates,
        "amounts_eur": amounts,
        "policy_ids": sorted(set(POLICY_PATTERN.findall(text))),
        "emails": sorted(set(EMAIL_PATTERN.findall(text))),
        "phones": sorted(set(PHONE_PATTERN.findall(text))),
    }

    # Loss date is usually the first date appearing in declaration/expert docs.
    if doc_type in {"declaration", "expert_report", "driver_statement"}:
        if dates:
            facts["primary_date"] = dates[0]

    # Amounts linked to damages: keep values over 100 EUR to avoid trivial prices.
    damage_amounts = [value for value in amounts if value >= 100]
    if doc_type in {"declaration", "expert_report", "repair_quote", "driver_statement"}:
        facts["damage_related_amounts_eur"] = damage_amounts

    # Coverage start is generally the first date in a policy.
    if doc_type == "insurance_policy" and dates:
        facts["coverage_start"] = dates[0]

    # Premium unpaid: keep date mentions around "impay".
    if doc_type == "premium_payment" and "impay" in lower:
        facts["has_unpaid_installment"] = True
        facts["unpaid_mentions"] = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", text)
            if "impay" in sentence.lower()
        ]

    return facts


def extract_document_metadata(
    claim_id: str,
    file_name: str,
    doc_type: str,
    blocks: list[dict[str, Any]],
) -> dict[str, Any]:
    text = "\n".join(block.get("text", "") for block in blocks)
    facts = extract_keys(text, doc_type)
    return {
        "claim_id": claim_id,
        "file_name": file_name,
        "doc_type": doc_type,
        "page_count": max((block.get("page") or 0) for block in blocks) if blocks else 0,
        "block_count": len(blocks),
        "metadata": facts,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)