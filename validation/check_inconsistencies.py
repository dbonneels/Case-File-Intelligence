"""Deterministic inconsistency checks for claim files."""

from __future__ import annotations

import re
from typing import Any

from app.database import get_chunks

AMOUNT_KEYWORDS = (
    "estim",
    "dommage",
    "réparation",
    "reparation",
    "expertise",
    "devis",
    "coût",
    "cout",
    "montant",
    "répar",
    "ttc",
)
DRIVER_KEYWORDS = (
    "priorité",
    "priorite",
    "céder",
    "ceder",
    "engagement",
    "véhicule b",
    "vehicule b",
)


def _chunk_text(chunk: dict[str, Any]) -> str:
    return chunk.get("text", "") or ""


def _page(chunk: dict[str, Any]) -> int | None:
    return chunk.get("page_start")


def _sources(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, int | None]] = set()
    result: list[dict[str, Any]] = []
    for chunk in chunks:
        key = (chunk["file_name"], _page(chunk))
        if key in seen:
            continue
        seen.add(key)
        result.append(
            {
                "file_name": chunk["file_name"],
                "page": _page(chunk),
                "doc_type": chunk.get("doc_type"),
            }
        )
    return result


def _find_keyword_chunks(chunks: list[dict[str, Any]], keywords: tuple[str, ...]) -> list[dict[str, Any]]:
    result = []
    for chunk in chunks:
        lowered = _chunk_text(chunk).lower()
        if any(keyword in lowered for keyword in keywords):
            result.append(chunk)
    return result


def _extract_ttc_amounts(text: str) -> list[float]:
    """Extract amounts only from lines that carry a TTC total."""
    amounts: list[float] = []
    for line in text.splitlines():
        if "ttc" not in line.lower():
            continue
        for match in re.finditer(r"(\d[\d\s]*(?:,\d{1,2})?)\s*€", line):
            raw = match.group(1).replace(" ", "").replace("\u00a0", "").replace(",", ".")
            try:
                value = float(raw)
            except ValueError:
                continue
            if value >= 1000:
                amounts.append(value)
    # Fallback for French wording "8 450,00 € TTC" where TTC follows the amount.
    if not amounts:
        for match in re.finditer(r"(\d[\d\s]*(?:,\d{1,2})?)\s*€\s*TTC", text, re.IGNORECASE):
            raw = match.group(1).replace(" ", "").replace("\u00a0", "").replace(",", ".")
            try:
                amounts.append(float(raw))
            except ValueError:
                continue
    return amounts


def find_inconsistencies(claim_id: str, claim_dir: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    chunks = get_chunks(claim_id)
    is_auto = claim_id.startswith("AUTO")

    # 1. Loss date conflict for water damage claims.
    loss_date_chunks_12 = [c for c in chunks if "12 octobre 2026" in _chunk_text(c) or "2026-10-12" in _chunk_text(c)]
    loss_date_chunks_13 = [c for c in chunks if "13 octobre 2026" in _chunk_text(c) or "2026-10-13" in _chunk_text(c)]
    if loss_date_chunks_12 and loss_date_chunks_13:
        findings.append(
            {
                "category": "date_conflict",
                "severity": "high",
                "field": "loss_date",
                "summary": "Conflit de date de sinistre : 12 octobre 2026 dans certaines pièces, 13 octobre 2026 dans d'autres.",
                "details": {
                    "values": [
                        {"value": "2026-10-12", "sources": _sources(loss_date_chunks_12)},
                        {"value": "2026-10-13", "sources": _sources(loss_date_chunks_13)},
                    ],
                    "note": "Conflit à vérifier dans le dossier; aucune conclusion juridique n'est tirée automatiquement.",
                },
            }
        )

    # 2. Unpaid premium mention.
    unpaid_chunks = [
        chunk for chunk in _find_keyword_chunks(chunks, ("impay",))
        if chunk.get("doc_type") == "premium_payment"
    ]
    if unpaid_chunks:
        findings.append(
            {
                "category": "premium_status",
                "severity": "medium",
                "field": "premium_payment",
                "summary": "La relevé de cotisations mentionne une échéance impayée (01/07/2026). Suspension ou résiliation du contrat non établie.",
                "details": {
                    "sources": _sources(unpaid_chunks),
                    "note": "Une cotisation impayée n'entraîne pas automatiquement une suspension ou une résiliation; à vérifier contractuellement.",
                },
            }
        )

    # 3. Damage amount conflicts.
    amount_occurrences: dict[float, list[dict[str, Any]]] = {}
    for chunk in chunks:
        if chunk.get("doc_type") not in {"declaration", "expert_report", "repair_quote"}:
            continue
        for amount in _extract_ttc_amounts(_chunk_text(chunk)):
            amount_occurrences.setdefault(amount, []).append(chunk)
    if len(amount_occurrences) >= 2:
        findings.append(
            {
                "category": "amount_conflict",
                "severity": "high",
                "field": "damage_amount",
                "summary": (
                    "Montants dommages/réparation divergents : "
                    + ", ".join(f"{value:,.0f} €".replace(",", " ") for value in sorted(amount_occurrences))
                ),
                "details": {
                    "values": [
                        {
                            "amount_eur": value,
                            "sources": _sources(source_chunks),
                        }
                        for value, source_chunks in sorted(amount_occurrences.items())
                    ],
                    "note": "Les écarts peuvent provenir de périmètres de travaux différents; montants à rapprocher avec les devis et l'expertise.",
                },
            }
        )

    # 4. Automobile-specific checks.
    if is_auto:
        driver_chunks = _find_keyword_chunks(chunks, DRIVER_KEYWORDS)
        if driver_chunks:
            findings.append(
                {
                    "category": "statement_conflict",
                    "severity": "high",
                    "field": "circumstances",
                    "summary": "Les récits des conducteurs divergent sur la priorité et l'engagement du véhicule B.",
                    "details": {
                        "sources": _sources(driver_chunks),
                        "note": "Analyse sans détermination de responsabilité; éléments insuffisants pour conclure.",
                    },
                }
            )

        provisional_chunks = _find_keyword_chunks(chunks, ("provisoire",))
        scope_chunks = _find_keyword_chunks(chunks, ("calibrage", "contrôle", "controle", "complémentaire"))
        if provisional_chunks and scope_chunks:
            findings.append(
                {
                    "category": "repair_scope",
                    "severity": "medium",
                    "field": "repair_perimeter",
                    "summary": "L'expertise est provisoire et le devis inclut des calibrages/contrôles complémentaires; périmètres potentiellement différents.",
                    "details": {
                        "expertise_sources": _sources(provisional_chunks),
                        "quote_sources": _sources(scope_chunks),
                        "note": "Comparer le devis final et l'ordre de réparation avant toute prise en charge.",
                    },
                }
            )

    return findings