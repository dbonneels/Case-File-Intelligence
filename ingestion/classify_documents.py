"""Document classification for insurance claim files.

This module maps incoming claim documents to a canonical document type.
Classification is first attempted from the file name, then falls back to a
simple keyword count over the extracted text content.

Quel est le type de chaque document PDF présent dans un dossier d'assurance ?
"""

import json
import re  # Normaliser les noms de fichiers avec les expressions régulières
import unicodedata  # Supprimer les accents des noms de fichiers
from pathlib import Path
from typing import Any

# Canonical document types shared by manifests and validations.
# Each entry pairs a canonical type with the keywords that identify it.
"""Dictionnaire de connaissances, sous la forme d'une liste de tuples.

Chaque tuple contient deux éléments :
1. Le type canonique du document.
2. Les mots-clés permettant de reconnaître ce document."""
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

# Filename fragments that strongly indicate a document type.
# The first matching fragment wins.
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

""" CANONICAL_TYPE_KEYWORDS	--> Contenu du PDF
    FILENAME_HINTS --> Nom du fichier
"""

"""
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


Il faut aussi prévoir les documents facultatifs, car une photo originale peut être utile sans être obligatoire.

"""


def _normalize_name(name: str) -> str:
    """Normalize a file name to lowercase, underscore-separated tokens.
    Normalise les noms de fichiers. Transforme différents formats de noms en une représentation standardisée."""
    # Remove diacritics/accents before normalization.
    name = unicodedata.normalize("NFKD", name)
    name = "".join(char for char in name if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def classify_by_filename(file_name: str) -> str | None:
    """Classify a document from its file name when a known hint matches.
    Returns the canonical document type, or None if no hint matches.
    Cette fonction reçoit le nom d'un fichier.
    Elle retourne :
        - Le type canonique si elle trouve une correspondance.
        - None si elle ne trouve rien.
    """
    normalized = _normalize_name(file_name)
    for hint, doc_type in FILENAME_HINTS:
        if hint in normalized:
            return doc_type
    return None


"""
Classify a document by counting canonical keywords in its text.
    The type with the highest keyword match count wins. Falls back to 'unknown' when no keyword matches.
    Fonction qui calcule un score pour chaque type de document et sélectionne celui qui obtient le meilleur score.

def classify_by_text(text: str) -> str:
        lowered = (
        text.lower()
    )  # On met tout le texte en minuscules pour éviter les problèmes de casse.
    best_type = "unknown"
    best_score = 0
    for doc_type, keywords in CANONICAL_TYPE_KEYWORDS:
        score = sum(
            1 for keyword in keywords if keyword in lowered
        )  # Pour chaque mot-clé, ajouter 1 si ce mot apparaît dans le texte.
        if score > best_score:
            best_score = score
            best_type = doc_type
    return best_type
"""


def classify_by_text(text: str) -> str:
    """On compare les deux meilleurs scores.
    - Si les scores sont différents : on conserve le type ayant le meilleur score.
    - Si les scores sont identiques : on signale une classification ambiguë.
    - Si aucun mot-clé n'est trouvé : on retourne unknown."""
    lowered = text.lower()

    scores = {}

    for doc_type, keywords in CANONICAL_TYPE_KEYWORDS:
        score = sum(1 for keyword in keywords if keyword in lowered)
        if score > 0:
            scores[doc_type] = score

    if not scores:
        return "unknown"

    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)

    best_type, best_score = sorted_scores[0]

    # Detect ambiguity when multiple types have the same score.
    if len(sorted_scores) > 1:
        second_type, second_score = sorted_scores[1]

        if second_score == best_score:
            print(f"WARNING: Ambiguous classification: {best_type} / {second_type}")

    return best_type


"""Attention : cette version détecte les égalités entre les deux meilleurs scores, mais ne traite pas encore les cas où deux catégories ont des scores proches, par exemple 3 et 2.


Une amélioration importante serait de stocker les scores et les types alternatifs dans les métadonnées du document, plutôt que de simplement afficher un avertissement avec print(). Cela permettrait à l'application de présenter une alerte dans l'interface web et de demander une validation humaine.: 
Ne pas simplement afficher un WARNING avec print().

À terme, il serait préférable de conserver l'information dans les métadonnées du document :
{    "doc_type": "expert_report",    "classification_status": "ambiguous",    "alternative_types": ["driver_statement"],    "classification_scores": {        "expert_report": 3,        "driver_statement": 3    }}



Cela permettra à l'application de présenter une alerte dans l'interface web et de demander une validation humaine.

Dans un système d'analyse de dossiers d'assurance, une classification incertaine doit être signalée plutôt que silencieusement acceptée."""


def classify_document(file_name: str, text: str = "") -> str:
    """Classify a document using filename first, then content fallback.
    Le nom du fichier est prioritaire, même si le contenu du PDF contredit ce nom. Un fichier mal nommé peut donc être mal classifié sans vérification."""
    by_name = classify_by_filename(file_name)
    if by_name:
        return by_name
    return classify_by_text(text)


def build_claim_inventory(claim_id: str, claim_dir: Path) -> list[dict[str, Any]]:
    """Return one inventory row per received PDF in a claim folder. Cette fonction construit l'inventaire des fichiers PDF reçus pour un sinistre."""
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
    """Load and return the manifest.json located in a claim folder."""
    manifest_path = claim_dir / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
