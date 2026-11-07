"""PDF parsing with Docling, preserving page numbers and reading order.

The functions here convert one PDF into an ordered list of text blocks.
Each block keeps enough metadata to cite the source file and exact PDF page.
"""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from docling.document_converter import DocumentConverter

LOGGER = logging.getLogger(__name__)


@dataclass
class ParsedBlock:
    """One ordered, citable content block extracted from a PDF."""

    claim_id: str
    file_name: str
    doc_type: str
    page: int | None
    block_type: str
    heading: str
    text: str


def _first_page(item: Any) -> int | None:
    """Return the first page number attached to an item, if available."""
    provenance = getattr(item, "prov", None) or []
    if not provenance:
        return None
    return getattr(provenance[0], "page_no", None)


class PdfParser:
    """Thin wrapper around Docling's converter with deterministic output."""

    """Couche logicielle légère autour de Docling.
       Deterministic output : le programme cherche à produire une représentation stable et prévisible.
       Cela ne garantit pas que Docling produira toujours exactement le même résultat, notamment avec des PDF complexes ou des OCR."""

    def __init__(self) -> None:
        # Docling's DocumentConverter performs the PDF-to-structured-document conversion.
        self.converter = (
            DocumentConverter()
        )  # DocumentConverter() est une classe de Docling.

    def convert(self, pdf_path: Path) -> Any:
        """Convert a PDF path into its Docling document representation."""
        result = self.converter.convert(str(pdf_path))
        return result.document

    def blocks_from_document(
        self,
        document: Any,
        claim_id: str,
        file_name: str,
        doc_type: str = "unknown",
    ) -> list[ParsedBlock]:
        """Extract ordered, citable text blocks from a converted Docling document.
        Transforme le document Docling en une liste de blocs structurés.
        document	Document converti par Docling
        claim_id	Identifiant du sinistre
        file_name	Nom du fichier PDF
        doc_type	Type de document
        Retourne une liste d'objets ParsedBlock"""
        blocks: list[ParsedBlock] = []  # liste qui contient les blocs extraits.
        headings: dict[
            int, str
        ] = {}  # dictionnaire qui mémorise les titres et sous-titres du document. Quand on extrait un paragraphe, il est important de savoir à quelle section il appartient. Cela permet d'ajouter du contexte aux chunks utilisés par FAISS.

        for item, level in document.iterate_items():
            """Chaque itération fournit deux informations :
            - item : l'élément courant (titre, paragraphe, tableau, etc.).
            - level : son niveau hiérarchique dans le document.
            Le niveau est utile pour reconstruire la hiérarchie des titres."""
            page = _first_page(item)
            item_type = type(item).__name__

            # Update the hierarchical heading stack.
            if item_type == "SectionHeaderItem":
                text = getattr(item, "text", "") or ""
                if text.strip():
                    headings[level] = text.strip()
                    for deeper_level in list(headings):
                        if deeper_level > level:
                            del headings[
                                deeper_level
                            ]  # supprime les anciens sous-titres qui ne sont plus pertinents.
                    blocks.append(
                        ParsedBlock(
                            claim_id=claim_id,
                            file_name=file_name,
                            doc_type=doc_type,
                            page=page,
                            block_type="heading",
                            heading=text.strip(),
                            text=text.strip(),
                        )
                    )
                continue

            # Docling tables are exported as compact Markdown. On demande à Docling de convertir les tableaux en Markdown.
            if item_type == "TableItem":
                try:
                    text = item.export_to_markdown(document)
                except Exception:
                    text = (
                        getattr(item, "text", "") or ""
                    )  # Si l'export Markdown échoue, on essaie de récupérer le texte brut.
            else:
                text = (
                    getattr(item, "text", "") or ""
                )  # Si ce n'est pas un tableau, on récupère le texte de l'élément.

            text = text.strip()  # on supprime les espaces inutiles.
            if not text:  # Si le texte est vide après nettoyage, on ignore cet élément.
                continue

            # Skip repeated footer/header boilerplate to reduce noise. On ignore les en-têtes et pieds de page. Les conserver dans les chunks peut introduire du bruit dans la recherche vectorielle.
            label = getattr(item, "label", "")
            if label in {"page_footer", "page_header"}:
                continue

            heading_path = " > ".join(
                headings[level] for level in sorted(headings) if level is not None
            )
            # On reconstruit le chemin complet des titres auxquels appartient le bloc.
            blocks.append(
                ParsedBlock(
                    claim_id=claim_id,
                    file_name=file_name,
                    doc_type=doc_type,
                    page=page,
                    block_type=label or item_type,
                    heading=heading_path,
                    text=text,
                )
            )
            """On crée un objet ParsedBlock contenant toutes les informations extraites. Exemple de bloc extrait :
            ParsedBlock(
                claim_id="CLAIM-2026-001",
                file_name="rapport_expertise.pdf",
                doc_type="expert_report",
                page=3,
                block_type="text",
                heading="Expertise automobile > Evaluation financière",
                text="Montant des réparations estimé à 6 600 €."
            )"""

        return blocks


def blocks_to_json(blocks: list[ParsedBlock]) -> list[dict[str, Any]]:
    """Serialize parsed blocks into JSON-compatible dictionaries."""
    return [asdict(block) for block in blocks]


def load_blocks(path: Path) -> list[dict[str, Any]]:
    """Load parsed block dictionaries from a JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
