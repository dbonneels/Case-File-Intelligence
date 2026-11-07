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
    provenance = getattr(item, "prov", None) or []
    if not provenance:
        return None
    return getattr(provenance[0], "page_no", None)


class PdfParser:
    """Thin wrapper around Docling's converter with deterministic output."""

    def __init__(self) -> None:
        self.converter = DocumentConverter()

    def convert(self, pdf_path: Path) -> Any:
        result = self.converter.convert(str(pdf_path))
        return result.document

    def blocks_from_document(
        self,
        document: Any,
        claim_id: str,
        file_name: str,
        doc_type: str = "unknown",
    ) -> list[ParsedBlock]:
        blocks: list[ParsedBlock] = []
        headings: dict[int, str] = {}

        for item, level in document.iterate_items():
            page = _first_page(item)
            item_type = type(item).__name__

            # Update the hierarchical heading stack.
            if item_type == "SectionHeaderItem":
                text = getattr(item, "text", "") or ""
                if text.strip():
                    headings[level] = text.strip()
                    for deeper_level in list(headings):
                        if deeper_level > level:
                            del headings[deeper_level]
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

            # Docling tables are exported as compact Markdown.
            if item_type == "TableItem":
                try:
                    text = item.export_to_markdown(document)
                except Exception:  # pragma: no cover - defensive
                    text = getattr(item, "text", "") or ""
            else:
                text = getattr(item, "text", "") or ""

            text = text.strip()
            if not text:
                continue

            # Skip repeated footer/header boilerplate to reduce noise.
            label = getattr(item, "label", "")
            if label in {"page_footer", "page_header"}:
                continue

            heading_path = " > ".join(
                headings[level] for level in sorted(headings) if level is not None
            )
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

        return blocks


def blocks_to_json(blocks: list[ParsedBlock]) -> list[dict[str, Any]]:
    return [asdict(block) for block in blocks]


def load_blocks(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)