"""Final output assembly.

Takes a page's detected+recognized elements and assembles a dict that conforms
to ``schemas/output_schema.json``. Also merges per-page dicts into a single
document-level result.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# The 17 valid element types, mirrored from the schema enum, used for defaults.
_CONTENT_KEYS = ("text", "html", "latex", "image_ref")


class OutputAssembler:
    """Assemble schema-conformant page and document output dicts."""

    def __init__(self, schema_path: str | None = None) -> None:
        """Initialize the assembler.

        Args:
            schema_path: Optional path to the output schema (kept for reference;
                validation itself is handled by ``OutputValidator``).
        """
        self.schema_path = schema_path

    # -- page assembly ----------------------------------------------------- #

    def assemble(
        self,
        page_item: Any,
        elements: list[dict],
        processing_time_ms: float,
        model_version: str,
    ) -> dict:
        """Assemble a single page's output dict.

        Args:
            page_item: A ``PageItem`` (provides page number, dims, source file).
            elements: Recognized element dicts for this page.
            processing_time_ms: Time spent processing this page.
            model_version: Identifier of the producing model/pipeline.

        Returns:
            A dict conforming to the page output schema.
        """
        source_file = getattr(page_item, "source_file", "") or ""
        page_number = int(getattr(page_item, "page_number", 1))
        page_width = int(getattr(page_item, "original_width", 0))
        page_height = int(getattr(page_item, "original_height", 0))

        normalized = [self._normalize_element(e) for e in elements]

        return {
            "document_id": self._generate_document_id(source_file, page_number),
            "source_file": source_file,
            "page_number": page_number,
            "page_width": page_width,
            "page_height": page_height,
            "processing_time_ms": round(float(processing_time_ms), 3),
            "model_version": model_version,
            "elements": normalized,
        }

    def assemble_document(self, pages: list[dict]) -> dict:
        """Merge per-page output dicts into one document-level dict.

        Args:
            pages: List of page dicts as returned by :meth:`assemble`.

        Returns:
            A document dict with ``total_pages``, ``total_elements`` and the
            list of page results under ``pages``.
        """
        total_elements = sum(len(p.get("elements", [])) for p in pages)
        source_file = pages[0].get("source_file", "") if pages else ""
        model_version = pages[0].get("model_version", "") if pages else ""
        return {
            "source_file": source_file,
            "model_version": model_version,
            "total_pages": len(pages),
            "total_elements": total_elements,
            "pages": pages,
        }

    # -- helpers ----------------------------------------------------------- #

    def _generate_document_id(self, source_file: str, page_number: int) -> str:
        """Generate a stable id from the source file stem and page number."""
        stem = Path(source_file).stem if source_file else "document"
        digest = hashlib.sha1(
            f"{source_file}:{page_number}".encode("utf-8")
        ).hexdigest()[:8]
        return f"{stem}_p{page_number}_{digest}"

    def _normalize_element(self, element: dict) -> dict:
        """Ensure an element has all required fields with sane defaults.

        Fills missing ``content`` sub-fields with None, ensures ``confidence``,
        ``language`` and ``is_truncated`` exist, and coerces bbox_pixel to ints.
        """
        content_in = element.get("content") or {}
        content = {key: content_in.get(key) for key in _CONTENT_KEYS}

        bbox = element.get("bbox") or [0.0, 0.0, 1.0, 1.0]
        bbox_pixel = element.get("bbox_pixel") or [0, 0, 0, 0]

        return {
            "id": element.get("id", "elem_000"),
            "type": element.get("type", "paragraph"),
            "bbox": [float(v) for v in bbox],
            "bbox_pixel": [int(v) for v in bbox_pixel],
            "reading_order": int(element.get("reading_order", 1)),
            "confidence": float(element.get("confidence", 1.0)),
            "content": content,
            "language": element.get("language"),
            "is_truncated": bool(element.get("is_truncated", False)),
        }


def assemble_output(
    page_item: Any,
    elements: list[dict],
    processing_time_ms: float,
    model_version: str,
) -> dict:
    """Convenience function: assemble a single page output dict."""
    return OutputAssembler().assemble(
        page_item, elements, processing_time_ms, model_version
    )
