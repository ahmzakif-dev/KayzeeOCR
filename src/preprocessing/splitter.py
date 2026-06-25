"""Multi-page splitting.

Wraps a list of page images (as produced by the loader) into :class:`PageItem`
records that carry the original dimensions and source-file provenance needed by
later pipeline stages.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class PageItem:
    """A single document page and its provenance.

    Attributes:
        page_number: 1-indexed page number within the source document.
        image: The page image (RGB PIL Image).
        original_width: Width of ``image`` in pixels at split time.
        original_height: Height of ``image`` in pixels at split time.
        source_file: Path/name of the originating file.
    """

    page_number: int
    image: Image.Image
    original_width: int
    original_height: int
    source_file: str


class PageSplitter:
    """Turn a list of page images into a list of :class:`PageItem`."""

    def split(
        self, pages: list[Image.Image], source_file: str = ""
    ) -> list[PageItem]:
        """Wrap every page image into a :class:`PageItem` (1-indexed).

        Args:
            pages: Page images in document order.
            source_file: Originating file path/name, stored on each item.

        Returns:
            List of :class:`PageItem`, one per input image.
        """
        items: list[PageItem] = []
        for idx, img in enumerate(pages, start=1):
            items.append(
                PageItem(
                    page_number=idx,
                    image=img,
                    original_width=img.width,
                    original_height=img.height,
                    source_file=source_file,
                )
            )
        logger.debug("Split %d page(s) from '%s'", len(items), source_file)
        return items

    def split_range(
        self,
        pages: list[Image.Image],
        start: int,
        end: int,
        source_file: str = "",
    ) -> list[PageItem]:
        """Split a 1-indexed inclusive page range ``[start, end]``.

        Args:
            pages: All page images in document order.
            start: First page number to include (1-indexed, inclusive).
            end: Last page number to include (1-indexed, inclusive).
            source_file: Originating file path/name.

        Returns:
            List of :class:`PageItem` for the requested range, preserving the
            original (1-indexed) page numbers.

        Raises:
            ValueError: If the range is invalid.
        """
        if start < 1 or end < start:
            raise ValueError(f"Invalid page range: start={start}, end={end}")
        all_items = self.split(pages, source_file=source_file)
        selected = [it for it in all_items if start <= it.page_number <= end]
        logger.debug(
            "Selected %d page(s) for range [%d, %d]", len(selected), start, end
        )
        return selected
