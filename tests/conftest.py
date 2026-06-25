"""Shared pytest fixtures for the KayzeeOCR test suite."""

from __future__ import annotations

import pytest
from PIL import Image

from src.preprocessing.splitter import PageItem


@pytest.fixture
def sample_image() -> Image.Image:
    """A small white RGB test image."""
    return Image.new("RGB", (800, 1000), "white")


@pytest.fixture
def sample_element_dict() -> dict:
    """A single valid, fully-populated element dict."""
    return {
        "id": "elem_001",
        "type": "paragraph",
        "bbox": [0.1, 0.1, 0.9, 0.2],
        "bbox_pixel": [80, 100, 720, 200],
        "reading_order": 1,
        "confidence": 0.98,
        "content": {
            "text": "Hello world.",
            "html": None,
            "latex": None,
            "image_ref": None,
        },
        "language": "en",
        "is_truncated": False,
    }


@pytest.fixture
def sample_page_item(sample_image: Image.Image) -> PageItem:
    """A PageItem wrapping the sample image."""
    return PageItem(
        page_number=1,
        image=sample_image,
        original_width=sample_image.width,
        original_height=sample_image.height,
        source_file="sample.png",
    )
