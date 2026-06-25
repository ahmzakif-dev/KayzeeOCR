"""Tests for the FileLoader / input loading layer."""

from __future__ import annotations

import pytest
from PIL import Image

from src.input.loader import (
    FileLoader,
    UnsupportedFormatError,
    load_file,
)

try:
    import fitz  # PyMuPDF

    HAS_FITZ = True
except ImportError:  # pragma: no cover
    HAS_FITZ = False


def _make_image(path, size=(640, 480), color="white", fmt=None):
    img = Image.new("RGB", size, color)
    img.save(path, format=fmt)
    return path


def test_load_jpeg(tmp_path):
    p = _make_image(tmp_path / "img.jpg", fmt="JPEG")
    images = load_file(p)
    assert len(images) == 1
    assert images[0].mode == "RGB"


def test_load_png(tmp_path):
    p = _make_image(tmp_path / "img.png", fmt="PNG")
    images = FileLoader().load(p)
    assert len(images) == 1
    assert images[0].size == (640, 480)


@pytest.mark.skipif(not HAS_FITZ, reason="PyMuPDF not installed")
def test_load_pdf_single_page(tmp_path):
    pdf_path = tmp_path / "single.pdf"
    doc = fitz.open()
    doc.new_page(width=595, height=842)
    doc.save(pdf_path)
    doc.close()

    images = FileLoader(dpi=72).load(pdf_path)
    assert len(images) == 1
    assert images[0].mode == "RGB"


@pytest.mark.skipif(not HAS_FITZ, reason="PyMuPDF not installed")
def test_load_pdf_multi_page(tmp_path):
    pdf_path = tmp_path / "multi.pdf"
    doc = fitz.open()
    for _ in range(3):
        doc.new_page(width=595, height=842)
    doc.save(pdf_path)
    doc.close()

    images = FileLoader(dpi=72).load(pdf_path)
    assert len(images) == 3


def test_unsupported_format(tmp_path):
    bogus = tmp_path / "file.xyz"
    bogus.write_text("not a real format")
    with pytest.raises(UnsupportedFormatError):
        FileLoader().load(bogus)


def test_heic_registered():
    formats = FileLoader().get_supported_formats()
    assert ".heic" in formats
    assert ".heif" in formats


def test_output_is_rgb(tmp_path):
    # Save an RGBA PNG; loader must return RGB.
    p = tmp_path / "rgba.png"
    Image.new("RGBA", (100, 100), (255, 0, 0, 128)).save(p)
    images = FileLoader().load(p)
    assert all(img.mode == "RGB" for img in images)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        FileLoader().load(tmp_path / "does_not_exist.png")
