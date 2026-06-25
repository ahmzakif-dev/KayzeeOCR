"""Tests for preprocessing (converter, normalizer) and the reading-order sorter."""

from __future__ import annotations

from PIL import Image

from src.postprocessing.sorter import ReadingOrderSorter, sort_elements
from src.preprocessing.converter import ImageConverter
from src.preprocessing.normalizer import ImageNormalizer, ResolutionConfig


# -- normalizer -------------------------------------------------------------- #


def test_normalizer_stage1_downsample():
    norm = ImageNormalizer(ResolutionConfig(stage1_max_px=1036))
    big = Image.new("RGB", (4000, 3000), "white")
    out = norm.for_stage1(big)
    assert max(out.size) == 1036
    # Aspect ratio preserved.
    assert abs(out.width / out.height - 4000 / 3000) < 0.01


def test_normalizer_stage1_no_upscale_small():
    norm = ImageNormalizer(ResolutionConfig(stage1_max_px=1036))
    small = Image.new("RGB", (500, 400), "white")
    out = norm.for_stage1(small)
    assert out.size == (500, 400)


def test_normalizer_stage2_upscale():
    norm = ImageNormalizer(ResolutionConfig(stage2_min_px=1036, stage2_max_px=4096))
    small = Image.new("RGB", (400, 300), "white")
    out = norm.for_stage2(small)
    assert max(out.size) == 1036


def test_normalizer_stage2_cap():
    norm = ImageNormalizer(ResolutionConfig(stage2_min_px=1036, stage2_max_px=4096))
    huge = Image.new("RGB", (8000, 6000), "white")
    out = norm.for_stage2(huge)
    assert max(out.size) == 4096


# -- converter --------------------------------------------------------------- #


def test_converter_rgba_to_rgb():
    conv = ImageConverter()
    rgba = Image.new("RGBA", (50, 50), (10, 20, 30, 128))
    out = conv.to_rgb(rgba)
    assert out.mode == "RGB"


def test_converter_grayscale_to_rgb():
    conv = ImageConverter()
    gray = Image.new("L", (50, 50), 128)
    out = conv.to_rgb(gray)
    assert out.mode == "RGB"


def test_converter_rgb_passthrough():
    conv = ImageConverter()
    rgb = Image.new("RGB", (10, 10), "white")
    assert conv.to_rgb(rgb) is rgb


# -- sorter ------------------------------------------------------------------ #


def _elem(eid, y1, x1=0.1, x2=0.9, y2=None, etype="paragraph"):
    if y2 is None:
        y2 = y1 + 0.05
    return {
        "id": eid,
        "type": etype,
        "bbox": [x1, y1, x2, y2],
        "bbox_pixel": [0, 0, 0, 0],
        "reading_order": 0,
    }


def test_sorter_basic_order():
    # Provided out of order; expect top-to-bottom.
    elements = [_elem("c", 0.7), _elem("a", 0.1), _elem("b", 0.4)]
    out = sort_elements(elements)
    assert [e["id"] for e in out] == ["a", "b", "c"]


def test_sorter_reading_order_assigned():
    elements = [_elem("a", 0.1), _elem("b", 0.5)]
    out = ReadingOrderSorter().sort(elements)
    assert [e["reading_order"] for e in out] == [1, 2]


def test_sorter_empty():
    assert sort_elements([]) == []


def test_sorter_does_not_mutate_input():
    elements = [_elem("a", 0.1)]
    elements[0]["reading_order"] = 99
    sort_elements(elements)
    assert elements[0]["reading_order"] == 99
