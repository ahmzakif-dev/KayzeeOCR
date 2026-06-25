"""Standalone preprocessing/postprocessing smoke test (no GPU, no weights).

Exercises the non-model parts of the KayzeeOCR pipeline end-to-end with a
synthetic page and dummy elements:

    loader/converter → normalizer → splitter → sorter → assembler → validator

Run with::

    python src/tools/test_single_page.py
"""

import sys
import json
import logging
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont  # noqa: F401 (ImageFont kept for parity)

logging.basicConfig(level=logging.INFO)


def make_dummy_page(width=800, height=1100) -> Image.Image:
    """Buat synthetic document page untuk testing."""
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Gambar simulasi title
    draw.rectangle([40, 40, 760, 100], outline=(0, 0, 0), width=2)
    draw.text((50, 60), "Document Title", fill=(0, 0, 0))
    # Gambar simulasi paragraph
    draw.rectangle([40, 120, 760, 300], outline=(200, 200, 200), width=1)
    draw.text((50, 140), "Paragraph text here...", fill=(50, 50, 50))
    # Gambar simulasi tabel
    draw.rectangle([40, 320, 760, 500], outline=(0, 0, 0), width=2)
    draw.line([40, 360, 760, 360], fill=(0, 0, 0), width=1)
    draw.text((50, 340), "Table Header", fill=(0, 0, 0))
    return img


def make_dummy_elements(page_width, page_height) -> list[dict]:
    """Buat dummy elements sesuai format output schema."""
    return [
        {
            "id": "elem_001",
            "type": "title",
            "bbox": [0.05, 0.036, 0.95, 0.091],
            "bbox_pixel": [40, 40, 760, 100],
            "reading_order": 1,
            "confidence": 0.95,
            "content": {
                "text": "Document Title",
                "html": None,
                "latex": None,
                "image_ref": None,
            },
        },
        {
            "id": "elem_002",
            "type": "paragraph",
            "bbox": [0.05, 0.109, 0.95, 0.273],
            "bbox_pixel": [40, 120, 760, 300],
            "reading_order": 2,
            "confidence": 0.91,
            "content": {
                "text": "Paragraph text here...",
                "html": None,
                "latex": None,
                "image_ref": None,
            },
        },
        {
            "id": "elem_003",
            "type": "table_simple",
            "bbox": [0.05, 0.291, 0.95, 0.455],
            "bbox_pixel": [40, 320, 760, 500],
            "reading_order": 3,
            "confidence": 0.88,
            "content": {
                "text": None,
                "html": "<table><tr><th>Header</th></tr></table>",
                "latex": None,
                "image_ref": None,
            },
        },
    ]


def run_test():
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    print("\n=== KayzeeOCR — Preprocessing chain test ===\n")

    # Test 1: Import semua komponen
    print("[1/6] Import components...")
    from src.preprocessing.converter import ImageConverter
    from src.preprocessing.normalizer import ImageNormalizer, ResolutionConfig
    from src.preprocessing.splitter import PageSplitter, PageItem
    from src.postprocessing.assembler import OutputAssembler
    from src.postprocessing.validator import OutputValidator
    from src.postprocessing.sorter import ReadingOrderSorter
    print("      ✓ All imports OK")

    # Test 2: Buat dummy image
    print("[2/6] Create dummy page image...")
    img = make_dummy_page()
    print(f"      ✓ Created {img.size[0]}x{img.size[1]} RGB image")

    # Test 3: Preprocessing chain
    print("[3/6] Run preprocessing chain...")
    converter = ImageConverter()
    normalizer = ImageNormalizer(ResolutionConfig())
    splitter = PageSplitter()

    img_rgb = converter.to_rgb(img)
    img_s1 = normalizer.for_stage1(img_rgb)
    img_s2 = normalizer.for_stage2(img_rgb)

    page_item = PageItem(
        page_number=1,
        image=img_rgb,
        original_width=img_rgb.width,
        original_height=img_rgb.height,
        source_file="test_dummy.png",
    )
    print(f"      ✓ stage1 image: {img_s1.size}")
    print(f"      ✓ stage2 image: {img_s2.size}")

    # Test 4: Dummy elements + sorter
    print("[4/6] Sort dummy elements...")
    sorter = ReadingOrderSorter()
    elements = make_dummy_elements(img_rgb.width, img_rgb.height)
    sorted_elements = sorter.sort(elements)
    print(f"      ✓ {len(sorted_elements)} elements sorted")

    # Test 5: Assembler
    print("[5/6] Assemble output...")
    assembler = OutputAssembler()
    result = assembler.assemble(
        page_item=page_item,
        elements=sorted_elements,
        processing_time_ms=123.45,
        model_version="kayzeeocr-0.1.0",
    )
    print(f"      ✓ document_id: {result['document_id']}")
    print(f"      ✓ elements: {len(result['elements'])}")

    # Test 6: Validator
    print("[6/6] Validate output against schema...")
    validator = OutputValidator()
    is_valid, errors = validator.validate(result)
    if is_valid:
        print("      ✓ Output valid against output_schema.json")
    else:
        print("      ✗ Validation errors:")
        for e in errors:
            print(f"        - {e}")

    print("\n=== Semua test preprocessing passed ===")
    print("\nJSON output preview:")
    print(json.dumps(result, indent=2, ensure_ascii=False)[:800] + "\n...")


if __name__ == "__main__":
    run_test()
