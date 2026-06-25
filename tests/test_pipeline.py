"""Integration-ish tests for config, assembler, and validator.

Model-dependent paths are skipped unless a GPU/model is available; the rest run
purely on CPU with synthetic data.
"""

from __future__ import annotations

import copy

import pytest

from src.pipeline import MODEL_VERSION, PipelineConfig
from src.postprocessing.assembler import OutputAssembler, assemble_output
from src.postprocessing.validator import OutputValidator, validate_output


# -- config ------------------------------------------------------------------ #


def test_pipeline_config_defaults():
    cfg = PipelineConfig()
    assert cfg.model_name == "Qwen/Qwen2.5-VL-2B-Instruct"
    assert cfg.input_dpi == 150
    assert cfg.stage1_max_px == 1036
    assert cfg.stage2_max_px == 4096
    assert cfg.validate_output is True
    assert cfg.max_pages is None


# -- assembler --------------------------------------------------------------- #


def test_assembler_output_structure(sample_page_item, sample_element_dict):
    result = assemble_output(
        sample_page_item, [sample_element_dict], 12.5, MODEL_VERSION
    )
    for field in (
        "document_id",
        "source_file",
        "page_number",
        "page_width",
        "page_height",
        "processing_time_ms",
        "model_version",
        "elements",
    ):
        assert field in result
    assert result["page_number"] == 1
    assert result["model_version"] == MODEL_VERSION
    assert len(result["elements"]) == 1


def test_assembler_normalizes_missing_content(sample_page_item):
    minimal = {
        "id": "elem_001",
        "type": "title",
        "bbox": [0.0, 0.0, 1.0, 0.1],
        "bbox_pixel": [0, 0, 800, 100],
        "reading_order": 1,
    }
    result = OutputAssembler().assemble(
        sample_page_item, [minimal], 1.0, MODEL_VERSION
    )
    content = result["elements"][0]["content"]
    assert set(content.keys()) == {"text", "html", "latex", "image_ref"}
    assert result["elements"][0]["confidence"] == 1.0


def test_assemble_document_aggregates(sample_page_item, sample_element_dict):
    page = assemble_output(
        sample_page_item, [sample_element_dict], 1.0, MODEL_VERSION
    )
    doc = OutputAssembler().assemble_document([page, page])
    assert doc["total_pages"] == 2
    assert doc["total_elements"] == 2


# -- validator --------------------------------------------------------------- #


def _valid_page(sample_page_item, sample_element_dict) -> dict:
    return assemble_output(
        sample_page_item, [sample_element_dict], 5.0, MODEL_VERSION
    )


def test_validator_valid_schema(sample_page_item, sample_element_dict):
    page = _valid_page(sample_page_item, sample_element_dict)
    is_valid, errors = OutputValidator().validate(page)
    assert is_valid, errors
    assert validate_output(page) is True


def test_validator_invalid_type(sample_page_item, sample_element_dict):
    page = _valid_page(sample_page_item, sample_element_dict)
    page["elements"][0]["type"] = "not_a_real_type"
    is_valid, errors = OutputValidator().validate(page)
    assert not is_valid
    assert errors


def test_validator_invalid_bbox(sample_page_item, sample_element_dict):
    page = _valid_page(sample_page_item, sample_element_dict)
    page["elements"][0]["bbox"] = [0.1, 0.1, 1.5, 0.2]  # 1.5 > 1.0
    is_valid, _ = OutputValidator().validate(page)
    assert not is_valid


def test_validator_missing_required_field(sample_page_item, sample_element_dict):
    page = _valid_page(sample_page_item, sample_element_dict)
    del page["page_number"]
    is_valid, _ = OutputValidator().validate(page)
    assert not is_valid


# -- model-dependent (skipped without torch+model) --------------------------- #


@pytest.mark.skipif(
    True, reason="Requires GPU/model weights; enable manually for e2e checks."
)
def test_full_pipeline_smoke(sample_page_item):  # pragma: no cover
    from src.pipeline import DocumentOCRPipeline

    pipe = DocumentOCRPipeline()
    pipe.setup()
    try:
        result = pipe.process_page(sample_page_item, 0, 1)
        assert "elements" in result
    finally:
        pipe.teardown()
