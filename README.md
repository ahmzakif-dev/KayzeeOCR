# KayzeeOCR

Document layout detection and structured OCR pipeline based on
Qwen2.5-VL-2B. Detects 17 element classes and outputs structured JSON.

[![License: KayzeeOCR Research](https://img.shields.io/badge/license-Research%20Only-blue)]()
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-green)]()

## Features

- **17 layout classes**: title, headings, paragraph, lists, tables
  (simple/merged/borderless), figure, caption, header/footer,
  footnote, page number, math formula
- **Universal input**: JPEG, PNG, WEBP, TIFF, HEIC, HEIF, PDF, DOCX
- **Structured JSON output** with bounding boxes (relative + pixel)
- **Two-stage pipeline**: layout detection → content recognition
- **≤2B parameters**: runs on mid-range GPU

## Quick Start

```bash
pip install -r requirements.txt
python -m src.pipeline document.pdf --device cuda
```

## Output Format

```json
{
  "document_id": "abc123",
  "page_number": 1,
  "elements": [
    {
      "id": "elem_001",
      "type": "title",
      "bbox": [0.05, 0.03, 0.90, 0.09],
      "content": { "text": "Document Title" }
    }
  ]
}
```

## Architecture

Two-stage decoupled VLM pipeline inspired by MinerU 2.5:
- **Stage I**: Layout detection on downsampled image (1036px)
- **Stage II**: Content recognition on native-resolution crops

Base model: `Qwen/Qwen2.5-VL-2B-Instruct`

## License

KayzeeOCR Research License — free for academic research and
personal experimentation. Commercial use requires a separate license.
See [LICENSE](LICENSE) for details.
