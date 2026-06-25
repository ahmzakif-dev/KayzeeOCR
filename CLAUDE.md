# KAYZEEOCR — Claude Code Context

## Project
Document OCR pipeline berbasis Qwen2.5-VL-2B (atau Qwen3-VL-2B).
Mendeteksi 17 class elemen layout dokumen dan menghasilkan JSON
terstruktur. Target: ≤2B parameter, deployable via vLLM atau
HuggingFace Transformers.

## Cara menjalankan

```bash
# Install dependencies
pip install -r requirements.txt

# Inference satu file
python -m src.pipeline document.pdf --device cuda

# Inference dengan opsi lengkap
python -m src.pipeline document.pdf \
  --model Qwen/Qwen2.5-VL-2B-Instruct \
  --device cuda \
  --output-dir ./outputs \
  --dpi 150

# Run tests
python -m pytest tests/ -v
```

## Struktur folder

```
src/
  input/loader.py          # FileLoader: semua format → list[PIL.Image]
  model/
    vl_engine.py           # VisionLanguageEngine: load + generate
    prompts.py             # Semua prompt string Stage I & II
    layout_detector.py     # Stage I: layout detection → bbox + 17 class
    content_recognizer.py  # Stage II: content recognition per elemen
    parsing.py             # Helper: strip_code_fence, extract_json_object
  postprocessing/
    assembler.py           # Rakit final JSON output
    sorter.py              # Reading order sorting
    validator.py           # Validasi terhadap output_schema.json
  preprocessing/
    converter.py           # RGB normalization
    normalizer.py          # Resolution normalization (stage1/stage2)
    splitter.py            # Split halaman → PageItem list
  pipeline.py              # Orchestrator end-to-end + CLI
schemas/
  output_schema.json       # JSON Schema formal untuk output
tests/                     # Unit tests (27 passed, 1 skipped GPU)
```

## Alur data

```
Input file
  → FileLoader (semua format → list[PIL.Image])
  → ImageConverter (pastikan RGB)
  → PageSplitter (split per halaman → PageItem)
  → Per halaman:
      → ImageNormalizer.for_stage1() (max 1036px)
      → LayoutDetector.detect() → list[dict] bbox + type
      → ReadingOrderSorter.sort()
      → ImageNormalizer.for_stage2() (native res, max 4096px)
      → ContentRecognizer.recognize_batch() → content per elemen
      → OutputAssembler.assemble() → dict sesuai schema
      → OutputValidator.validate_and_raise()
  → Simpan JSON ke output_dir
  → Return dict document-level
```

## 17 class elemen yang dideteksi

```
title, heading_h1, heading_h2, heading_h3,
paragraph, list_item_ordered, list_item_unordered,
table_simple, table_merged, table_borderless,
figure, caption, page_header, page_footer,
page_number, footnote, math_formula
```

## JSON output format (per halaman)

```json
{
  "document_id": "abc123",
  "source_file": "doc.pdf",
  "page_number": 1,
  "page_width": 2480,
  "page_height": 3508,
  "processing_time_ms": 1234.5,
  "model_version": "kayzeeocr-0.1.0",
  "elements": [
    {
      "id": "elem_001",
      "type": "title",
      "bbox": [0.05, 0.03, 0.90, 0.09],
      "bbox_pixel": [124, 74, 2232, 315],
      "reading_order": 1,
      "confidence": 0.97,
      "content": {
        "text": "Judul Dokumen",
        "html": null,
        "latex": null,
        "image_ref": null
      }
    }
  ]
}
```

## Catatan penting untuk developer

1. Model diload SEKALI di setup() dan di-share antara Stage I dan II
   via VisionLanguageEngine. Jangan load ulang di detector/recognizer.

2. bbox disimpan dalam DUA format:
   - bbox: relatif [0.0–1.0] terhadap ukuran halaman
   - bbox_pixel: absolut pixel berdasarkan stage2 image resolution

3. figure tidak di-OCR (early-return di ContentRecognizer.recognize()).
   Hanya bbox dan metadata yang disimpan. Stage2Prompts.for_figure()
   reserved untuk future use.

4. JSON repair: layout_detector._parse_output() punya 4-attempt retry.
   Attempt 4 memanggil model lagi dengan JSON_REPAIR_PROMPT.

5. HEIC/HEIF butuh pillow-heif. Register via register_formats() yang
   dipanggil otomatis saat import loader.py.

6. output_schema.json adalah sumber kebenaran untuk struktur output.
   OutputValidator menggunakannya untuk validasi Draft-7.

## Cara extend

```
Tambah class baru:
  1. Tambahkan ke ELEMENT_TYPES di src/model/prompts.py
  2. Tambahkan definisi ke ELEMENT_DEFINITIONS di prompts.py
  3. Tambahkan prompt di Stage2Prompts jika butuh treatment khusus
  4. Update schemas/output_schema.json (enum type)
  5. Update tests/

Ganti base model (misal ke Qwen3-VL-2B):
  1. Ubah model_name di PipelineConfig (pipeline.py)
  2. Pastikan _resolve_model_class() di vl_engine.py mengenali
     nama model baru (tambahkan ke mapping jika perlu)
  3. Tidak ada perubahan lain yang diperlukan
```
