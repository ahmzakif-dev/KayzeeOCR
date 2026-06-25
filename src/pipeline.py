"""End-to-end Document OCR pipeline orchestrator.

Wires together loading, preprocessing, Stage I (layout), Stage II (content),
post-processing, validation and saving. This is the primary entry point and
also exposes a small CLI.

Data flow::

    file → FileLoader → ImageConverter → PageSplitter
         → [per page] ImageNormalizer
            → Stage I (LayoutDetector)  ── relative + pixel bboxes
            → ReadingOrderSorter
            → Stage II (ContentRecognizer)
            → OutputAssembler → OutputValidator
         → assembled document JSON → saved to output_dir
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .input.loader import FileLoader
from .model.content_recognizer import ContentRecognizer, ContentRecognizerConfig
from .model.layout_detector import LayoutDetector
from .model.vl_engine import EngineConfig, VisionLanguageEngine
from .postprocessing.assembler import OutputAssembler
from .postprocessing.sorter import ReadingOrderSorter
from .postprocessing.validator import OutputValidator, ValidationError
from .preprocessing.converter import ImageConverter
from .preprocessing.normalizer import ImageNormalizer, ResolutionConfig
from .preprocessing.splitter import PageItem, PageSplitter

logger = logging.getLogger(__name__)

MODEL_VERSION = "KayzeeOCR-0.1.0"


@dataclass
class PipelineConfig:
    """End-to-end pipeline configuration.

    Attributes:
        model_name: Base VL model id (Qwen2.5-VL-2B by default).
        device: Device map ("auto", "cuda", "cpu", ...).
        torch_dtype: Torch dtype string ("auto", "float16", ...).
        input_dpi: DPI for rendering PDF / office documents.
        stage1_max_px: Max longest-side pixels for Stage I.
        stage2_max_px: Max longest-side pixels for Stage II.
        crop_padding_px: Padding around Stage II crops.
        output_dir: Directory where result JSON files are written.
        save_cropped_figures: Whether to save cropped figure images.
        validate_output: Whether to validate each page against the schema.
        max_pages: Optional cap on the number of pages to process.
    """

    model_name: str = "Qwen/Qwen2.5-VL-2B-Instruct"
    device: str = "auto"
    torch_dtype: str = "auto"
    input_dpi: int = 150
    stage1_max_px: int = 1036
    stage2_max_px: int = 4096
    crop_padding_px: int = 8
    output_dir: str = "./outputs"
    save_cropped_figures: bool = True
    validate_output: bool = True
    max_pages: int | None = None


class DocumentOCRPipeline:
    """Orchestrates the full document → structured-JSON OCR pipeline."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        """Initialize the pipeline (call :meth:`setup` before processing)."""
        self.config = config or PipelineConfig()

        self._loader: FileLoader | None = None
        self._converter: ImageConverter | None = None
        self._normalizer: ImageNormalizer | None = None
        self._splitter: PageSplitter | None = None
        self._engine: VisionLanguageEngine | None = None
        self._detector: LayoutDetector | None = None
        self._recognizer: ContentRecognizer | None = None
        self._sorter: ReadingOrderSorter | None = None
        self._assembler: OutputAssembler | None = None
        self._validator: OutputValidator | None = None
        self._is_setup = False

    # -- lifecycle --------------------------------------------------------- #

    def setup(self) -> None:
        """Load the model and instantiate every pipeline component.

        Must be called before :meth:`process_file`. Idempotent.
        """
        if self._is_setup:
            return

        logger.info("Setting up KayzeeOCR pipeline (model=%s, device=%s)",
                    self.config.model_name, self.config.device)

        self._loader = FileLoader(dpi=self.config.input_dpi)
        self._converter = ImageConverter()
        self._normalizer = ImageNormalizer(
            ResolutionConfig(
                stage1_max_px=self.config.stage1_max_px,
                stage2_max_px=self.config.stage2_max_px,
            )
        )
        self._splitter = PageSplitter()
        self._sorter = ReadingOrderSorter()
        self._assembler = OutputAssembler()
        self._validator = OutputValidator()

        # One engine, loaded once, shared by both stages (Stage I and Stage II).
        self._engine = VisionLanguageEngine(
            EngineConfig(
                model_name=self.config.model_name,
                device=self.config.device,
                torch_dtype=self.config.torch_dtype,
            )
        )
        self._engine.load()
        self._detector = LayoutDetector(self._engine)
        self._recognizer = ContentRecognizer(
            self._engine,
            ContentRecognizerConfig(crop_padding_px=self.config.crop_padding_px),
        )

        self._setup_output_dir()
        self._is_setup = True
        logger.info("Pipeline ready.")

    def teardown(self) -> None:
        """Release the model and free GPU memory."""
        logger.info("Tearing down pipeline.")
        self._recognizer = None
        self._detector = None
        if self._engine is not None:
            self._engine.unload()
        self._engine = None
        self._is_setup = False

    # -- processing -------------------------------------------------------- #

    def process_file(self, file_path: str | Path) -> dict:
        """Run the full pipeline on ``file_path`` and return the document dict.

        Loads all pages, processes each (continuing past per-page errors), saves
        the assembled document JSON to ``output_dir``, and returns it.
        """
        if not self._is_setup:
            self.setup()

        path = Path(file_path)
        t_start = time.perf_counter()
        logger.info("Processing file: %s", path)

        raw_pages = self._loader.load(path)
        raw_pages = [self._converter.to_rgb(img) for img in raw_pages]
        page_items = self._splitter.split(raw_pages, source_file=str(path))

        if self.config.max_pages is not None:
            page_items = page_items[: self.config.max_pages]

        total = len(page_items)
        page_results: list[dict] = []
        for idx, page_item in enumerate(page_items):
            try:
                result = self.process_page(page_item, idx, total)
                page_results.append(result)
            except Exception as exc:  # noqa: BLE001 - isolate per-page failures
                logger.exception(
                    "Page %d/%d failed: %s", idx + 1, total, exc
                )

        document = self._assembler.assemble_document(page_results)
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        document["processing_time_ms"] = round(elapsed_ms, 3)
        logger.info(
            "Finished '%s': %d/%d page(s) in %.0f ms",
            path.name, len(page_results), total, elapsed_ms,
        )

        self._save_result(document, str(path))
        return document

    def process_page(
        self, page_item: PageItem, page_index: int, total_pages: int
    ) -> dict:
        """Process a single page and return its assembled dict.

        Args:
            page_item: The page to process.
            page_index: 0-based index (for logging "page X of Y").
            total_pages: Total page count (for logging).

        Returns:
            The assembled, optionally validated, page output dict.
        """
        t_page = time.perf_counter()
        logger.info("=== Page %d of %d ===", page_index + 1, total_pages)

        # Stage I works on a downsampled view; Stage II on a higher-res view.
        stage1_img = self._normalizer.for_stage1(page_item.image)
        stage2_img = self._normalizer.for_stage2(page_item.image)

        # Detect against Stage I dims, but express pixels in Stage II space so
        # crops align with the higher-resolution image used for recognition.
        t1 = time.perf_counter()
        elements = self._detector.detect(
            stage1_img, stage2_img.width, stage2_img.height
        )
        t1_ms = (time.perf_counter() - t1) * 1000
        logger.info("Stage I: %d element(s) in %.0f ms", len(elements), t1_ms)

        # Reading order before content recognition.
        elements = self._sorter.sort(elements)

        # Stage II content recognition.
        t2 = time.perf_counter()
        elements = self._recognizer.recognize_batch(stage2_img, elements)
        t2_ms = (time.perf_counter() - t2) * 1000
        logger.info("Stage II: %d element(s) in %.0f ms", len(elements), t2_ms)

        # Save figure crops if requested.
        if self.config.save_cropped_figures:
            self._save_figure_crops(stage2_img, elements, page_item)

        # Page dims reported to the assembler should match the recognition space.
        page_item.original_width = stage2_img.width
        page_item.original_height = stage2_img.height

        page_ms = (time.perf_counter() - t_page) * 1000
        result = self._assembler.assemble(
            page_item, elements, page_ms, MODEL_VERSION
        )

        if self.config.validate_output:
            try:
                self._validator.validate_and_raise(result)
            except ValidationError as exc:
                logger.warning(
                    "Page %d output failed validation: %s",
                    page_index + 1, exc.errors,
                )

        logger.info("Page %d done in %.0f ms", page_index + 1, page_ms)
        return result

    # -- io helpers -------------------------------------------------------- #

    def _setup_output_dir(self) -> None:
        """Ensure the output directory exists."""
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    def _save_result(self, result: dict, source_file: str) -> Path:
        """Write the assembled document JSON to ``output_dir``."""
        stem = Path(source_file).stem
        out_path = Path(self.config.output_dir) / f"{stem}_result.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2)
        logger.info("Saved result → %s", out_path)
        return out_path

    def _save_figure_crops(
        self, page_image, elements: list[dict], page_item: PageItem
    ) -> None:
        """Crop and save figure regions, setting each element's image_ref."""
        fig_dir = Path(self.config.output_dir) / "figures"
        stem = Path(page_item.source_file).stem
        for elem in elements:
            if elem.get("type") != "figure":
                continue
            bbox = elem.get("bbox_pixel")
            if not bbox:
                continue
            fig_dir.mkdir(parents=True, exist_ok=True)
            x1, y1, x2, y2 = (int(v) for v in bbox)
            x1, x2 = max(0, x1), min(page_image.width, x2)
            y1, y2 = max(0, y1), min(page_image.height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = page_image.crop((x1, y1, x2, y2))
            name = f"{stem}_p{page_item.page_number}_{elem.get('id', 'fig')}.png"
            crop_path = fig_dir / name
            crop.save(crop_path)
            elem.setdefault("content", {})
            elem["content"]["image_ref"] = str(crop_path)


def process_document(
    file_path: str, config: PipelineConfig | None = None
) -> dict:
    """Convenience: build, setup, run, teardown the pipeline; return the result."""
    pipeline = DocumentOCRPipeline(config)
    try:
        pipeline.setup()
        return pipeline.process_file(file_path)
    finally:
        pipeline.teardown()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kayzee-ocr",
        description="KayzeeOCR — document OCR to structured JSON.",
    )
    parser.add_argument("file_path", help="Path to the input document/image.")
    parser.add_argument(
        "--model", default=PipelineConfig.model_name, help="Base VL model id."
    )
    parser.add_argument(
        "--device", default=PipelineConfig.device, help="Device (auto/cuda/cpu)."
    )
    parser.add_argument(
        "--output-dir", default=PipelineConfig.output_dir, help="Output directory."
    )
    parser.add_argument(
        "--dpi", type=int, default=PipelineConfig.input_dpi, help="PDF render DPI."
    )
    parser.add_argument(
        "--max-pages", type=int, default=None, help="Limit number of pages."
    )
    parser.add_argument(
        "--no-validate", action="store_true", help="Skip schema validation."
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable DEBUG logging."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = PipelineConfig(
        model_name=args.model,
        device=args.device,
        output_dir=args.output_dir,
        input_dpi=args.dpi,
        max_pages=args.max_pages,
        validate_output=not args.no_validate,
    )

    try:
        result = process_document(args.file_path, config)
    except FileNotFoundError:
        logger.exception("Input file not found.")
        return 2
    except Exception as exc:  # noqa: BLE001
        logger.exception("Pipeline failed: %s", exc)
        return 1

    print(
        f"Done: {result.get('total_pages', 0)} page(s), "
        f"{result.get('total_elements', 0)} element(s). "
        f"Output in '{config.output_dir}'."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
