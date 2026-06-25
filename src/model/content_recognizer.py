"""Stage II — content recognition.

For each element detected in Stage I, crops its region from the full-resolution
page image, picks a type-specific prompt, runs the shared vision-language
engine, and attaches a ``content`` dict to the element.

Figures are intentionally not OCR'd; only their coordinates are kept (the
pipeline saves the cropped image and fills ``image_ref`` separately).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from PIL import Image

from .parsing import strip_code_fence
from .prompts import Stage2Prompts
from .vl_engine import VisionLanguageEngine

logger = logging.getLogger(__name__)

# Element type that is not OCR'd (only its bounding box is kept).
_FIGURE_TYPE = "figure"

# Matches an HTML <table>...</table> block in a recognizer response.
_TABLE_RE = re.compile(r"<table.*?</table>", re.DOTALL | re.IGNORECASE)

# Match a leading / trailing $ or $$ math delimiter (kept as two simple,
# single-anchor patterns to avoid ambiguous anchor + alternation precedence).
_LEADING_MATH_RE = re.compile(r"^\${1,2}")
_TRAILING_MATH_RE = re.compile(r"\${1,2}$")


@dataclass
class ContentRecognizerConfig:
    """Per-element-type generation settings for content recognition.

    Attributes:
        max_new_tokens_text: Token budget for text elements.
        max_new_tokens_table: Token budget for tables (HTML can be long).
        max_new_tokens_math: Token budget for LaTeX formulas.
        max_new_tokens_figure: Token budget for figure descriptions.
        crop_padding_px: Padding added around each crop, clamped to the image.
    """

    max_new_tokens_text: int = 512
    max_new_tokens_table: int = 2048
    max_new_tokens_math: int = 512
    max_new_tokens_figure: int = 256
    crop_padding_px: int = 8


def _empty_content() -> dict:
    """Return a fresh, fully-null content dict matching the output schema."""
    return {"text": None, "html": None, "latex": None, "image_ref": None}


class ContentRecognizer:
    """Recognize each detected element's content with the shared engine."""

    def __init__(
        self,
        engine: VisionLanguageEngine,
        config: ContentRecognizerConfig | None = None,
    ) -> None:
        """Initialize the recognizer with a shared inference engine.

        Args:
            engine: The shared vision-language engine (same instance as Stage I).
            config: Recognition settings. Defaults to
                :class:`ContentRecognizerConfig`.
        """
        self.engine = engine
        self.config = config or ContentRecognizerConfig()

    def recognize(self, page_image: Image.Image, element: dict) -> dict:
        """Recognize one element's content and return an enriched copy.

        Args:
            page_image: The full-resolution (Stage II) page image.
            element: An element dict with a ``bbox_pixel`` and ``type``.

        Returns:
            A shallow copy of ``element`` with its ``content`` field populated.
            The caller's dict is never mutated.
        """
        enriched = dict(element)
        enriched.setdefault("content", _empty_content())
        element_type = enriched.get("type", "")

        if element_type == _FIGURE_TYPE:
            # Figure elements are not passed through OCR by design.
            # bbox and metadata are preserved; content fields remain null.
            # Stage2Prompts.for_figure() is reserved for future use.
            logger.debug("Skipping OCR for figure %s.", enriched.get("id"))
            return enriched

        bbox_pixel = enriched.get("bbox_pixel")
        if not bbox_pixel:
            logger.warning(
                "Element %s has no bbox_pixel; skipping.", enriched.get("id")
            )
            return enriched

        crop = self._crop_region(page_image, bbox_pixel)
        prompt = Stage2Prompts.get_prompt(element_type)
        raw_output = self.engine.generate(
            [self.engine.image_text_message(crop, prompt)],
            self._max_tokens_for(element_type),
        )
        enriched["content"] = self._parse_content(element_type, raw_output)
        return enriched

    def recognize_batch(
        self, page_image: Image.Image, elements: list[dict]
    ) -> list[dict]:
        """Recognize all elements sequentially, logging per-element progress.

        Args:
            page_image: The full-resolution (Stage II) page image.
            elements: The detected elements to recognize.

        Returns:
            A new list of enriched element dicts in the original order.
        """
        total = len(elements)
        recognized = []
        for position, element in enumerate(elements, start=1):
            logger.info(
                "Stage II [%d/%d] recognizing %s (%s)",
                position,
                total,
                element.get("id"),
                element.get("type"),
            )
            recognized.append(self.recognize(page_image, element))
        return recognized

    def _crop_region(
        self, image: Image.Image, bbox_pixel: list[int]
    ) -> Image.Image:
        """Crop a padded region from ``image``, clamped to the image bounds.

        Args:
            image: The page image to crop from.
            bbox_pixel: The ``[x1, y1, x2, y2]`` pixel box to crop.

        Returns:
            The cropped image. Degenerate boxes are widened to at least 1 px.
        """
        padding = self.config.crop_padding_px
        x1, y1, x2, y2 = (int(value) for value in bbox_pixel)
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(image.width, x2 + padding)
        y2 = min(image.height, y2 + padding)
        if x2 <= x1:
            x2 = min(image.width, x1 + 1)
        if y2 <= y1:
            y2 = min(image.height, y1 + 1)
        return image.crop((x1, y1, x2, y2))

    def _max_tokens_for(self, element_type: str) -> int:
        """Return the token budget appropriate for ``element_type``."""
        if element_type in Stage2Prompts.TABLE_TYPES:
            return self.config.max_new_tokens_table
        if element_type == "math_formula":
            return self.config.max_new_tokens_math
        if element_type == _FIGURE_TYPE:
            return self.config.max_new_tokens_figure
        return self.config.max_new_tokens_text

    def _parse_content(self, element_type: str, raw_output: str) -> dict:
        """Parse raw output into a content dict based on the element type.

        Args:
            element_type: The element's layout class.
            raw_output: The raw decoded model text.

        Returns:
            A content dict with exactly one populated field.
        """
        content = _empty_content()
        cleaned = strip_code_fence(raw_output).strip()

        if element_type in Stage2Prompts.TABLE_TYPES:
            content["html"] = self._extract_table_html(cleaned)
        elif element_type == "math_formula":
            content["latex"] = self._extract_latex(cleaned)
        else:
            # Text-like elements and figure descriptions share plain-text output.
            content["text"] = cleaned or None
        return content

    @staticmethod
    def _extract_table_html(text: str) -> str | None:
        """Return the ``<table>...</table>`` block, or the text as a fallback."""
        match = _TABLE_RE.search(text)
        if match:
            return match.group(0)
        return text or None

    @staticmethod
    def _extract_latex(text: str) -> str | None:
        """Strip surrounding ``$``/``$$`` delimiters from a LaTeX string."""
        stripped = _LEADING_MATH_RE.sub("", text)
        stripped = _TRAILING_MATH_RE.sub("", stripped)
        return stripped.strip() or None


def create_recognizer(engine: VisionLanguageEngine) -> ContentRecognizer:
    """Convenience factory untuk penggunaan standalone atau testing.

    Pipeline utama (pipeline.py) membangun objek ini secara langsung.

    Contoh:
        engine = create_engine("Qwen/Qwen2.5-VL-2B-Instruct")
        recognizer = create_recognizer(engine)

    Args:
        engine: The shared vision-language engine.

    Returns:
        A ready-to-use recognizer.
    """
    return ContentRecognizer(engine)
