"""Stage I — document layout detection.

Runs a single vision-language generation pass over a page image to detect every
layout element, then parses the JSON response into element dicts carrying both
relative (0-1) and absolute-pixel bounding boxes.

The actual model inference is delegated to a shared
:class:`~src.model.vl_engine.VisionLanguageEngine`, so the same weights serve
both this stage and Stage II content recognition.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from PIL import Image

from .parsing import extract_json_object, strip_code_fence
from .prompts import (
    ELEMENT_TYPES,
    JSON_REPAIR_PROMPT,
    STAGE1_SYSTEM_PROMPT,
    STAGE1_USER_PROMPT,
)
from .vl_engine import VisionLanguageEngine, create_engine

logger = logging.getLogger(__name__)

_VALID_TYPES = frozenset(ELEMENT_TYPES)


@dataclass
class LayoutDetectorConfig:
    """Generation settings specific to layout detection.

    Attributes:
        max_new_tokens: Token budget for the layout JSON response.
        do_sample: Whether to sample. ``False`` keeps detection deterministic.
    """

    max_new_tokens: int = 2048
    do_sample: bool = False


class LayoutDetector:
    """Detect and classify layout elements on a document page image."""

    def __init__(
        self,
        engine: VisionLanguageEngine,
        config: LayoutDetectorConfig | None = None,
    ) -> None:
        """Initialize the detector with a shared inference engine.

        Args:
            engine: The shared vision-language engine.
            config: Detection settings. Defaults to
                :class:`LayoutDetectorConfig`.
        """
        self.engine = engine
        self.config = config or LayoutDetectorConfig()

    def detect(
        self, image: Image.Image, page_width: int, page_height: int
    ) -> list[dict]:
        """Detect layout elements on ``image``.

        Args:
            image: The (Stage I-normalized) page image.
            page_width: Page width in pixels, used to scale boxes to absolute.
            page_height: Page height in pixels, used to scale boxes to absolute.

        Returns:
            Element dicts ``{id, type, bbox, bbox_pixel, reading_order}`` where
            ``bbox`` is relative (0-1) and ``bbox_pixel`` is absolute. Returns an
            empty list when the response cannot be parsed.
        """
        messages = [
            {"role": "system", "content": STAGE1_SYSTEM_PROMPT},
            self.engine.image_text_message(image, STAGE1_USER_PROMPT),
        ]
        raw_output = self.engine.generate(
            messages, self.config.max_new_tokens, self.config.do_sample
        )
        elements = self._parse_output(raw_output, page_width, page_height)
        logger.info("Stage I detected %d element(s).", len(elements))
        return elements

    def _parse_output(
        self, raw_output: str, page_width: int, page_height: int
    ) -> list[dict]:
        """Parse raw model text into validated element dicts.

        Tolerant of extra prose or code fences around the JSON. Elements with an
        unknown type or an invalid box are dropped.

        Args:
            raw_output: The raw decoded model text.
            page_width: Page width in pixels.
            page_height: Page height in pixels.

        Returns:
            The list of valid element dicts (possibly empty).
        """
        payload = self._load_json_with_retry(raw_output)
        if payload is None:
            return []

        raw_elements = payload.get("elements") if isinstance(payload, dict) else None
        if not isinstance(raw_elements, list):
            logger.warning("Stage I: 'elements' is missing or not a list.")
            return []

        parsed = (
            self._normalize_element(raw, index, page_width, page_height)
            for index, raw in enumerate(raw_elements, start=1)
        )
        return [element for element in parsed if element is not None]

    def _load_json_with_retry(self, raw_output: str) -> dict | None:
        """Parse JSON from raw model text via a 4-attempt repair chain.

        The attempts escalate in cost: a direct parse, fence stripping, brace
        extraction, and finally a model-driven repair of the malformed text.

        Args:
            raw_output: The raw decoded model text.

        Returns:
            The parsed JSON value (typically a dict), or ``None`` if every
            attempt fails.
        """
        # Attempt 1: parse the raw output directly.
        try:
            return json.loads(raw_output)
        except (json.JSONDecodeError, TypeError) as error:
            logger.debug("Attempt 1 (raw json.loads) failed: %s", error)

        # Attempt 2: strip a surrounding markdown code fence, then parse.
        try:
            return json.loads(strip_code_fence(raw_output))
        except (json.JSONDecodeError, TypeError) as error:
            logger.debug("Attempt 2 (strip_code_fence) failed: %s", error)

        # Attempt 3: extract the outermost {...} object, then parse.
        try:
            return json.loads(extract_json_object(raw_output))
        except (json.JSONDecodeError, TypeError) as error:
            logger.debug("Attempt 3 (extract_json_object) failed: %s", error)

        # Attempt 4: ask the model to repair the malformed JSON.
        try:
            repair_prompt = JSON_REPAIR_PROMPT.format(malformed=raw_output[:800])
            repaired = self.engine.generate(
                [{"role": "user", "content": repair_prompt}],
                self.config.max_new_tokens,
            )
            return json.loads(extract_json_object(repaired))
        except Exception as error:  # noqa: BLE001 - repair is best-effort
            logger.debug("Attempt 4 (model repair) failed: %s", error)

        logger.warning(
            "JSON parse failed after 4 attempts. raw[:200]=%s", raw_output[:200]
        )
        return None

    def _normalize_element(
        self, raw: dict, index: int, page_width: int, page_height: int
    ) -> dict | None:
        """Validate and normalize one raw element, or return ``None`` to drop it.

        Args:
            raw: A single raw element dict from the model.
            index: 1-based position, used for fallback id and reading order.
            page_width: Page width in pixels.
            page_height: Page height in pixels.

        Returns:
            A normalized element dict, or ``None`` if the element is invalid.
        """
        if not isinstance(raw, dict):
            return None

        element_type = raw.get("type")
        if element_type not in _VALID_TYPES:
            logger.debug("Dropping element with invalid type: %r", element_type)
            return None

        relative_bbox = self._sanitize_bbox(raw.get("bbox"))
        if relative_bbox is None:
            logger.debug("Dropping element with invalid bbox: %r", raw.get("bbox"))
            return None

        return {
            "id": self._element_id(raw.get("id"), index),
            "type": element_type,
            "bbox": relative_bbox,
            "bbox_pixel": self._to_pixel_bbox(
                relative_bbox, page_width, page_height
            ),
            "reading_order": self._reading_order(raw.get("reading_order"), index),
        }

    @staticmethod
    def _sanitize_bbox(bbox: object) -> list[float] | None:
        """Validate, clamp to 0-1, and order a relative bounding box.

        Args:
            bbox: A candidate ``[x1, y1, x2, y2]`` box.

        Returns:
            The cleaned box, or ``None`` if it is not four numbers.
        """
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None
        if not all(isinstance(value, (int, float)) for value in bbox):
            return None

        clamp = lambda value: max(0.0, min(1.0, float(value)))  # noqa: E731
        x1, x2 = sorted((clamp(bbox[0]), clamp(bbox[2])))
        y1, y2 = sorted((clamp(bbox[1]), clamp(bbox[3])))
        return [x1, y1, x2, y2]

    @staticmethod
    def _to_pixel_bbox(
        bbox: list[float], width: int, height: int
    ) -> list[int]:
        """Convert a relative box to absolute pixel integers.

        Args:
            bbox: A relative ``[x1, y1, x2, y2]`` box in 0-1.
            width: Page width in pixels.
            height: Page height in pixels.

        Returns:
            The box as ``[x1, y1, x2, y2]`` pixel integers.
        """
        x1, y1, x2, y2 = bbox
        return [
            round(x1 * width),
            round(y1 * height),
            round(x2 * width),
            round(y2 * height),
        ]

    @staticmethod
    def _element_id(raw_id: object, index: int) -> str:
        """Return a valid element id, generating one from ``index`` if needed."""
        if isinstance(raw_id, str) and raw_id:
            return raw_id
        return f"elem_{index:03d}"

    @staticmethod
    def _reading_order(raw_order: object, index: int) -> int:
        """Coerce the model's reading order to a positive int, defaulting to index."""
        try:
            return max(1, int(raw_order))
        except (TypeError, ValueError):
            return index


def create_detector(
    model_name: str | None = None, device: str = "auto"
) -> LayoutDetector:
    """Convenience factory untuk penggunaan standalone atau testing.

    Pipeline utama (pipeline.py) membangun objek ini secara langsung
    (dengan engine yang dibagi antara Stage I dan Stage II).

    Contoh:
        detector = create_detector("Qwen/Qwen2.5-VL-2B-Instruct")

    Args:
        model_name: Optional model id override.
        device: Device map for the model.

    Returns:
        A ready-to-use detector backed by a fresh engine.
    """
    return LayoutDetector(create_engine(model_name, device))
