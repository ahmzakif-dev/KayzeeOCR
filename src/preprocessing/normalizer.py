"""Resolution normalization.

Two different resolution regimes are needed:

* **Stage I (layout detection)** works best on a downsampled view of the whole
  page — capped at ``stage1_max_px`` on the longest side.
* **Stage II (content recognition)** needs enough detail to read text, so the
  longest side is pushed up to at least ``stage2_min_px`` but capped at
  ``stage2_max_px`` to avoid out-of-memory errors.

Aspect ratio is always preserved.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class ResolutionConfig:
    """Resolution limits for the two inference stages.

    Attributes:
        stage1_max_px: Max longest-side pixels for Stage I (downsample target).
        stage2_min_px: Min longest-side pixels for Stage II (upsample floor).
        stage2_max_px: Max longest-side pixels for Stage II (OOM guard).
        interpolation: PIL resampling filter (default LANCZOS).
    """

    stage1_max_px: int = 1036
    stage2_min_px: int = 1036
    stage2_max_px: int = 4096
    interpolation: int = Image.LANCZOS


class ImageNormalizer:
    """Resize images for Stage I / Stage II while preserving aspect ratio."""

    def __init__(self, config: ResolutionConfig | None = None) -> None:
        """Initialize with an optional :class:`ResolutionConfig`."""
        self.config = config or ResolutionConfig()

    def for_stage1(self, img: Image.Image) -> Image.Image:
        """Downsample ``img`` so its longest side is <= ``stage1_max_px``.

        Smaller images are returned unchanged (no upscaling for Stage I).
        """
        return self._resize_to_max(img, self.config.stage1_max_px)

    def for_stage2(self, img: Image.Image) -> Image.Image:
        """Normalize ``img`` for Stage II.

        Ensures the longest side is at least ``stage2_min_px`` (upscaling small
        crops) and at most ``stage2_max_px`` (downscaling huge crops).
        """
        # Cap first to avoid OOM, then ensure the minimum.
        capped = self._resize_to_max(img, self.config.stage2_max_px)
        return self._resize_to_min(capped, self.config.stage2_min_px)

    def get_scale_factor(
        self, original: Image.Image, resized: Image.Image
    ) -> tuple[float, float]:
        """Return ``(scale_x, scale_y)`` mapping ``original`` → ``resized``.

        Multiply original coordinates by these factors to get resized
        coordinates (and divide to go the other way).
        """
        scale_x = resized.width / original.width
        scale_y = resized.height / original.height
        return scale_x, scale_y

    # -- internals --------------------------------------------------------- #

    def _resize_to_max(self, img: Image.Image, max_px: int) -> Image.Image:
        """Shrink so the longest side <= ``max_px``; no-op if already smaller."""
        longest = max(img.width, img.height)
        if longest <= max_px:
            return img
        scale = max_px / longest
        new_size = (
            max(1, round(img.width * scale)),
            max(1, round(img.height * scale)),
        )
        logger.debug("Downsampling %s -> %s (max_px=%d)", img.size, new_size, max_px)
        return img.resize(new_size, self.config.interpolation)

    def _resize_to_min(self, img: Image.Image, min_px: int) -> Image.Image:
        """Enlarge so the longest side >= ``min_px``; no-op if already larger."""
        longest = max(img.width, img.height)
        if longest >= min_px:
            return img
        scale = min_px / longest
        new_size = (
            max(1, round(img.width * scale)),
            max(1, round(img.height * scale)),
        )
        logger.debug("Upsampling %s -> %s (min_px=%d)", img.size, new_size, min_px)
        return img.resize(new_size, self.config.interpolation)
