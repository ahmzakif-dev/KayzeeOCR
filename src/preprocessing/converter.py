"""Image color-mode conversion.

Ensures images are in RGB and free of alpha channels before being fed to the
model (which expects 3-channel RGB input).
"""

from __future__ import annotations

import logging

from PIL import Image

logger = logging.getLogger(__name__)


class ImageConverter:
    """Convert images of any mode into model-ready RGB images."""

    def to_rgb(self, img: Image.Image) -> Image.Image:
        """Convert an image of any mode (RGBA, L, P, CMYK, ...) to RGB.

        Transparent (RGBA / LA / P-with-transparency) images are flattened onto
        a white background so that transparent regions become white rather than
        black.

        Args:
            img: The source PIL image.

        Returns:
            A new RGB :class:`PIL.Image.Image`.
        """
        if img.mode == "RGB":
            return img

        # Expand palette images so transparency is detectable.
        if img.mode == "P":
            img = img.convert("RGBA" if "transparency" in img.info else "RGB")
            if img.mode == "RGB":
                return img

        if img.mode in ("RGBA", "LA"):
            return self._flatten_alpha(img)

        try:
            return img.convert("RGB")
        except Exception:  # noqa: BLE001
            logger.warning("Direct RGB conversion from mode '%s' failed; "
                           "falling back via RGBA.", img.mode)
            return img.convert("RGBA").convert("RGB")

    def to_tensor_ready(self, img: Image.Image) -> Image.Image:
        """Ensure an image is RGB with no alpha channel, ready for tensoring.

        Args:
            img: The source PIL image.

        Returns:
            An RGB image with the alpha channel removed.
        """
        return self.to_rgb(img)

    @staticmethod
    def _flatten_alpha(img: Image.Image) -> Image.Image:
        """Composite an alpha image onto a white background and return RGB."""
        rgba = img.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        composited = Image.alpha_composite(background, rgba)
        return composited.convert("RGB")


def ensure_rgb(img: Image.Image) -> Image.Image:
    """Convenience function: return ``img`` converted to RGB."""
    return ImageConverter().to_rgb(img)
