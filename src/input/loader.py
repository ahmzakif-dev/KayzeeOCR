"""Input file loading.

Accepts any supported file path and returns a list of RGB :class:`PIL.Image.Image`
objects, one per page. Routing is by file extension.

Supported formats:
    * Images: JPEG, JPG, PNG, WEBP, BMP, GIF, TIFF, TIF
    * Apple:  HEIC, HEIF (via ``pillow-heif``)
    * Docs:   PDF (via PyMuPDF/fitz), DOCX, PPTX (rendered via LibreOffice when
              available; DOCX falls back to a rendered text image otherwise)
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# Tracks whether the HEIF/HEIC opener has been registered with Pillow.
_HEIF_REGISTERED = False

# Extensions handled directly by Pillow (after HEIF registration).
_IMAGE_EXTS = {
    ".jpeg",
    ".jpg",
    ".png",
    ".webp",
    ".bmp",
    ".gif",
    ".tiff",
    ".tif",
    ".heic",
    ".heif",
}
_PDF_EXTS = {".pdf"}
_OFFICE_EXTS = {".docx", ".pptx"}


class UnsupportedFormatError(Exception):
    """Raised when a file extension is not supported by :class:`FileLoader`."""


def register_formats() -> None:
    """Register the HEIC/HEIF opener with Pillow (idempotent).

    Safe to call multiple times; only the first call performs registration.
    If ``pillow-heif`` is not installed, a warning is logged and HEIC/HEIF
    loading will fail later with a clear error.
    """
    global _HEIF_REGISTERED
    if _HEIF_REGISTERED:
        return
    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
        _HEIF_REGISTERED = True
        logger.debug("Registered HEIF/HEIC opener with Pillow.")
    except ImportError:
        logger.warning(
            "pillow-heif is not installed; HEIC/HEIF files cannot be loaded. "
            "Install it with `pip install pillow-heif`."
        )


# Register on import so HEIC/HEIF work out of the box.
register_formats()


class FileLoader:
    """Load document/image files into lists of RGB PIL Images (one per page)."""

    def __init__(self, dpi: int = 150) -> None:
        """Initialize the loader.

        Args:
            dpi: Target DPI used when rendering PDF (and office) pages to images.
        """
        self.dpi = dpi

    # -- public API -------------------------------------------------------- #

    def load(self, file_path: str | Path) -> list[Image.Image]:
        """Load ``file_path`` and return a list of RGB PIL Images, one per page.

        Args:
            file_path: Path to a supported file.

        Returns:
            List of :class:`PIL.Image.Image` in RGB mode (length >= 1).

        Raises:
            FileNotFoundError: If the file does not exist.
            UnsupportedFormatError: If the extension is not supported.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

        ext = path.suffix.lower()
        logger.info("Loading file '%s' (ext=%s)", path.name, ext)

        if ext in _IMAGE_EXTS:
            images = self._load_image(path)
        elif ext in _PDF_EXTS:
            images = self._load_pdf(path)
        elif ext in _OFFICE_EXTS:
            images = self._load_office(path)
        else:
            raise UnsupportedFormatError(
                f"Unsupported file format '{ext}'. Supported: "
                f"{', '.join(self.get_supported_formats())}"
            )

        rgb_images = [self._ensure_rgb(img) for img in images]
        logger.info("Loaded %d page(s) from '%s'", len(rgb_images), path.name)
        return rgb_images

    def get_supported_formats(self) -> list[str]:
        """Return the sorted list of supported file extensions (with dots)."""
        return sorted(_IMAGE_EXTS | _PDF_EXTS | _OFFICE_EXTS)

    # -- format handlers --------------------------------------------------- #

    def _load_image(self, path: Path) -> list[Image.Image]:
        """Load a single image file. Multi-frame (GIF/TIFF) → one image per frame."""
        ext = path.suffix.lower()
        if ext in {".heic", ".heif"} and not _HEIF_REGISTERED:
            register_formats()
            if not _HEIF_REGISTERED:
                raise UnsupportedFormatError(
                    f"Cannot load '{path.name}': pillow-heif is required for "
                    "HEIC/HEIF support. Install it with `pip install pillow-heif`."
                )

        try:
            img = Image.open(path)
        except Exception as exc:  # noqa: BLE001 - re-raise with context
            raise UnsupportedFormatError(
                f"Failed to open image '{path.name}': {exc}"
            ) from exc

        frames: list[Image.Image] = []
        n_frames = getattr(img, "n_frames", 1)
        if n_frames > 1:
            logger.debug("'%s' has %d frames", path.name, n_frames)
            for i in range(n_frames):
                img.seek(i)
                frames.append(img.convert("RGB"))
        else:
            frames.append(img.convert("RGB"))
        return frames

    def _load_pdf(self, path: Path) -> list[Image.Image]:
        """Render every page of a PDF to an RGB image at ``self.dpi``."""
        try:
            import fitz  # PyMuPDF
        except ImportError as exc:
            raise UnsupportedFormatError(
                "PyMuPDF (pymupdf) is required to load PDF files. "
                "Install it with `pip install pymupdf`."
            ) from exc

        zoom = self.dpi / 72.0  # PDF base resolution is 72 DPI.
        matrix = fitz.Matrix(zoom, zoom)
        images: list[Image.Image] = []
        with fitz.open(path) as doc:
            logger.debug("PDF '%s' has %d page(s)", path.name, doc.page_count)
            for page_index in range(doc.page_count):
                page = doc.load_page(page_index)
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                images.append(img)
        if not images:
            raise UnsupportedFormatError(f"PDF '{path.name}' contains no pages.")
        return images

    def _load_office(self, path: Path) -> list[Image.Image]:
        """Load a DOCX/PPTX file.

        Strategy:
            1. If LibreOffice (``soffice``) is available, convert the document to
               PDF and render that PDF (best fidelity).
            2. Otherwise, for DOCX, fall back to extracting text and rendering it
               onto a simple page image (PPTX has no text fallback).
        """
        if self._libreoffice_available():
            try:
                return self._load_office_via_libreoffice(path)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "LibreOffice conversion of '%s' failed (%s); falling back.",
                    path.name,
                    exc,
                )

        if path.suffix.lower() == ".docx":
            logger.info("Using text-extraction fallback for '%s'.", path.name)
            return self._load_docx(path)

        raise UnsupportedFormatError(
            f"Cannot render '{path.name}': LibreOffice is not available and no "
            "fallback exists for this format. Install LibreOffice for PPTX/DOCX "
            "rendering."
        )

    def _load_docx(self, path: Path) -> list[Image.Image]:
        """Fallback DOCX loader: extract text and render to image pages.

        This is a low-fidelity fallback used only when LibreOffice is absent.
        """
        try:
            import docx  # python-docx
        except ImportError as exc:
            raise UnsupportedFormatError(
                "python-docx is required for the DOCX text fallback. "
                "Install it with `pip install python-docx`."
            ) from exc

        document = docx.Document(str(path))
        paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
        text = "\n".join(paragraphs) if paragraphs else "(empty document)"
        return self._render_text_to_images(text)

    def _load_office_via_libreoffice(self, path: Path) -> list[Image.Image]:
        """Convert an office document to PDF via LibreOffice, then render it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [
                self._soffice_binary(),
                "--headless",
                "--convert-to",
                "pdf",
                "--outdir",
                tmpdir,
                str(path),
            ]
            logger.debug("Running LibreOffice: %s", " ".join(cmd))
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                timeout=120,
            )
            pdf_path = Path(tmpdir) / (path.stem + ".pdf")
            if not pdf_path.exists():
                raise RuntimeError("LibreOffice did not produce a PDF.")
            return self._load_pdf(pdf_path)

    # -- helpers ----------------------------------------------------------- #

    @staticmethod
    def _ensure_rgb(img: Image.Image) -> Image.Image:
        """Return ``img`` in RGB mode (convert if needed)."""
        return img if img.mode == "RGB" else img.convert("RGB")

    @staticmethod
    def _libreoffice_available() -> bool:
        """Return True if a LibreOffice ``soffice`` binary is on PATH."""
        return FileLoader._soffice_binary() is not None

    @staticmethod
    def _soffice_binary() -> str | None:
        """Locate the LibreOffice binary, if installed."""
        for name in ("soffice", "libreoffice"):
            found = shutil.which(name)
            if found:
                return found
        return None

    @staticmethod
    def _render_text_to_images(
        text: str,
        page_size: tuple[int, int] = (1240, 1754),  # ~A4 @ 150 DPI
        margin: int = 60,
        line_height: int = 22,
    ) -> list[Image.Image]:
        """Render plain text onto one or more white RGB page images."""
        width, height = page_size
        usable_lines = max(1, (height - 2 * margin) // line_height)

        # Naive wrapping by characters-per-line estimate.
        chars_per_line = max(20, (width - 2 * margin) // 9)
        wrapped: list[str] = []
        for raw_line in text.split("\n"):
            if not raw_line:
                wrapped.append("")
                continue
            for i in range(0, len(raw_line), chars_per_line):
                wrapped.append(raw_line[i : i + chars_per_line])

        try:
            font = ImageFont.load_default()
        except Exception:  # noqa: BLE001
            font = None

        images: list[Image.Image] = []
        for start in range(0, max(len(wrapped), 1), usable_lines):
            chunk = wrapped[start : start + usable_lines]
            page = Image.new("RGB", page_size, "white")
            draw = ImageDraw.Draw(page)
            y = margin
            for line in chunk:
                draw.text((margin, y), line, fill="black", font=font)
                y += line_height
            images.append(page)
        return images


def load_file(file_path: str | Path, dpi: int = 150) -> list[Image.Image]:
    """Convenience function: load ``file_path`` into a list of RGB PIL Images.

    Args:
        file_path: Path to a supported file.
        dpi: Target DPI for PDF/office rendering.

    Returns:
        List of RGB :class:`PIL.Image.Image`, one per page.
    """
    return FileLoader(dpi=dpi).load(file_path)
