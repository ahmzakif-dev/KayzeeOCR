"""All prompt strings used to drive the Qwen-VL model (Stage I and Stage II).

Centralizing prompts here keeps them versionable and avoids hard-coded strings
scattered across the model code.

Usage map:
    * ``STAGE1_SYSTEM_PROMPT`` / ``STAGE1_USER_PROMPT`` — layout detection
      (one call per page, see ``layout_detector.LayoutDetector``).
    * ``Stage2Prompts`` — content recognition, one prompt per detected element
      (see ``content_recognizer.ContentRecognizer``).
    * ``JSON_REPAIR_PROMPT`` — recover from malformed JSON returned by Stage I.

All prompts are written in English for best model compatibility.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Element taxonomy
# --------------------------------------------------------------------------- #

#: The 17 layout element classes KayzeeOCR detects. Must stay in sync with
#: ``schemas/output_schema.json`` (element ``type`` enum).
ELEMENT_TYPES: list[str] = [
    "title",
    "heading_h1",
    "heading_h2",
    "heading_h3",
    "paragraph",
    "list_item_ordered",
    "list_item_unordered",
    "table_simple",
    "table_merged",
    "table_borderless",
    "page_header",
    "page_footer",
    "page_number",
    "figure",
    "caption",
    "footnote",
    "math_formula",
]

#: Short, single-line definition of each element type, embedded into the Stage I
#: system prompt so the model classifies consistently.
ELEMENT_DEFINITIONS: dict[str, str] = {
    "title": "The main document title, usually the largest text near the top.",
    "heading_h1": "A top-level section heading.",
    "heading_h2": "A second-level subsection heading.",
    "heading_h3": "A third-level sub-subsection heading.",
    "paragraph": "A normal block of body text.",
    "list_item_ordered": "An item in a numbered/ordered list (1., 2., a), ...).",
    "list_item_unordered": "An item in a bulleted/unordered list (-, *, •).",
    "table_simple": "A table with a regular grid and no merged cells.",
    "table_merged": "A table that contains merged (rowspan/colspan) cells.",
    "table_borderless": "A table laid out by alignment with no visible borders.",
    "page_header": "Running header text at the top margin of the page.",
    "page_footer": "Running footer text at the bottom margin of the page.",
    "page_number": "The page number, usually isolated in a margin.",
    "figure": "An image, photo, chart, diagram or logo.",
    "caption": "A short text describing a figure or table.",
    "footnote": "A footnote at the bottom of the page, often referenced by a marker.",
    "math_formula": "A mathematical equation or formula, inline-block or display.",
}


def _format_element_definitions() -> str:
    """Render ELEMENT_DEFINITIONS as a bullet list for prompt embedding."""
    return "\n".join(f"- {name}: {desc}" for name, desc in ELEMENT_DEFINITIONS.items())


# --------------------------------------------------------------------------- #
# Stage I — layout detection
# --------------------------------------------------------------------------- #

STAGE1_SYSTEM_PROMPT: str = (
    "You are a precise document layout analysis engine. Given an image of a "
    "single document page, you detect every layout element, classify each one, "
    "and report its bounding box and reading order.\n\n"
    "You MUST output VALID JSON ONLY. Do not wrap the JSON in markdown, do not "
    "use code fences, do not add explanations, comments, or any text before or "
    "after the JSON.\n\n"
    "Each element's `type` MUST be exactly one of the following 17 classes:\n"
    f"{_format_element_definitions()}\n\n"
    "Be exhaustive: detect headers, footers, page numbers, captions and "
    "footnotes as well as the main body content. Pay special attention to "
    "borderless tables, which are tables aligned by whitespace with no visible "
    "grid lines."
)

STAGE1_USER_PROMPT: str = (
    "Detect ALL layout elements on this page. Do not skip any element.\n\n"
    "Return a single JSON object with exactly this shape:\n"
    "{\n"
    '  "elements": [\n'
    '    {"id": "elem_001", "type": "<one of the 17 types>", '
    '"bbox": [x1, y1, x2, y2], "reading_order": 1}\n'
    "  ]\n"
    "}\n\n"
    "Rules:\n"
    "- `bbox` is [x1, y1, x2, y2] as floats in 0.0-1.0, relative to the page "
    "size (x is horizontal from the left, y is vertical from the top). "
    "x1<x2 and y1<y2.\n"
    "- `id` is sequential: elem_001, elem_002, elem_003, ...\n"
    "- `reading_order` starts at 1 and follows natural reading flow "
    "(top-to-bottom, left-to-right; for multi-column layouts read each column "
    "top-to-bottom before moving to the next column).\n"
    "- Use the most specific type that applies.\n"
    "- Output VALID JSON ONLY, nothing else."
)


# --------------------------------------------------------------------------- #
# Stage II — content recognition (per element)
# --------------------------------------------------------------------------- #


class Stage2Prompts:
    """Factory of per-element-type prompts for Stage II content recognition.

    Each method returns the user prompt for one category of element. Use
    :meth:`get_prompt` to route an element ``type`` to the right prompt.
    """

    # Element types handled as plain text OCR.
    TEXT_TYPES: frozenset[str] = frozenset(
        {
            "title",
            "heading_h1",
            "heading_h2",
            "heading_h3",
            "paragraph",
            "list_item_ordered",
            "list_item_unordered",
            "caption",
            "footnote",
        }
    )

    # Element types treated as page furniture (header/footer/number).
    PAGE_TYPES: frozenset[str] = frozenset(
        {"page_header", "page_footer", "page_number"}
    )

    # Table element types.
    TABLE_TYPES: frozenset[str] = frozenset(
        {"table_simple", "table_merged", "table_borderless"}
    )

    @staticmethod
    def for_text(element_type: str) -> str:
        """Prompt for plain-text OCR of a text element.

        Used for titles, headings, paragraphs, list items, captions, footnotes.
        """
        return (
            "This image is a cropped region of a document containing a "
            f"'{element_type}'. Transcribe ALL the text in this region exactly "
            "as it appears, preserving the original language, punctuation, "
            "casing, and line breaks. Do not translate. Do not summarize. Do "
            "not add markdown or any commentary. Output ONLY the transcribed "
            "text."
        )

    @staticmethod
    def for_table(element_type: str) -> str:
        """Prompt for table recognition, returning an HTML table.

        Handles simple, merged, and borderless tables. Merged cells must use
        rowspan/colspan.
        """
        hint = ""
        if element_type == "table_borderless":
            hint = (
                " This is a borderless table aligned by whitespace; infer the "
                "row and column structure from the alignment of the text."
            )
        elif element_type == "table_merged":
            hint = (
                " This table contains merged cells; represent them faithfully "
                "with rowspan and/or colspan attributes."
            )
        return (
            "This image is a cropped table from a document. Convert it to a "
            "single valid HTML <table> element. Use <thead>/<tbody>, <tr>, "
            "<th> for header cells and <td> for data cells. Use rowspan and "
            "colspan attributes to represent merged cells. Preserve the cell "
            "text exactly, in its original language. Output ONLY the HTML "
            "<table>...</table>, with no markdown fences or extra text." + hint
        )

    @staticmethod
    def for_math() -> str:
        """Prompt for converting a math formula region into LaTeX."""
        return (
            "This image is a cropped mathematical formula. Convert it into a "
            "single valid LaTeX expression that reproduces it exactly. Output "
            "ONLY the LaTeX code (no $$ delimiters, no markdown fences, no "
            "explanation)."
        )

    @staticmethod
    def for_figure() -> str:
        """Reserved for future figure description/captioning.

        Not called by the current pipeline — figures are detected (Stage I)
        but not OCR-processed (Stage II) by design.
        """
        return (
            "This image is a cropped figure (photo, chart, diagram or logo) "
            "from a document. Provide a concise one-sentence description of "
            "what it shows. If it contains a short embedded label or title, "
            "include that text. Output ONLY the description."
        )

    @staticmethod
    def for_page_element(element_type: str) -> str:
        """Prompt for page furniture: header, footer, or page number."""
        return (
            "This image is a cropped "
            f"'{element_type}' region from the margin of a document page. "
            "Transcribe its text exactly as it appears, preserving the "
            "original language and characters. Output ONLY the text."
        )

    @classmethod
    def get_prompt(cls, element_type: str) -> str:
        """Route an element ``type`` to the appropriate Stage II prompt.

        Args:
            element_type: One of :data:`ELEMENT_TYPES`.

        Returns:
            The user prompt string best suited for that element type.
        """
        if element_type in cls.TABLE_TYPES:
            return cls.for_table(element_type)
        if element_type == "math_formula":
            return cls.for_math()
        if element_type == "figure":
            return cls.for_figure()
        if element_type in cls.PAGE_TYPES:
            return cls.for_page_element(element_type)
        # Default: treat as text (covers TEXT_TYPES and any unknown text-like).
        return cls.for_text(element_type)


# --------------------------------------------------------------------------- #
# JSON repair
# --------------------------------------------------------------------------- #

JSON_REPAIR_PROMPT: str = (
    "The following text was supposed to be a single valid JSON object but it is "
    "malformed. Fix it and return ONLY the corrected, valid JSON object with "
    "the exact same intended content. Do not add explanations, comments, or "
    "markdown code fences.\n\n"
    "Malformed JSON:\n"
    "{malformed}"
)
