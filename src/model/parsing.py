"""Shared parsing helpers for cleaning up raw model output.

Both the layout detector and the content recognizer receive free-form text from
the model that may be wrapped in markdown code fences or surrounded by stray
prose. These helpers centralize that cleanup so the logic is written once.
"""

from __future__ import annotations

import re

# Matches a single ```lang ... ``` fenced block and captures its inner content.
_CODE_FENCE_RE = re.compile(r"```(?:\w+)?\s*(.*?)```", re.DOTALL)


def strip_code_fence(text: str) -> str:
    """Return the contents of a markdown code fence, if the text has one.

    Args:
        text: Raw text that may contain a ```...``` fenced block.

    Returns:
        The text inside the first code fence, or the original text unchanged
        when no fence is present. Returns an empty string for empty input.
    """
    if not text:
        return ""
    match = _CODE_FENCE_RE.search(text)
    return match.group(1) if match else text


def extract_json_object(text: str) -> str:
    """Extract the outermost ``{...}`` JSON object from noisy model output.

    Strips any surrounding code fence first, then returns the span from the
    first ``{`` to the last ``}``.

    Args:
        text: Raw model output that should contain a JSON object.

    Returns:
        The JSON object substring, or an empty string when none is found.
    """
    if not text:
        return ""

    cleaned = strip_code_fence(text).strip()
    if cleaned.startswith("{") and cleaned.endswith("}"):
        return cleaned

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        return cleaned[start : end + 1]
    return ""
