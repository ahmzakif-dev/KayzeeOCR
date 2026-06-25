"""Output validation against the KayzeeOCR JSON Schema.

Loads ``schemas/output_schema.json`` and validates assembled page output dicts
against it before they are saved or returned to the caller.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import jsonschema
from jsonschema import Draft7Validator

logger = logging.getLogger(__name__)

# Default schema location: <repo_root>/schemas/output_schema.json
_DEFAULT_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2] / "schemas" / "output_schema.json"
)


class ValidationError(Exception):
    """Raised when an output dict fails JSON Schema validation.

    Attributes:
        errors: Human-readable validation error messages.
    """

    def __init__(self, errors: list[str]) -> None:
        self.errors: list[str] = errors
        message = "Output validation failed:\n" + "\n".join(
            f"  - {e}" for e in errors
        )
        super().__init__(message)


def load_schema(schema_path: str | Path | None = None) -> dict[str, Any]:
    """Load the output JSON Schema from disk.

    Args:
        schema_path: Optional path to a schema file. Defaults to the bundled
            ``schemas/output_schema.json``.

    Returns:
        The parsed schema as a dict.
    """
    path = Path(schema_path) if schema_path else _DEFAULT_SCHEMA_PATH
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


class OutputValidator:
    """Validate page-output dicts against the KayzeeOCR JSON Schema."""

    def __init__(self, schema_path: str | Path | None = None) -> None:
        """Initialize the validator and compile the schema.

        Args:
            schema_path: Optional override for the schema file location.
        """
        self.schema: dict[str, Any] = load_schema(schema_path)
        # Surface schema authoring errors early.
        Draft7Validator.check_schema(self.schema)
        self._validator = Draft7Validator(self.schema)

    def validate(self, data: dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate ``data`` against the schema.

        Args:
            data: The assembled page-output dict.

        Returns:
            A tuple ``(is_valid, errors)`` where ``errors`` is empty when valid.
        """
        errors = sorted(self._validator.iter_errors(data), key=lambda e: e.path)
        if not errors:
            return True, []
        messages = [self._format_error(e) for e in errors]
        return False, messages

    def validate_and_raise(self, data: dict[str, Any]) -> None:
        """Validate ``data`` and raise :class:`ValidationError` if invalid.

        Args:
            data: The assembled page-output dict.

        Raises:
            ValidationError: If validation fails.
        """
        is_valid, errors = self.validate(data)
        if not is_valid:
            logger.warning("Output failed schema validation: %d error(s)", len(errors))
            raise ValidationError(errors)

    @staticmethod
    def _format_error(error: jsonschema.exceptions.ValidationError) -> str:
        """Render a jsonschema error into a compact, human-readable string."""
        location = "/".join(str(p) for p in error.absolute_path) or "<root>"
        return f"{location}: {error.message}"


# Module-level singleton reused by the convenience function.
_default_validator: OutputValidator | None = None


def _get_default_validator() -> OutputValidator:
    global _default_validator
    if _default_validator is None:
        _default_validator = OutputValidator()
    return _default_validator


def validate_output(data: dict[str, Any]) -> bool:
    """Return ``True`` if ``data`` is valid against the default schema.

    Convenience wrapper around :class:`OutputValidator` using a cached instance.

    Args:
        data: The assembled page-output dict.

    Returns:
        ``True`` when valid, ``False`` otherwise. Errors are logged at WARNING.
    """
    is_valid, errors = _get_default_validator().validate(data)
    if not is_valid:
        for err in errors:
            logger.warning("Validation error: %s", err)
    return is_valid
