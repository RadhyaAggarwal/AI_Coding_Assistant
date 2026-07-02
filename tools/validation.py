"""Minimal JSON-schema-style validation for tool call arguments.

Deliberately a small subset (required, type, properties) rather than a
full jsonschema dependency — our tool schemas are simple, and this only
needs to catch the failure modes local models actually produce (missing
required arguments, wrong types), per the Build Plan Addendum's call for
schema validation around every tool call.
"""
from typing import Any


class ToolCallValidationError(Exception):
    """Raised when a tool call's arguments don't match the tool's schema."""


_TYPE_MAP: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "object": dict,
    "array": list,
}


def validate_arguments(schema: dict[str, Any], arguments: dict[str, Any]) -> None:
    """Raise ToolCallValidationError if arguments don't satisfy schema."""
    if not isinstance(arguments, dict):
        raise ToolCallValidationError(
            f"Arguments must be an object, got {type(arguments).__name__}"
        )

    for required_key in schema.get("required", []):
        if required_key not in arguments:
            raise ToolCallValidationError(f"Missing required argument '{required_key}'")

    properties = schema.get("properties", {})
    for key, value in arguments.items():
        prop_schema = properties.get(key)
        if prop_schema is None:
            continue
        expected_type = _TYPE_MAP.get(prop_schema.get("type"))
        if expected_type is not None and not isinstance(value, expected_type):
            raise ToolCallValidationError(
                f"Argument '{key}' should be of type '{prop_schema['type']}', "
                f"got {type(value).__name__}"
            )
