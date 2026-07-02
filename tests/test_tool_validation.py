import pytest

from tools.validation import ToolCallValidationError, validate_arguments

SCHEMA = {
    "type": "object",
    "properties": {"path": {"type": "string"}},
    "required": ["path"],
}


def test_valid_arguments_pass():
    validate_arguments(SCHEMA, {"path": "config.yaml"})  # should not raise


def test_missing_required_argument_raises():
    with pytest.raises(ToolCallValidationError):
        validate_arguments(SCHEMA, {})


def test_wrong_type_raises():
    with pytest.raises(ToolCallValidationError):
        validate_arguments(SCHEMA, {"path": 123})


def test_non_dict_arguments_raises():
    with pytest.raises(ToolCallValidationError):
        validate_arguments(SCHEMA, "not-a-dict")
