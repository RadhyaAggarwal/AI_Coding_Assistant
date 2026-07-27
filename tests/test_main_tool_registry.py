from main import build_tool_registry
from model_interface.base import ModelInterface, ModelResponse
from state.snapshot import SnapshotManager


class FakeModel(ModelInterface):
    def generate(self, messages, tools=None):
        return ModelResponse(text="")

    def embed(self, text):
        return [0.0]


def _registered_names(registry) -> set[str]:
    return {schema["function"]["name"] for schema in registry.schemas()}


def test_semantic_search_is_not_registered_when_embedding_not_configured(tmp_path):
    snapshots = SnapshotManager(tmp_path / ".agent_state")
    registry = build_tool_registry(
        str(tmp_path), 60, snapshots, tmp_path / ".agent_state",
        model=FakeModel(), embedding_configured=False,
    )
    assert "semantic_search" not in _registered_names(registry)


def test_semantic_search_is_not_registered_without_a_model_even_if_flagged_configured(tmp_path):
    """Defensive: embedding_configured=True with no model at all must not
    crash or register a tool with nothing to call -- both signals are
    required, not either alone."""
    snapshots = SnapshotManager(tmp_path / ".agent_state")
    registry = build_tool_registry(
        str(tmp_path), 60, snapshots, tmp_path / ".agent_state",
        model=None, embedding_configured=True,
    )
    assert "semantic_search" not in _registered_names(registry)


def test_semantic_search_is_registered_when_embedding_is_configured(tmp_path):
    snapshots = SnapshotManager(tmp_path / ".agent_state")
    registry = build_tool_registry(
        str(tmp_path), 60, snapshots, tmp_path / ".agent_state",
        model=FakeModel(), embedding_configured=True,
    )
    assert "semantic_search" in _registered_names(registry)


def test_every_other_tool_is_still_registered_regardless_of_embedding_config(tmp_path):
    snapshots = SnapshotManager(tmp_path / ".agent_state")
    registry = build_tool_registry(
        str(tmp_path), 60, snapshots, tmp_path / ".agent_state",
    )
    names = _registered_names(registry)
    assert "read_file" in names
    assert "edit_file" in names
    assert len(names) == 12  # the original 12, unaffected by the new optional tool
