from model_interface.base import ModelInterface, ModelResponse
from tools.semantic_search import SemanticSearchTool


class WordVectorModel(ModelInterface):
    VOCAB = ["email", "validate", "user", "delete", "note"]

    def generate(self, messages, tools=None):
        return ModelResponse(text="")

    def embed(self, text):
        lower = text.lower()
        return [float(lower.count(word)) for word in self.VOCAB]


def test_returns_a_real_snippet_with_file_and_line_reference(tmp_path):
    (tmp_path / "validators.py").write_text(
        "def validate_email(addr):\n    return '@' in addr\n", encoding="utf-8"
    )
    tool = SemanticSearchTool(tmp_path, WordVectorModel())

    result = tool.run(query="check if an email is valid")

    assert "validate_email" in result
    assert "validators.py:1-2" in result
    assert "return '@' in addr" in result


def test_returns_a_clear_message_when_nothing_is_indexed(tmp_path):
    tool = SemanticSearchTool(tmp_path, WordVectorModel())
    result = tool.run(query="anything")
    assert "No indexed code found" in result


def test_progress_message_names_the_query(tmp_path):
    tool = SemanticSearchTool(tmp_path, WordVectorModel())
    message = tool.progress_message({"query": "how do we validate email"})
    assert message == "Searching semantically for 'how do we validate email'..."


def test_reports_real_progress_when_embedding_new_chunks(tmp_path):
    (tmp_path / "a.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
    events = []
    tool = SemanticSearchTool(tmp_path, WordVectorModel(), report=events.append)

    tool.run(query="anything")

    assert any("Embedding" in e for e in events)


def test_defaults_to_print_so_progress_is_visible_without_extra_wiring(tmp_path, capsys):
    (tmp_path / "a.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
    tool = SemanticSearchTool(tmp_path, WordVectorModel())

    tool.run(query="anything")

    captured = capsys.readouterr()
    assert "Embedding" in captured.out
