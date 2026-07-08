from unittest.mock import MagicMock, patch

from model_interface.base import Message
from model_interface.ollama_adapter import OllamaAdapter


def _mock_response(json_body):
    response = MagicMock()
    response.json.return_value = json_body
    response.raise_for_status.return_value = None
    return response


@patch("model_interface.ollama_adapter.requests.post")
def test_omits_options_when_temperature_not_set(mock_post):
    mock_post.return_value = _mock_response({"message": {"content": "ok"}})
    adapter = OllamaAdapter("http://localhost:11434", "qwen2.5-coder:7b")

    adapter.generate([Message(role="user", content="hi")])

    payload = mock_post.call_args.kwargs["json"]
    assert "options" not in payload


@patch("model_interface.ollama_adapter.requests.post")
def test_includes_temperature_in_options_when_set(mock_post):
    mock_post.return_value = _mock_response({"message": {"content": "ok"}})
    adapter = OllamaAdapter("http://localhost:11434", "qwen2.5-coder:7b", temperature=0.2)

    adapter.generate([Message(role="user", content="hi")])

    payload = mock_post.call_args.kwargs["json"]
    assert payload["options"] == {"temperature": 0.2}
