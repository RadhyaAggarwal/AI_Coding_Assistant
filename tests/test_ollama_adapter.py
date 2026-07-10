import pytest
import requests
from unittest.mock import MagicMock, patch

from model_interface.base import Message, ModelUnavailableError
from model_interface.ollama_adapter import OllamaAdapter


def _mock_response(json_body):
    response = MagicMock()
    response.json.return_value = json_body
    response.raise_for_status.return_value = None
    return response


@patch("model_interface.ollama_adapter.requests.get")
@patch("model_interface.ollama_adapter.requests.post")
def test_omits_options_when_temperature_not_set(mock_post, mock_get):
    mock_post.return_value = _mock_response({"message": {"content": "ok"}})
    adapter = OllamaAdapter("http://localhost:11434", "qwen2.5-coder:7b")

    adapter.generate([Message(role="user", content="hi")])

    payload = mock_post.call_args.kwargs["json"]
    assert "options" not in payload


@patch("model_interface.ollama_adapter.requests.get")
@patch("model_interface.ollama_adapter.requests.post")
def test_includes_temperature_in_options_when_set(mock_post, mock_get):
    mock_post.return_value = _mock_response({"message": {"content": "ok"}})
    adapter = OllamaAdapter("http://localhost:11434", "qwen2.5-coder:7b", temperature=0.2)

    adapter.generate([Message(role="user", content="hi")])

    payload = mock_post.call_args.kwargs["json"]
    assert payload["options"] == {"temperature": 0.2}


@patch("model_interface.ollama_adapter.requests.get")
def test_health_check_runs_before_the_real_request(mock_get):
    mock_get.return_value = MagicMock()
    adapter = OllamaAdapter("http://localhost:11434", "qwen2.5-coder:7b")

    with patch("model_interface.ollama_adapter.requests.post") as mock_post:
        mock_post.return_value = _mock_response({"message": {"content": "ok"}})
        adapter.generate([Message(role="user", content="hi")])

    mock_get.assert_called_once()
    assert mock_get.call_args.args[0] == "http://localhost:11434/api/tags"


@patch("model_interface.ollama_adapter.requests.get")
def test_raises_model_unavailable_when_health_check_times_out(mock_get):
    mock_get.side_effect = requests.exceptions.Timeout()
    adapter = OllamaAdapter("http://localhost:11434", "qwen2.5-coder:7b")

    with patch("model_interface.ollama_adapter.requests.post") as mock_post:
        with pytest.raises(ModelUnavailableError):
            adapter.generate([Message(role="user", content="hi")])
        mock_post.assert_not_called()


@patch("model_interface.ollama_adapter.requests.get")
@patch("model_interface.ollama_adapter.requests.post")
def test_raises_model_unavailable_when_real_request_times_out_after_healthy_check(
    mock_post, mock_get
):
    mock_get.return_value = MagicMock()
    mock_post.side_effect = requests.exceptions.Timeout()
    adapter = OllamaAdapter("http://localhost:11434", "qwen2.5-coder:7b")

    with pytest.raises(ModelUnavailableError):
        adapter.generate([Message(role="user", content="hi")])
