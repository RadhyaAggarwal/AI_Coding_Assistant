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


def _mock_bad_status_response(status_code):
    response = MagicMock()
    response.raise_for_status.side_effect = requests.exceptions.HTTPError(
        f"{status_code} error", response=MagicMock(status_code=status_code)
    )
    return response


@patch("model_interface.ollama_adapter.requests.get")
def test_raises_model_unavailable_when_health_check_returns_a_bad_status(mock_get):
    """A broken tunnel/proxy hop can connect fine but return a 404/502/503
    body -- that used to pass the health check silently (no exception was
    raised for a non-2xx GET without calling raise_for_status), only
    surfacing later as a raw traceback from the real request."""
    mock_get.return_value = _mock_bad_status_response(404)
    adapter = OllamaAdapter("http://localhost:11434", "qwen2.5-coder:7b")

    with patch("model_interface.ollama_adapter.requests.post") as mock_post:
        with pytest.raises(ModelUnavailableError):
            adapter.generate([Message(role="user", content="hi")])
        mock_post.assert_not_called()


@patch("model_interface.ollama_adapter.requests.get")
@patch("model_interface.ollama_adapter.requests.post")
def test_raises_model_unavailable_not_a_raw_http_error_when_real_request_returns_bad_status(
    mock_post, mock_get
):
    """Live-observed: an unhandled HTTPError (a 404, then later a 503,
    from a flaky ngrok tunnel) crashed main.py with a raw traceback
    instead of the clean ModelUnavailableError message every other
    availability failure already produces."""
    mock_get.return_value = MagicMock()
    mock_post.return_value = _mock_bad_status_response(503)
    adapter = OllamaAdapter("http://localhost:11434", "qwen2.5-coder:7b")

    with pytest.raises(ModelUnavailableError):
        adapter.generate([Message(role="user", content="hi")])


def test_embed_without_a_configured_embedding_model_raises_immediately():
    """No embedding_model_name means semantic search is unavailable --
    must fail clearly and immediately, not attempt a request with a
    missing/wrong model name and produce a confusing downstream error."""
    adapter = OllamaAdapter("http://localhost:11434", "qwen2.5-coder:7b")

    with pytest.raises(ModelUnavailableError, match="No embedding model configured"):
        adapter.embed("some text")


@patch("model_interface.ollama_adapter.requests.get")
@patch("model_interface.ollama_adapter.requests.post")
def test_embed_sends_the_embedding_model_name_and_returns_the_real_vector(mock_post, mock_get):
    mock_get.return_value = MagicMock()
    mock_post.return_value = _mock_response({"embedding": [0.1, 0.2, 0.3]})
    adapter = OllamaAdapter(
        "http://localhost:11434",
        "qwen2.5-coder:7b",
        embedding_model_name="nomic-embed-text",
    )

    result = adapter.embed("def foo(): pass")

    assert result == [0.1, 0.2, 0.3]
    payload = mock_post.call_args.kwargs["json"]
    assert payload == {"model": "nomic-embed-text", "prompt": "def foo(): pass"}
    assert mock_post.call_args.args[0] == "http://localhost:11434/api/embeddings"


@patch("model_interface.ollama_adapter.requests.get")
@patch("model_interface.ollama_adapter.requests.post")
def test_embed_raises_model_unavailable_on_request_failure(mock_post, mock_get):
    mock_get.return_value = MagicMock()
    mock_post.side_effect = requests.exceptions.Timeout()
    adapter = OllamaAdapter(
        "http://localhost:11434", "qwen2.5-coder:7b", embedding_model_name="nomic-embed-text"
    )

    with pytest.raises(ModelUnavailableError):
        adapter.embed("some text")


@patch("model_interface.ollama_adapter.requests.get")
@patch("model_interface.ollama_adapter.requests.post")
def test_embed_raises_a_clear_error_when_the_response_has_no_embedding_list(mock_post, mock_get):
    """Reproduces the realistic misconfiguration case: embedding_name
    points at a real, working model that just isn't an embedding model
    (e.g. the chat model itself) -- the response won't have the expected
    shape, and this must fail clearly rather than return something
    silently wrong (like an empty vector) to the similarity search."""
    mock_get.return_value = MagicMock()
    mock_post.return_value = _mock_response({"message": {"content": "not an embedding"}})
    adapter = OllamaAdapter(
        "http://localhost:11434", "qwen2.5-coder:7b", embedding_model_name="qwen2.5-coder:7b"
    )

    with pytest.raises(ModelUnavailableError, match="expected 'embedding' list"):
        adapter.embed("some text")
