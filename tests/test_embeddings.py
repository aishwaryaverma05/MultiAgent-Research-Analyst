from types import SimpleNamespace
from unittest.mock import Mock

from retrieval import embeddings


def test_create_embeddings_model(monkeypatch):
    fake_settings = SimpleNamespace(
        embedding_model="models/gemini-embedding-001",
        require_gemini_credentials=lambda: "fake-api-key",
    )

    fake_embeddings = object()
    mock_embedding_class = Mock(return_value=fake_embeddings)

    monkeypatch.setattr(
        embeddings,
        "get_settings",
        lambda: fake_settings,
    )
    monkeypatch.setattr(
        embeddings,
        "GoogleGenerativeAIEmbeddings",
        mock_embedding_class,
    )

    result = embeddings.create_embeddings_model()

    assert result is fake_embeddings
    mock_embedding_class.assert_called_once_with(
        model="models/gemini-embedding-001",
        api_key="fake-api-key",
    )
