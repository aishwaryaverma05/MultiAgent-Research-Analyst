import pytest

from retrieval.semantic_search import index_documents, search_documents
from retrieval.vector_store import create_qdrant_client


class FakeEmbeddingModel:
    def embed_documents(self, texts):
        vectors = []

        for text in texts:
            if "Artificial intelligence" in text:
                vectors.append([1.0, 0.0, 0.0])
            else:
                vectors.append([0.0, 1.0, 0.0])

        return vectors

    def embed_query(self, query):
        return [1.0, 0.0, 0.0]


def test_index_and_search_documents():
    client = create_qdrant_client(in_memory=True)
    model = FakeEmbeddingModel()

    documents = [
        {
            "text": "Artificial intelligence research",
            "source_id": "source-1",
            "url": "https://example.com/ai",
        },
        {
            "text": "Machine learning research",
            "source_id": "source-2",
            "url": "https://example.com/ml",
        },
    ]

    ids = index_documents(
        client,
        documents,
        embedding_model=model,
    )

    results = search_documents(
        client,
        "Tell me about artificial intelligence",
        embedding_model=model,
        limit=2,
    )

    assert len(ids) == 2
    assert len(results) == 2
    assert results[0].payload["source_id"] == "source-1"
    assert results[0].payload["text"] == documents[0]["text"]
    assert results[0].payload["url"] == documents[0]["url"]

    client.close()


def test_index_documents_returns_empty_list_for_no_documents():
    client = create_qdrant_client(in_memory=True)

    result = index_documents(
        client,
        [],
        embedding_model=FakeEmbeddingModel(),
    )

    assert result == []

    client.close()


def test_index_documents_rejects_empty_text():
    client = create_qdrant_client(in_memory=True)

    with pytest.raises(ValueError, match="non-empty text"):
        index_documents(
            client,
            [{"text": "   "}],
            embedding_model=FakeEmbeddingModel(),
        )

    client.close()


def test_search_documents_rejects_empty_query():
    client = create_qdrant_client(in_memory=True)

    with pytest.raises(ValueError, match="Search query"):
        search_documents(
            client,
            "   ",
            embedding_model=FakeEmbeddingModel(),
        )

    client.close()


def test_index_documents_rejects_empty_embeddings():
    class EmptyEmbeddingModel:
        def embed_documents(self, texts):
            return []

    client = create_qdrant_client(in_memory=True)

    with pytest.raises(ValueError, match="no vectors"):
        index_documents(
            client,
            [{"text": "Artificial intelligence research"}],
            embedding_model=EmptyEmbeddingModel(),
        )

    client.close()


@pytest.mark.parametrize("limit", [0, -1])
def test_search_documents_rejects_invalid_limit(limit):
    client = create_qdrant_client(in_memory=True)

    with pytest.raises(ValueError, match="limit"):
        search_documents(
            client,
            "Artificial intelligence",
            embedding_model=FakeEmbeddingModel(),
            limit=limit,
        )

    client.close()


def test_search_documents_rejects_non_string_query():
    client = create_qdrant_client(in_memory=True)

    with pytest.raises(ValueError, match="non-empty string"):
        search_documents(
            client,
            None,
            embedding_model=FakeEmbeddingModel(),
        )

    client.close()
