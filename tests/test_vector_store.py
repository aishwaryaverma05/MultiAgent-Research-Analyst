import pytest

from retrieval.vector_store import (
    DEFAULT_COLLECTION_NAME,
    create_qdrant_client,
    ensure_collection,
    search_vectors,
    upsert_vectors,
)


def test_create_in_memory_qdrant_client():
    client = create_qdrant_client(in_memory=True)

    assert client is not None

    client.close()


def test_ensure_collection_creates_collection():
    client = create_qdrant_client(in_memory=True)

    ensure_collection(client, vector_size=3072)

    assert client.collection_exists(DEFAULT_COLLECTION_NAME)

    client.close()


def test_ensure_collection_does_not_duplicate_collection():
    client = create_qdrant_client(in_memory=True)

    ensure_collection(client, vector_size=3072)
    ensure_collection(client, vector_size=3072)

    assert client.collection_exists(DEFAULT_COLLECTION_NAME)

    client.close()


def test_ensure_collection_rejects_invalid_vector_size():
    client = create_qdrant_client(in_memory=True)

    with pytest.raises(ValueError, match="vector_size"):
        ensure_collection(client, vector_size=0)

    client.close()




def test_upsert_and_search_vectors():
    client = create_qdrant_client(in_memory=True)

    ensure_collection(client, vector_size=3)

    upsert_vectors(
        client,
        ids=[1, 2],
        vectors=[
            [0.9, 0.1, 0.0],
            [0.0, 0.1, 0.9],
        ],
        payloads=[
            {"text": "Artificial intelligence research", "source": "source-1"},
            {"text": "Climate change research", "source": "source-2"},
        ],
    )

    results = search_vectors(
        client,
        query_vector=[0.9, 0.1, 0.0],
        limit=2,
    )

    assert len(results) == 2
    assert results[0].payload["source"] == "source-1"
    assert results[0].score > 0

    client.close()


def test_upsert_rejects_mismatched_lengths():
    client = create_qdrant_client(in_memory=True)

    ensure_collection(client, vector_size=3)

    with pytest.raises(ValueError, match="equal lengths"):
        upsert_vectors(
            client,
            ids=[1],
            vectors=[[0.1, 0.2, 0.3]],
            payloads=[],
        )

    client.close()


def test_search_rejects_invalid_limit():
    client = create_qdrant_client(in_memory=True)

    ensure_collection(client, vector_size=3)

    with pytest.raises(ValueError, match="limit"):
        search_vectors(
            client,
            query_vector=[0.1, 0.2, 0.3],
            limit=0,
        )

    client.close()


def test_ensure_collection_rejects_dimension_mismatch():
    client = create_qdrant_client(in_memory=True)

    ensure_collection(client, vector_size=3)

    with pytest.raises(ValueError, match="dimension"):
        ensure_collection(client, vector_size=4)

    client.close()


def test_upsert_rejects_empty_vectors():
    client = create_qdrant_client(in_memory=True)

    ensure_collection(client, vector_size=3)

    with pytest.raises(ValueError, match="non-empty"):
        upsert_vectors(
            client,
            ids=[1],
            vectors=[[]],
            payloads=[{"text": "Test document"}],
        )

    client.close()


def test_upsert_rejects_inconsistent_vector_dimensions():
    client = create_qdrant_client(in_memory=True)

    ensure_collection(client, vector_size=3)

    with pytest.raises(ValueError, match="same dimensions"):
        upsert_vectors(
            client,
            ids=[1, 2],
            vectors=[
                [0.1, 0.2, 0.3],
                [0.4, 0.5],
            ],
            payloads=[
                {"text": "Document one"},
                {"text": "Document two"},
            ],
        )

    client.close()


def test_search_rejects_empty_query_vector():
    client = create_qdrant_client(in_memory=True)

    ensure_collection(client, vector_size=3)

    with pytest.raises(ValueError, match="non-empty"):
        search_vectors(
            client,
            query_vector=[],
        )

    client.close()
