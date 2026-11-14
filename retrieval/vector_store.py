"""Qdrant vector database client and collection management."""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from config import get_settings

DEFAULT_COLLECTION_NAME = "research_documents"


def create_qdrant_client(*, in_memory: bool = False) -> QdrantClient:
    """Create a local in-memory or Qdrant Cloud client."""

    if in_memory:
        return QdrantClient(":memory:")

    settings = get_settings()
    url, api_key = settings.require_qdrant_credentials()

    return QdrantClient(
        url=url,
        api_key=api_key,
        timeout=settings.request_timeout_seconds,
    )


def ensure_collection(
    client: QdrantClient,
    *,
    vector_size: int,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> None:
    """Create a collection or validate its existing vector dimensions."""

    if vector_size <= 0:
        raise ValueError("vector_size must be greater than zero.")

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )
        return

    collection = client.get_collection(collection_name)
    existing_vectors = collection.config.params.vectors

    if not hasattr(existing_vectors, "size"):
        raise ValueError("Collection must use a single unnamed vector configuration.")

    if existing_vectors.size != vector_size:
        raise ValueError(
            f"Collection vector dimension mismatch: "
            f"expected {vector_size}, found {existing_vectors.size}."
        )


def upsert_vectors(
    client: QdrantClient,
    *,
    ids: list[int | str],
    vectors: list[list[float]],
    payloads: list[dict],
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> None:
    """Store vectors and their associated document metadata."""

    if not (len(ids) == len(vectors) == len(payloads)):
        raise ValueError("ids, vectors, and payloads must have equal lengths.")

    if not ids:
        return

    if any(not vector for vector in vectors):
        raise ValueError("Vectors must contain non-empty values.")

    vector_sizes = {len(vector) for vector in vectors}

    if len(vector_sizes) != 1:
        raise ValueError("All vectors must have the same dimensions.")

    points = [
        PointStruct(
            id=point_id,
            vector=vector,
            payload=payload,
        )
        for point_id, vector, payload in zip(ids, vectors, payloads, strict=True)
    ]

    client.upsert(
        collection_name=collection_name,
        points=points,
        wait=True,
    )


def search_vectors(
    client: QdrantClient,
    *,
    query_vector: list[float],
    limit: int = 5,
    collection_name: str = DEFAULT_COLLECTION_NAME,
):
    """Find documents with vectors most similar to the query."""

    if limit <= 0:
        raise ValueError("limit must be greater than zero.")

    if not query_vector:
        raise ValueError("Query vector must contain non-empty values.")

    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=limit,
        with_payload=True,
    )

    return response.points
