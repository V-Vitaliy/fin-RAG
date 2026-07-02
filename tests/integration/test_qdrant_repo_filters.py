import uuid

import pytest
from qdrant_client.http import models

from app.repositories.qdrant_repo import VectorIndexRepository


class FakeAsyncQdrantClient:
    def __init__(self):
        self.deleted = None
        self.counted = None
        self.upserted = None
        self.collection_exists_called = None
        self.created_collection = None
        self.created_indexes = []

    async def collection_exists(self, collection_name):
        self.collection_exists_called = collection_name
        return False

    async def create_collection(self, **kwargs):
        self.created_collection = kwargs

    async def create_payload_index(self, **kwargs):
        self.created_indexes.append(kwargs)

    async def upsert(self, **kwargs):
        self.upserted = kwargs

    async def delete(self, **kwargs):
        self.deleted = kwargs

    async def count(self, **kwargs):
        self.counted = kwargs

        class CountResult:
            count = 7

        return CountResult()


def _extract_document_id_from_filter(qdrant_filter: models.Filter) -> str:
    condition = qdrant_filter.must[0]
    return condition.match.value


@pytest.mark.asyncio
async def test_delete_points_by_document_id_uses_document_id_filter():
    client = FakeAsyncQdrantClient()
    repo = VectorIndexRepository(client=client, collection_name="finance_documents")
    document_id = uuid.uuid4()

    await repo.delete_points_by_document_id(document_id)

    assert client.deleted["collection_name"] == "finance_documents"

    selector = client.deleted["points_selector"]
    assert isinstance(selector, models.FilterSelector)

    qdrant_filter = selector.filter
    assert qdrant_filter.must[0].key == "document_id"
    assert _extract_document_id_from_filter(qdrant_filter) == str(document_id)


@pytest.mark.asyncio
async def test_count_points_by_document_id_uses_document_id_filter():
    client = FakeAsyncQdrantClient()
    repo = VectorIndexRepository(client=client, collection_name="finance_documents")
    document_id = uuid.uuid4()

    count = await repo.count_points_by_document_id(document_id)

    assert count == 7
    assert client.counted["collection_name"] == "finance_documents"
    assert client.counted["exact"] is True

    qdrant_filter = client.counted["count_filter"]
    assert qdrant_filter.must[0].key == "document_id"
    assert _extract_document_id_from_filter(qdrant_filter) == str(document_id)


@pytest.mark.asyncio
async def test_ensure_collection_uses_dense_and_bm25_vectors():
    client = FakeAsyncQdrantClient()
    repo = VectorIndexRepository(client=client, collection_name="finance_documents")

    await repo.ensure_collection(dense_vector_size=3072)

    assert client.collection_exists_called == "finance_documents"
    assert client.created_collection["collection_name"] == "finance_documents"

    vectors_config = client.created_collection["vectors_config"]
    sparse_config = client.created_collection["sparse_vectors_config"]

    assert "dense" in vectors_config
    assert vectors_config["dense"].size == 3072
    assert "bm25" in sparse_config


@pytest.mark.asyncio
async def test_upsert_points_skips_empty_list():
    client = FakeAsyncQdrantClient()
    repo = VectorIndexRepository(client=client, collection_name="finance_documents")

    await repo.upsert_points([])

    assert client.upserted is None


def test_allowed_documents_filter_uses_match_any_document_ids():
    doc_a = uuid.uuid4()
    doc_b = uuid.uuid4()

    qdrant_filter = VectorIndexRepository.allowed_documents_filter([doc_a, doc_b])

    condition = qdrant_filter.must[0]

    assert condition.key == "document_id"
    assert condition.match.any == [str(doc_a), str(doc_b)]