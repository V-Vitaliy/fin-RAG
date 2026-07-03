import uuid

import pytest
from qdrant_client.http import models

from app.repositories.qdrant_repo import VectorIndexRepository


class FakeAsyncQdrantClient:
    def __init__(self):
        self.deleted = None
        self.counted = None
        self.upserted = None
        self.queried = None
        self.collection_exists_called = None
        self.created_collection = None
        self.scrolled = None
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

    async def query_points(self, **kwargs):
        self.queried = kwargs

        class QueryResult:
            points = []

        return QueryResult()

    async def scroll(self, **kwargs):
        self.scrolled = kwargs
        return [], None


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

@pytest.mark.asyncio
async def test_hybrid_search_skips_empty_allowed_documents():
    client = FakeAsyncQdrantClient()
    repo = VectorIndexRepository(client=client, collection_name="finance_documents")

    result = await repo.hybrid_search(
        dense_vector=[0.1, 0.2],
        sparse_indices=[1, 2],
        sparse_values=[0.5, 0.8],
        allowed_document_ids=[],
        limit=5,
        prefetch_limit=10,
    )

    assert result == []
    assert client.queried is None


@pytest.mark.asyncio
async def test_hybrid_search_filters_by_allowed_document_ids():
    client = FakeAsyncQdrantClient()
    repo = VectorIndexRepository(client=client, collection_name="finance_documents")

    await repo.hybrid_search(
        dense_vector=[0.1, 0.2],
        sparse_indices=[1, 2],
        sparse_values=[0.5, 0.8],
        allowed_document_ids=["doc-1", "doc-2"],
        limit=5,
        prefetch_limit=10,
    )

    assert client.queried["collection_name"] == "finance_documents"
    assert client.queried["limit"] == 5
    assert client.queried["with_payload"] is True
    assert client.queried["with_vectors"] is False

    qdrant_filter = client.queried["query_filter"]
    condition = qdrant_filter.must[0]

    assert condition.key == "document_id"
    assert condition.match.any == ["doc-1", "doc-2"]

    prefetch = client.queried["prefetch"]

    assert len(prefetch) == 2
    assert prefetch[0].using == "dense"
    assert prefetch[0].limit == 10
    assert prefetch[1].using == "bm25"
    assert prefetch[1].limit == 10


@pytest.mark.asyncio
async def test_hybrid_search_filters_table_stubs_when_requested():
    client = FakeAsyncQdrantClient()
    repo = VectorIndexRepository(client=client, collection_name="finance_documents")

    await repo.hybrid_search(
        dense_vector=[0.1, 0.2],
        sparse_indices=[1, 2],
        sparse_values=[0.5, 0.8],
        allowed_document_ids=["doc-1"],
        limit=5,
        prefetch_limit=10,
        only_table_stubs=True,
    )

    qdrant_filter = client.queried["query_filter"]
    conditions_by_key = {condition.key: condition for condition in qdrant_filter.must}

    assert conditions_by_key["document_id"].match.any == ["doc-1"]
    assert conditions_by_key["is_table_stub"].match.value is True
    assert conditions_by_key["source_type"].match.value == "table_stub"

@pytest.mark.asyncio
async def test_scroll_text_neighbors_filters_by_document_id_and_local_chunk_index():
    client = FakeAsyncQdrantClient()
    repo = VectorIndexRepository(client=client, collection_name="finance_documents")

    result = await repo.scroll_text_neighbors(
        document_id="doc-1",
        center_index=5,
        window=1,
        limit=3,
    )

    assert result == []
    assert client.scrolled["collection_name"] == "finance_documents"
    assert client.scrolled["limit"] == 3
    assert client.scrolled["with_payload"] is True
    assert client.scrolled["with_vectors"] is False

    qdrant_filter = client.scrolled["scroll_filter"]
    conditions_by_key = {condition.key: condition for condition in qdrant_filter.must}

    assert conditions_by_key["document_id"].match.value == "doc-1"
    assert conditions_by_key["is_table_stub"].match.value is False
    assert conditions_by_key["local_chunk_index"].range.gte == 4
    assert conditions_by_key["local_chunk_index"].range.lte == 6


@pytest.mark.asyncio
async def test_hybrid_search_does_not_filter_by_doc_name():
    client = FakeAsyncQdrantClient()
    repo = VectorIndexRepository(
        client=client,
        collection_name="finance_documents",
    )

    result = await repo.hybrid_search(
        dense_vector=[0.1, 0.2, 0.3],
        sparse_indices=[1, 2, 3],
        sparse_values=[0.3, 0.2, 0.1],
        allowed_document_ids=["doc-1", "doc-2"],
        limit=5,
        prefetch_limit=10,
    )

    assert result == []

    qdrant_filter = client.queried["query_filter"]
    must_keys = [condition.key for condition in qdrant_filter.must]

    assert "document_id" in must_keys
    assert "doc_name" not in must_keys

    document_condition = next(
        condition for condition in qdrant_filter.must
        if condition.key == "document_id"
    )

    assert document_condition.match.any == ["doc-1", "doc-2"]