from __future__ import annotations

import logging
from uuid import UUID

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models

logger = logging.getLogger(__name__)


class VectorIndexRepository:
    """
    Async repository boundary for Qdrant vector index operations.

    This repository owns:
    - collection creation
    - payload index creation
    - document-scoped vector deletion
    - point upsert/count

    Retrieval ranking logic stays in services/rag/retrieval.py.
    """

    def __init__(self, client: AsyncQdrantClient, collection_name: str):
        self._client = client
        self._collection_name = collection_name

    @property
    def collection_name(self) -> str:
        return self._collection_name

    @staticmethod
    def document_id_filter(document_id: UUID | str) -> models.Filter:
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=str(document_id)),
                )
            ]
        )

    @staticmethod
    def allowed_documents_filter(document_ids: list[UUID | str]) -> models.Filter:
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchAny(
                        any=[str(document_id) for document_id in document_ids]
                    ),
                )
            ]
        )

    async def ensure_collection(self, dense_vector_size: int) -> None:
        """
        Create collection if it does not exist.
        - dense
        - bm25
        """
        exists = await self._client.collection_exists(self._collection_name)
        if exists:
            return

        await self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=dense_vector_size,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "bm25": models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                )
            },
        )

    async def ensure_payload_indexes(self) -> None:
        """
        Ensure payload indexes used for access control and retrieval filtering.
        Qdrant may raise when an index already exists, so we log and continue.
        """
        index_fields = {
            # Security / access control
            "document_id": models.PayloadSchemaType.KEYWORD,
            "workspace_id": models.PayloadSchemaType.KEYWORD,

            # Optional user/document narrowing
            "filename": models.PayloadSchemaType.KEYWORD,

            # Retrieval filters
            "source_type": models.PayloadSchemaType.KEYWORD,
            "is_table_stub": models.PayloadSchemaType.BOOL,
            "page_number": models.PayloadSchemaType.INTEGER,
            "statement_type": models.PayloadSchemaType.KEYWORD,
            "table_name": models.PayloadSchemaType.KEYWORD,

            # Ordering/debug/versioning
            "chunk_index": models.PayloadSchemaType.INTEGER,
            "content_hash": models.PayloadSchemaType.KEYWORD,
            "ingestion_version": models.PayloadSchemaType.INTEGER,
        }

        for field_name, field_schema in index_fields.items():
            try:
                await self._client.create_payload_index(
                    collection_name=self._collection_name,
                    field_name=field_name,
                    field_schema=field_schema,
                )
            except Exception as exc:
                logger.debug(
                    "Qdrant payload index skipped/failed for %s: %s",
                    field_name,
                    exc,
                )

    async def upsert_points(self, points: list[models.PointStruct]) -> None:
        if not points:
            return

        await self._client.upsert(
            collection_name=self._collection_name,
            points=points,
        )

    async def delete_points_by_document_id(self, document_id: UUID | str) -> None:
        """
        Delete all Qdrant points for one document.

        Used before reindexing/replacing a document version.
        """
        await self._client.delete(
            collection_name=self._collection_name,
            points_selector=models.FilterSelector(
                filter=self.document_id_filter(document_id),
            ),
        )

    async def count_points_by_document_id(self, document_id: UUID | str) -> int:
        result = await self._client.count(
            collection_name=self._collection_name,
            count_filter=self.document_id_filter(document_id),
            exact=True,
        )
        return int(result.count)