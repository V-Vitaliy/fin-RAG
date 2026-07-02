import uuid

import pytest

from app.models.domain import Document, DocumentStatus
from app.repositories.pg_repo import DocumentRepository


class FakeScalarResult:
    def __init__(self, value=None, values=None):
        self._value = value
        self._values = values if values is not None else ([] if value is None else [value])

    def first(self):
        return self._value

    def all(self):
        return self._values


class FakeExecuteResult:
    def __init__(self, value=None, values=None):
        self._scalar_result = FakeScalarResult(value=value, values=values)

    def scalars(self):
        return self._scalar_result


class FakeAsyncSession:
    def __init__(self, execute_value=None, execute_values=None):
        self.added = []
        self.deleted = []
        self.flushed = 0
        self.refreshed = []
        self.execute_value = execute_value
        self.execute_values = execute_values
        self.executed_statements = []

    def add(self, obj):
        self.added.append(obj)

    async def flush(self):
        self.flushed += 1

    async def refresh(self, obj):
        self.refreshed.append(obj)

    async def delete(self, obj):
        self.deleted.append(obj)

    async def execute(self, stmt):
        self.executed_statements.append(stmt)
        return FakeExecuteResult(
            value=self.execute_value,
            values=self.execute_values,
        )


@pytest.mark.asyncio
async def test_create_document_sets_ingestion_metadata():
    session = FakeAsyncSession()
    repo = DocumentRepository(session)

    workspace_id = uuid.uuid4()
    document_id = uuid.uuid4()

    document = await repo.create_document(
        workspace_id=workspace_id,
        filename="doc.pdf",
        s3_object_key="workspaces/x/doc.pdf",
        document_id=document_id,
        content_hash="a" * 64,
        status=DocumentStatus.PROCESSING,
        ingestion_version=3,
    )

    assert document.id == document_id
    assert document.workspace_id == workspace_id
    assert document.filename == "doc.pdf"
    assert document.s3_object_key == "workspaces/x/doc.pdf"
    assert document.status == DocumentStatus.PROCESSING
    assert document.content_hash == "a" * 64
    assert document.is_active is True
    assert document.ingestion_version == 3

    assert session.added == [document]
    assert session.flushed == 1
    assert session.refreshed == [document]


@pytest.mark.asyncio
async def test_update_status_failed_sets_failure_reason():
    document = Document(
        id=uuid.uuid4(),
        workspace_id=uuid.uuid4(),
        filename="doc.pdf",
        s3_object_key="key",
        status=DocumentStatus.PROCESSING,
    )

    session = FakeAsyncSession(execute_value=document)
    repo = DocumentRepository(session)

    updated = await repo.update_status(
        document_id=document.id,
        status=DocumentStatus.FAILED,
        failure_reason="boom",
    )

    assert updated is document
    assert document.status == DocumentStatus.FAILED
    assert document.failure_reason == "boom"
    assert session.flushed == 1
    assert session.refreshed == [document]


@pytest.mark.asyncio
async def test_update_status_processing_clears_failure_reason():
    document = Document(
        id=uuid.uuid4(),
        workspace_id=uuid.uuid4(),
        filename="doc.pdf",
        s3_object_key="key",
        status=DocumentStatus.FAILED,
        failure_reason="old error",
    )

    session = FakeAsyncSession(execute_value=document)
    repo = DocumentRepository(session)

    updated = await repo.update_status(
        document_id=document.id,
        status=DocumentStatus.PROCESSING,
    )

    assert updated is document
    assert document.status == DocumentStatus.PROCESSING
    assert document.failure_reason is None


@pytest.mark.asyncio
async def test_mark_ready_sets_indexing_metadata():
    document = Document(
        id=uuid.uuid4(),
        workspace_id=uuid.uuid4(),
        filename="doc.pdf",
        s3_object_key="key",
        status=DocumentStatus.PROCESSING,
        failure_reason="old error",
    )

    session = FakeAsyncSession(execute_value=document)
    repo = DocumentRepository(session)

    updated = await repo.mark_ready(
        document_id=document.id,
        qdrant_points_count=12,
        duckdb_tables_count=3,
    )

    assert updated is document
    assert document.status == DocumentStatus.READY
    assert document.failure_reason is None
    assert document.indexed_at is not None
    assert document.qdrant_points_count == 12
    assert document.duckdb_tables_count == 3
    assert document.is_active is True


@pytest.mark.asyncio
async def test_mark_failed_sets_failure_reason():
    document = Document(
        id=uuid.uuid4(),
        workspace_id=uuid.uuid4(),
        filename="doc.pdf",
        s3_object_key="key",
        status=DocumentStatus.PROCESSING,
    )

    session = FakeAsyncSession(execute_value=document)
    repo = DocumentRepository(session)

    updated = await repo.mark_failed(
        document_id=document.id,
        failure_reason="ingestion failed",
    )

    assert updated is document
    assert document.status == DocumentStatus.FAILED
    assert document.failure_reason == "ingestion failed"


@pytest.mark.asyncio
async def test_mark_inactive_sets_is_active_false():
    document = Document(
        id=uuid.uuid4(),
        workspace_id=uuid.uuid4(),
        filename="doc.pdf",
        s3_object_key="key",
        status=DocumentStatus.READY,
        is_active=True,
    )

    session = FakeAsyncSession(execute_value=document)
    repo = DocumentRepository(session)

    updated = await repo.mark_inactive(document.id)

    assert updated is document
    assert document.is_active is False