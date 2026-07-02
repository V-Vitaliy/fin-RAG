import uuid
from io import BytesIO

import pytest

from app.models.domain import Document, DocumentStatus
from app.services.jobs.ingestion_queue import IngestDocumentJob, IngestionQueue
from app.use_cases.documents import DocumentUseCase


class FakeWorkspaceRepository:
    def __init__(self, workspace):
        self._workspace = workspace

    async def get_workspace(self, workspace_id):
        return self._workspace


class FakeDocumentRepository:
    def __init__(self):
        self.created = None
        self.status_updates = []

    async def create_document(self, **kwargs):
        document = Document(
            id=kwargs["document_id"],
            workspace_id=kwargs["workspace_id"],
            filename=kwargs["filename"],
            s3_object_key=kwargs["s3_object_key"],
            status=kwargs["status"],
            content_hash=kwargs["content_hash"],
            ingestion_version=kwargs["ingestion_version"],
            is_active=True,
        )
        self.created = document
        return document

    async def update_status(self, document_id, status, failure_reason=None):
        self.created.status = status
        self.created.failure_reason = failure_reason
        self.status_updates.append((document_id, status, failure_reason))
        return self.created

    async def get_document(self, document_id):
        return self.created

    async def mark_inactive(self, document_id):
        self.created.is_active = False
        return self.created


class FakeUoW:
    def __init__(self, workspace):
        self.workspaces = FakeWorkspaceRepository(workspace)
        self.documents = FakeDocumentRepository()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None


class FakeS3Repository:
    def __init__(self):
        self.uploaded = []
        self.deleted = []

    async def upload_file(self, file_obj, object_key):
        self.uploaded.append((file_obj.read(), object_key))
        return object_key

    async def delete_file(self, object_key):
        self.deleted.append(object_key)


class WorkspaceStub:
    def __init__(self, workspace_id):
        self.id = workspace_id


@pytest.mark.asyncio
async def test_upload_document_sets_processing_and_enqueues_ingestion_job():
    workspace_id = uuid.uuid4()
    workspace = WorkspaceStub(workspace_id)

    uow = FakeUoW(workspace)
    s3_repo = FakeS3Repository()
    queue = IngestionQueue()

    use_case = DocumentUseCase(
        uow=uow,
        s3_repository=s3_repo,
        ingestion_queue=queue,
    )

    document = await use_case.upload_document(
        workspace_id=workspace_id,
        filename="report.pdf",
        file_obj=BytesIO(b"fake pdf bytes"),
    )

    assert document.status == DocumentStatus.PROCESSING
    assert document.content_hash
    assert document.ingestion_version == 1
    assert document.is_active is True

    assert len(s3_repo.uploaded) == 1
    uploaded_bytes, object_key = s3_repo.uploaded[0]
    assert uploaded_bytes == b"fake pdf bytes"
    assert object_key == document.s3_object_key

    job = await queue.get()
    assert isinstance(job, IngestDocumentJob)
    assert job.document_id == document.id
    queue.task_done()