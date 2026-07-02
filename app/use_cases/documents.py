from __future__ import annotations

import hashlib
import uuid
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

from app.models.domain import Document, DocumentStatus
from app.repositories.s3storage_repository import S3StorageRepository
from app.repositories.uow import SqlAlchemyUnitOfWork
from app.services.jobs.ingestion_queue import IngestDocumentJob, IngestionQueue


class DocumentUseCase:
    def __init__(
        self,
        uow: SqlAlchemyUnitOfWork,
        s3_repository: S3StorageRepository,
        ingestion_queue: IngestionQueue,
    ):
        self._uow = uow
        self._s3_repository = s3_repository
        self._ingestion_queue = ingestion_queue

    async def upload_document(
        self,
        workspace_id: uuid.UUID,
        filename: str,
        file_obj: BinaryIO,
    ) -> Document:
        document_id = uuid.uuid4()
        filename = self._normalize_filename(filename)
        file_bytes = self._read_file_bytes(file_obj)
        content_hash = hashlib.sha256(file_bytes).hexdigest()

        object_key = self._build_object_key(
            workspace_id=workspace_id,
            document_id=document_id,
            filename=filename,
        )

        async with self._uow as uow:
            workspace = await uow.workspaces.get_workspace(workspace_id)
            if not workspace:
                raise ValueError("Workspace not found")

            document = await uow.documents.create_document(
                document_id=document_id,
                workspace_id=workspace_id,
                filename=filename,
                s3_object_key=object_key,
                content_hash=content_hash,
                status=DocumentStatus.UPLOADING,
                ingestion_version=1,
            )

        try:
            await self._s3_repository.upload_file(
                file_obj=BytesIO(file_bytes),
                object_key=object_key,
            )
        except Exception:
            await self._mark_document_failed(document.id, "S3 upload failed")
            await self._safe_delete_from_s3(object_key)
            raise

        document = await self._update_document_status(
            document_id=document.id,
            status=DocumentStatus.PROCESSING,
        )

        await self._ingestion_queue.enqueue(
            IngestDocumentJob(document_id=document.id)
        )

        return document

    async def list_workspace_documents(
        self,
        workspace_id: uuid.UUID,
    ) -> list[Document]:
        async with self._uow as uow:
            workspace = await uow.workspaces.get_workspace(workspace_id)
            if not workspace:
                raise ValueError("Workspace not found")

            return await uow.documents.get_by_workspace(workspace_id)

    async def delete_document(
        self,
        document_id: uuid.UUID,
    ) -> Document:
        async with self._uow as uow:
            document = await uow.documents.get_document(document_id)

            if not document:
                raise ValueError("Document not found")

            object_key = document.s3_object_key
            await uow.documents.mark_inactive(document_id)

        await self._safe_delete_from_s3(object_key)

        return document

    async def mark_document_processing(
        self,
        document_id: uuid.UUID,
    ) -> Document:
        return await self._update_document_status(
            document_id=document_id,
            status=DocumentStatus.PROCESSING,
        )

    async def mark_document_ready(
        self,
        document_id: uuid.UUID,
    ) -> Document:
        async with self._uow as uow:
            document = await uow.documents.mark_ready(document_id=document_id)

            if not document:
                raise ValueError("Document not found")

            return document

    async def mark_document_failed(
        self,
        document_id: uuid.UUID,
        failure_reason: str | None = None,
    ) -> Document:
        return await self._update_document_status(
            document_id=document_id,
            status=DocumentStatus.FAILED,
            failure_reason=failure_reason,
        )

    async def _update_document_status(
        self,
        document_id: uuid.UUID,
        status: DocumentStatus,
        failure_reason: str | None = None,
    ) -> Document:
        async with self._uow as uow:
            document = await uow.documents.update_status(
                document_id=document_id,
                status=status,
                failure_reason=failure_reason,
            )

            if not document:
                raise ValueError("Document not found")

            return document

    async def _mark_document_failed(
        self,
        document_id: uuid.UUID,
        failure_reason: str | None = None,
    ) -> None:
        try:
            await self._update_document_status(
                document_id=document_id,
                status=DocumentStatus.FAILED,
                failure_reason=failure_reason,
            )
        except Exception:
            pass

    async def _safe_delete_from_s3(self, object_key: str) -> None:
        try:
            await self._s3_repository.delete_file(object_key)
        except Exception:
            pass

    @staticmethod
    def _read_file_bytes(file_obj: BinaryIO) -> bytes:
        try:
            file_obj.seek(0)
        except Exception:
            pass

        data = file_obj.read()

        if isinstance(data, str):
            return data.encode("utf-8")

        return bytes(data)

    @staticmethod
    def _normalize_filename(filename: str) -> str:
        name = Path(str(filename or "document.pdf")).name.strip()
        return name or "document.pdf"

    @staticmethod
    def _build_object_key(
        workspace_id: uuid.UUID,
        document_id: uuid.UUID,
        filename: str,
    ) -> str:
        return f"workspaces/{workspace_id}/documents/{document_id}/{filename}"