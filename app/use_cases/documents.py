import uuid
from typing import BinaryIO

from app.models.domain import Document, DocumentStatus
from app.repositories.s3storage_repository import S3StorageRepository
from app.repositories.uow import SqlAlchemyUnitOfWork


class DocumentUseCase:
    def __init__(
        self,
        uow: SqlAlchemyUnitOfWork,
        s3_repository: S3StorageRepository,
    ):
        self._uow = uow
        self._s3_repository = s3_repository

    async def upload_document(
        self,
        workspace_id: uuid.UUID,
        filename: str,
        file_obj: BinaryIO,
    ) -> Document:
        document_id = uuid.uuid4()
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
            )

        try:
            await self._s3_repository.upload_file(
                file_obj=file_obj,
                object_key=object_key,
            )
        except Exception:
            await self._mark_document_failed(document.id)
            await self._safe_delete_from_s3(object_key)
            raise

        return await self._update_document_status(
            document_id=document.id,
            status=DocumentStatus.READY,
        )

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
            document = await uow.documents.delete_document(document_id)

            if not document:
                raise ValueError("Document not found")

            object_key = document.s3_object_key

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
        return await self._update_document_status(
            document_id=document_id,
            status=DocumentStatus.READY,
        )

    async def mark_document_failed(
        self,
        document_id: uuid.UUID,
    ) -> Document:
        return await self._update_document_status(
            document_id=document_id,
            status=DocumentStatus.FAILED,
        )

    async def _update_document_status(
        self,
        document_id: uuid.UUID,
        status: DocumentStatus,
    ) -> Document:
        async with self._uow as uow:
            document = await uow.documents.update_status(
                document_id=document_id,
                status=status,
            )

            if not document:
                raise ValueError("Document not found")

            return document

    async def _mark_document_failed(self, document_id: uuid.UUID) -> None:
        try:
            await self._update_document_status(
                document_id=document_id,
                status=DocumentStatus.FAILED,
            )
        except Exception:
            pass

    async def _safe_delete_from_s3(self, object_key: str) -> None:
        try:
            await self._s3_repository.delete_file(object_key)
        except Exception:
            pass

    @staticmethod
    def _build_object_key(
        workspace_id: uuid.UUID,
        document_id: uuid.UUID,
        filename: str,
    ) -> str:
        return f"workspaces/{workspace_id}/documents/{document_id}/{filename}"