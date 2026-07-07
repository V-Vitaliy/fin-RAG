from __future__ import annotations

from typing import Annotated
from uuid import UUID
import json

from collections.abc import AsyncIterator
from fastapi.responses import StreamingResponse
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.api.dependencies import get_ingest_use_case
from app.use_cases.ingestion import IngestDocumentUseCase
from app.api.dependencies import get_document_use_case, get_uow, get_workspace_id
from app.models.schemas import DocumentResponse
from app.repositories.uow import SqlAlchemyUnitOfWork
from app.use_cases.documents import DocumentUseCase

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("", response_model=list[DocumentResponse])
async def list_documents(
    workspace_id: Annotated[UUID, Depends(get_workspace_id)],
    document_use_case: Annotated[DocumentUseCase, Depends(get_document_use_case)],
) -> list[DocumentResponse]:
    try:
        documents = await document_use_case.list_workspace_documents(workspace_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    return [DocumentResponse.model_validate(document) for document in documents]


@router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    workspace_id: Annotated[UUID, Depends(get_workspace_id)],
    document_use_case: Annotated[DocumentUseCase, Depends(get_document_use_case)],
    file: UploadFile = File(...),
) -> DocumentResponse:
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required.",
        )

    await file.seek(0)

    try:
        document = await document_use_case.upload_document(
            workspace_id=workspace_id,
            filename=file.filename,
            file_obj=file.file,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    return DocumentResponse.model_validate(document)


def _sse(event: str, data: dict) -> str:
    return (
        f"event: {event}\n"
        f"data: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"
    )

@router.post("/upload/stream")
async def upload_document_stream(
    workspace_id: Annotated[UUID, Depends(get_workspace_id)],
    document_use_case: Annotated[DocumentUseCase, Depends(get_document_use_case)],
    ingest_use_case: Annotated[IngestDocumentUseCase, Depends(get_ingest_use_case)],
    file: UploadFile = File(...),
) -> StreamingResponse:
    async def event_generator() -> AsyncIterator[str]:
        if not file.filename:
            yield _sse(
                "error",
                {"message": "Filename is required."},
            )
            yield _sse("done", {})
            return

        try:
            yield _sse(
                "stage",
                {
                    "stage": "uploading",
                    "message": "Uploading PDF",
                },
            )

            await file.seek(0)

            document = await document_use_case.upload_document(
                workspace_id=workspace_id,
                filename=file.filename,
                file_obj=file.file,
                enqueue=False,
            )

            yield _sse(
                "document_created",
                {
                    "document_id": str(document.id),
                    "filename": document.filename,
                    "status": str(document.status.value if hasattr(document.status, "value") else document.status),
                },
            )

            async for item in ingest_use_case.ingest_stream(document.id):
                yield _sse(
                    str(item.get("event") or "message"),
                    item.get("data") or {},
                )

        except ValueError as exc:
            yield _sse(
                "error",
                {"message": str(exc)},
            )
            yield _sse("done", {})

        except Exception as exc:
            yield _sse(
                "error",
                {
                    "message": "Upload or ingestion stream failed.",
                    "detail": str(exc),
                },
            )
            yield _sse("done", {})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: UUID,
    workspace_id: Annotated[UUID, Depends(get_workspace_id)],
    uow: Annotated[SqlAlchemyUnitOfWork, Depends(get_uow)],
) -> DocumentResponse:
    async with uow:
        document = await uow.documents.get_document(document_id)

    if not document or document.workspace_id != workspace_id or not document.is_active:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    return DocumentResponse.model_validate(document)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: UUID,
    workspace_id: Annotated[UUID, Depends(get_workspace_id)],
    uow: Annotated[SqlAlchemyUnitOfWork, Depends(get_uow)],
) -> None:
    async with uow:
        document = await uow.documents.get_document(document_id)

        if not document or document.workspace_id != workspace_id or not document.is_active:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found.",
            )

        await uow.documents.mark_inactive(document_id)
