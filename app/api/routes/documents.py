from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

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
