from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.dependencies import get_document_use_case, get_workspace_id
from app.api.routes.router import api_router
from app.models.domain import DocumentStatus


class FakeDocumentUseCase:
    def __init__(self):
        self.calls = []
        self.workspace_id = uuid4()
        self.document_id = uuid4()

    def _document(self, *, filename: str = "test.pdf"):
        now = datetime.utcnow()
        return SimpleNamespace(
            id=self.document_id,
            workspace_id=self.workspace_id,
            filename=filename,
            status=DocumentStatus.PROCESSING,
            created_at=now,
            updated_at=now,
        )

    async def list_workspace_documents(self, workspace_id):
        self.calls.append(
            {
                "method": "list_workspace_documents",
                "workspace_id": workspace_id,
            }
        )
        return [self._document(filename="existing.pdf")]

    async def upload_document(self, *, workspace_id, filename, file_obj):
        content = file_obj.read()

        self.calls.append(
            {
                "method": "upload_document",
                "workspace_id": workspace_id,
                "filename": filename,
                "content": content,
            }
        )

        return self._document(filename=filename)


def build_client():
    app = FastAPI()
    app.include_router(api_router)

    fake_documents = FakeDocumentUseCase()

    async def override_workspace_id():
        return fake_documents.workspace_id

    def override_document_use_case():
        return fake_documents

    app.dependency_overrides[get_workspace_id] = override_workspace_id
    app.dependency_overrides[get_document_use_case] = override_document_use_case

    return TestClient(app), fake_documents


def test_list_documents_route():
    client, fake_documents = build_client()

    response = client.get("/api/v1/documents")

    assert response.status_code == 200

    body = response.json()
    assert len(body) == 1
    assert body[0]["filename"] == "existing.pdf"
    assert body[0]["status"] == DocumentStatus.PROCESSING.value

    assert fake_documents.calls[0]["method"] == "list_workspace_documents"


def test_upload_document_route():
    client, fake_documents = build_client()

    response = client.post(
        "/api/v1/documents",
        files={
            "file": (
                "uploaded.pdf",
                b"fake pdf bytes",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 201

    body = response.json()
    assert body["filename"] == "uploaded.pdf"
    assert body["status"] == DocumentStatus.PROCESSING.value

    call = fake_documents.calls[0]
    assert call["method"] == "upload_document"
    assert call["workspace_id"] == fake_documents.workspace_id
    assert call["filename"] == "uploaded.pdf"
    assert call["content"] == b"fake pdf bytes"