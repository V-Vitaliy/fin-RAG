from __future__ import annotations

import os
import uuid
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from app.core.security import PasswordHasher
from app.main import app
from app.repositories.uow import SqlAlchemyUnitOfWork
from app.core.config import settings


pytestmark = pytest.mark.smoke


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        pytest.skip(f"{name} is required for agent smoke test")
    return value


def _smoke_document_id() -> UUID:
    raw = _require_env("AGENT_SMOKE_DOCUMENT_ID")
    return UUID(raw)


async def _get_document_workspace_id(db_sessionmaker, document_id: UUID) -> UUID:
    async with SqlAlchemyUnitOfWork(db_sessionmaker) as uow:
        document = await uow.documents.get_document(document_id)
        if not document:
            pytest.skip(f"Smoke document not found: {document_id}")

        return document.workspace_id


async def _ensure_smoke_user_in_workspace(
    db_sessionmaker,
    email: str,
    password: str,
    workspace_id: UUID,
):
    hasher = PasswordHasher()

    async with SqlAlchemyUnitOfWork(db_sessionmaker) as uow:
        existing = await uow.users.get_user_by_email(email)

        if existing:
            existing.password_hash = hasher.hash_password(password)
            existing.workspace_id = workspace_id
            return existing

        return await uow.users.create_user(
            email=email,
            password_hash=hasher.hash_password(password),
            workspace_id=workspace_id,
        )


def _auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _login(client: TestClient, *, email: str, password: str) -> str:
    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": email,
            "password": password,
        },
    )

    assert response.status_code == 200, response.text

    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["access_token"]

    return body["access_token"]


def _register(client: TestClient, *, email: str, password: str) -> None:
    response = client.post(
        "/api/v1/auth/register",
        json={
            "email": email,
            "password": password,
            "workspace_name": f"smoke-registration-{uuid.uuid4()}",
        },
    )

    # If the test was already run, duplicate user is fine.
    if response.status_code == 400 and "already exists" in response.text:
        return

    assert response.status_code == 201, response.text


def _base_payload(question: str) -> dict:
    document_id = _smoke_document_id()
    return {
        "question": question,
        "document_ids": [str(document_id)],
    }


def _search_text_payload(query: str) -> dict:
    document_id = _smoke_document_id()
    return {
        "query": query,
        "document_ids": [str(document_id)],
        "top_k": 5,
        "expand_neighbors": True,
        "neighbor_window": 1,
    }


def _search_tables_payload(concept: str) -> dict:
    document_id = _smoke_document_id()
    return {
        "concept": concept,
        "document_ids": [str(document_id)],
        "top_k": 5,
    }


@pytest.fixture(scope="module")
def smoke_client_and_headers():
    if os.getenv("RUN_AGENT_SMOKE") != "1":
        pytest.skip("Set RUN_AGENT_SMOKE=1 to run agent smoke tests")

    if not settings.OPENAI_API_KEY:
        pytest.skip("OPENAI_API_KEY is required in settings/.env for agent smoke test")

    document_id = _smoke_document_id()

    smoke_email = os.getenv("AGENT_SMOKE_EMAIL") or f"smoke-{document_id.hex[:8]}@example.com"
    smoke_password = os.getenv("AGENT_SMOKE_PASSWORD") or "password123"

    with TestClient(app) as client:
        workspace_id = client.portal.call(
            _get_document_workspace_id,
            client.app.state.db_sessionmaker,
            document_id,
        )

        client.portal.call(
            _ensure_smoke_user_in_workspace,
            client.app.state.db_sessionmaker,
            smoke_email,
            smoke_password,
            workspace_id,
        )

        _register(
            client,
            email=f"registration-{uuid.uuid4().hex}@example.com",
            password="password123",
        )

        token = _login(
            client,
            email=smoke_email,
            password=smoke_password,
        )

        yield client, _auth_headers(token)


def test_smoke_retrieval_text_route(smoke_client_and_headers):
    client, headers = smoke_client_and_headers

    response = client.post(
        "/api/v1/rag/search/text",
        json=_search_text_payload("revenue results of operations"),
        headers=headers,
    )

    assert response.status_code == 200, response.text

    body = response.json()
    assert isinstance(body.get("text"), str)
    assert body["text"].strip()
    assert "No accessible ready documents found" not in body["text"]


def test_smoke_retrieval_tables_route(smoke_client_and_headers):
    client, headers = smoke_client_and_headers

    response = client.post(
        "/api/v1/rag/search/tables",
        json=_search_tables_payload("income statement revenue"),
        headers=headers,
    )

    assert response.status_code == 200, response.text

    body = response.json()
    assert isinstance(body.get("text"), str)
    assert body["text"].strip()
    assert "No accessible ready documents found" not in body["text"]


def test_smoke_agent_ask_non_stream(smoke_client_and_headers):
    client, headers = smoke_client_and_headers

    response = client.post(
        "/api/v1/rag/ask",
        json=_base_payload("What was the company's revenue? Answer with citations."),
        headers=headers,
        timeout=120,
    )

    assert response.status_code == 200, response.text

    body = response.json()
    assert isinstance(body.get("answer"), str)
    assert body["answer"].strip()
    assert not body["answer"].startswith("API Error")
    assert "No accessible ready documents found" not in body["answer"]
    assert isinstance(body.get("tool_calls"), list)
    assert body["tool_calls"], body


def test_smoke_agent_ask_stream_sse(smoke_client_and_headers):
    client, headers = smoke_client_and_headers

    with client.stream(
        "POST",
        "/api/v1/rag/ask/stream",
        json=_base_payload("What was the company's revenue? Answer with citations."),
        headers=headers,
        timeout=120,
    ) as response:
        assert response.status_code == 200, response.text
        assert response.headers["content-type"].startswith("text/event-stream")

        raw = "".join(response.iter_text())

    assert "event: stage" in raw
    assert "event: final_answer" in raw
    assert "event: done" in raw
    assert "event: token" not in raw
    assert '"citations"' in raw
    assert '"source_documents"' in raw
    assert "event: error" not in raw
    assert "No accessible ready documents found" not in raw
    assert "API Error" not in raw