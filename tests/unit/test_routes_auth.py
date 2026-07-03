from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.dependencies import get_auth_use_case
from app.api.routes.router import api_router
from app.core.security import decode_access_token


class FakeAuthUseCase:
    def __init__(self):
        self.user_id = uuid4()
        self.workspace_id = uuid4()
        self.calls = []

    async def authenticate_user(self, *, email: str, password: str):
        self.calls.append({"email": email, "password": password})

        if email != "demo@example.com" or password != "password123":
            raise ValueError("Invalid credentials")

        return SimpleNamespace(
            id=self.user_id,
            email=email,
            workspace_id=self.workspace_id,
            created_at=datetime.utcnow(),
        )


def build_client():
    app = FastAPI()
    app.include_router(api_router)

    fake_auth = FakeAuthUseCase()

    def override_auth_use_case():
        return fake_auth

    app.dependency_overrides[get_auth_use_case] = override_auth_use_case

    return TestClient(app), fake_auth


def test_login_returns_bearer_token():
    client, fake_auth = build_client()

    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": "demo@example.com",
            "password": "password123",
        },
    )

    assert response.status_code == 200

    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["access_token"]

    payload = decode_access_token(body["access_token"])
    assert payload.sub == str(fake_auth.user_id)
    assert payload.workspace_id == fake_auth.workspace_id

    assert fake_auth.calls == [
        {
            "email": "demo@example.com",
            "password": "password123",
        }
    ]


def test_login_rejects_invalid_credentials():
    client, _fake_auth = build_client()

    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": "demo@example.com",
            "password": "wrong-password",
        },
    )

    assert response.status_code == 401