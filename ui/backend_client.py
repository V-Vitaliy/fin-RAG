from __future__ import annotations

import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from ui.sse import SSEEvent, parse_sse_lines


DEFAULT_API_BASE_URL = "http://localhost:8000/api/v1"


@dataclass(frozen=True)
class BackendToken:
    access_token: str
    token_type: str


class BackendClient:
    def __init__(self, *, base_url: str | None = None, timeout: float = 120.0):
        self.base_url = (
            base_url
            or os.getenv("FIN_RAG_API_BASE_URL")
            or DEFAULT_API_BASE_URL
        ).rstrip("/")
        self.timeout = float(timeout)

    async def login(self, *, email: str, password: str) -> BackendToken:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/auth/login",
                json={
                    "email": email,
                    "password": password,
                },
            )

        if response.status_code != 200:
            raise ValueError("Invalid email or password.")

        payload = response.json()

        access_token = payload.get("access_token")
        token_type = payload.get("token_type") or "bearer"

        if not access_token:
            raise ValueError("Backend did not return access_token.")

        return BackendToken(
            access_token=str(access_token),
            token_type=str(token_type),
        )

    async def upload_document(
        self,
        *,
        access_token: str,
        file_path: str | Path,
        filename: str | None = None,
        content_type: str | None = None,
    ) -> dict[str, Any]:
        path = Path(file_path)
        upload_name = filename or path.name
        mime_type = content_type or "application/pdf"

        headers = {
            "Authorization": f"Bearer {access_token}",
        }

        async with httpx.AsyncClient(timeout=None) as client:
            with path.open("rb") as file_obj:
                response = await client.post(
                    f"{self.base_url}/documents",
                    headers=headers,
                    files={
                        "file": (
                            upload_name,
                            file_obj,
                            mime_type,
                        )
                    },
                )

        if response.status_code not in {200, 201}:
            raise RuntimeError(
                f"Document upload failed with status={response.status_code}: {response.text}"
            )

        return response.json()

    async def list_documents(
        self,
        *,
        access_token: str,
    ) -> list[dict[str, Any]]:
        headers = {
            "Authorization": f"Bearer {access_token}",
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(
                f"{self.base_url}/documents",
                headers=headers,
            )

        if response.status_code != 200:
            raise RuntimeError(
                f"Document list failed with status={response.status_code}: {response.text}"
            )

        payload = response.json()
        if isinstance(payload, list):
            return payload

        if isinstance(payload, dict) and isinstance(payload.get("items"), list):
            return payload["items"]

        return []

    async def ask_stream(
        self,
        *,
        access_token: str,
        question: str,
        document_ids: list[str] | None = None,
    ) -> AsyncIterator[SSEEvent]:

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "text/event-stream",
        }

        payload: dict[str, Any] = {
            "question": question,
        }

        if document_ids:
            payload["document_ids"] = document_ids

        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/rag/ask/stream",
                headers=headers,
                json=payload,
            ) as response:
                if response.status_code != 200:
                    raw = await response.aread()
                    detail = raw.decode("utf-8", errors="replace")
                    raise RuntimeError(
                        f"Backend stream failed with status={response.status_code}: {detail}"
                    )

                async for event in parse_sse_lines(response.aiter_lines()):
                    yield event

    async def upload_document_stream(
            self,
            *,
            access_token: str,
            file_path: str | Path,
            filename: str | None = None,
            content_type: str | None = None,
    ) -> AsyncIterator[SSEEvent]:
        path = Path(file_path)
        upload_name = filename or path.name
        mime_type = content_type or "application/pdf"

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "text/event-stream",
        }

        async with httpx.AsyncClient(timeout=None) as client:
            with path.open("rb") as file_obj:
                async with client.stream(
                        "POST",
                        f"{self.base_url}/documents/upload/stream",
                        headers=headers,
                        files={
                            "file": (
                                    upload_name,
                                    file_obj,
                                    mime_type,
                            )
                        },
                ) as response:
                    if response.status_code != 200:
                        raw = await response.aread()
                        detail = raw.decode("utf-8", errors="replace")
                        raise RuntimeError(
                            f"Document upload stream failed with status={response.status_code}: {detail}"
                        )

                    async for event in parse_sse_lines(response.aiter_lines()):
                        yield event