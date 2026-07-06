from __future__ import annotations

from openai import AsyncOpenAI

from app.core.config import settings


def build_openai_client() -> AsyncOpenAI:
    if not settings.OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured. "
            "Set it in .env or export it before starting the app."
        )

    return AsyncOpenAI(api_key=settings.OPENAI_API_KEY)