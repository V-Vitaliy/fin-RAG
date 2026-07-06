from __future__ import annotations

from fastapi import APIRouter

from app.api.routes.auth import router as auth_router
from app.api.routes.documents import router as documents_router
from app.api.routes.rag import router as rag_router

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(auth_router)
api_router.include_router(documents_router)
api_router.include_router(rag_router)