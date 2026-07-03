from __future__ import annotations

from collections.abc import AsyncGenerator
from uuid import UUID
from fastapi import Header, HTTPException, Request, status
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Annotated

from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer

from app.core.security import decode_access_token
from app.models.domain import User
from app.core.config import settings
from app.use_cases.agent import AskAgentUseCase
from app.use_cases.ingestion import IngestDocumentUseCase
from app.use_cases.retrieval import RetrieveDocumentsUseCase
from app.repositories.s3storage_repository import S3StorageRepository
from app.repositories.uow import SqlAlchemyUnitOfWork
from app.core.security import PasswordHasher
from app.use_cases.auth import AuthUseCase
from app.use_cases.documents import DocumentUseCase
from app.services.jobs.ingestion_queue import IngestionQueue

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


async def get_db(request: Request) -> AsyncGenerator[AsyncSession, None]:
    sessionmaker = request.app.state.db_sessionmaker

    async with sessionmaker() as session:
        yield session

def get_uow(request: Request) -> SqlAlchemyUnitOfWork:
    return SqlAlchemyUnitOfWork(request.app.state.db_sessionmaker)


def get_s3_repository(request: Request) -> S3StorageRepository:
    return request.app.state.s3_repo


def get_qdrant_client(request: Request) -> AsyncQdrantClient:
    return request.app.state.qdrant_client

def get_password_hasher() -> PasswordHasher:
    return PasswordHasher()


def get_auth_use_case(request: Request) -> AuthUseCase:
    return AuthUseCase(
        uow=SqlAlchemyUnitOfWork(request.app.state.db_sessionmaker),
        password_hasher=PasswordHasher(),
    )

def get_ingestion_queue(request: Request) -> IngestionQueue:
    return request.app.state.ingestion_queue

def get_document_use_case(request: Request) -> DocumentUseCase:
    return DocumentUseCase(
        uow=SqlAlchemyUnitOfWork(request.app.state.db_sessionmaker),
        s3_repository=request.app.state.s3_repo,
        ingestion_queue=request.app.state.ingestion_queue,
    )

async def get_workspace_id(
    current_user: Annotated[User, Depends(get_current_user)],
) -> UUID:
    return current_user.workspace_id


def get_ingest_use_case(request: Request) -> IngestDocumentUseCase:
    use_case = getattr(request.app.state, "ingest_use_case", None)
    if not use_case:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ingestion service is not initialized.",
        )
    return use_case


def get_retrieval_use_case(request: Request) -> RetrieveDocumentsUseCase:
    use_case = getattr(request.app.state, "retrieval_use_case", None)
    if not use_case:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Retrieval service is not initialized.",
        )
    return use_case


def get_agent_use_case(request: Request) -> AskAgentUseCase:
    use_case = getattr(request.app.state, "agent_use_case", None)
    if not use_case:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent service is not initialized.",
        )
    return use_case


def get_uow_factory(request: Request):
    uow_factory = getattr(request.app.state, "uow_factory", None)
    if not uow_factory:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service is not initialized.",
        )
    return uow_factory

async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    auth_use_case: Annotated[AuthUseCase, Depends(get_auth_use_case)],
) -> User:
    try:
        payload = decode_access_token(token)
        if not payload.sub:
            raise ValueError("Token subject is missing")

        return await auth_use_case.get_user_by_id(UUID(str(payload.sub)))
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc