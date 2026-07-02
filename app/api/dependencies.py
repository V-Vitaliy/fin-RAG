from collections.abc import AsyncGenerator

from fastapi import Request
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.s3storage_repository import S3StorageRepository
from app.repositories.uow import SqlAlchemyUnitOfWork
from app.core.security import PasswordHasher
from app.use_cases.auth import AuthUseCase
from app.use_cases.documents import DocumentUseCase
from app.services.jobs.ingestion_queue import IngestionQueue


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