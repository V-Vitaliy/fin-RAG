from __future__ import annotations

import os
import uuid
from io import BytesIO
from pathlib import Path

import pytest

from app.infrastructure.duckdb import DuckDBManager
from app.infrastructure.postgres import build_postgres_engine, build_sessionmaker
from app.infrastructure.qdrant import build_qdrant_client
from app.infrastructure.rag import build_chunker, build_dense_embedder, build_sparse_embedder
from app.infrastructure.s3storage import s3session
from app.core.config import settings
from app.models.domain import DocumentStatus, WorkspaceType
from app.repositories.duckdb_repo import DuckDBTableRepository
from app.repositories.qdrant_repo import VectorIndexRepository
from app.repositories.s3storage_repository import S3StorageRepository
from app.repositories.uow import SqlAlchemyUnitOfWork
from app.services.ingestion.pipeline import IngestionPipeline
from app.services.jobs.ingestion_queue import IngestDocumentJob, IngestionQueue
from app.use_cases.documents import DocumentUseCase
from app.use_cases.ingestion import IngestDocumentUseCase


pytestmark = [
    pytest.mark.smoke,
    pytest.mark.slow,
    pytest.mark.asyncio,
]


def _smoke_pdf_path() -> Path:
    path = os.getenv("INGESTION_SMOKE_PDF")
    if not path:
        pytest.skip("Set INGESTION_SMOKE_PDF=/path/to/small.pdf to run this smoke test")

    pdf_path = Path(path)

    if not pdf_path.exists():
        pytest.skip(f"INGESTION_SMOKE_PDF does not exist: {pdf_path}")

    return pdf_path


async def _ensure_minio_bucket(s3_client, bucket_name: str) -> None:
    buckets = await s3_client.list_buckets()
    existing = {bucket["Name"] for bucket in buckets.get("Buckets", [])}

    if bucket_name not in existing:
        await s3_client.create_bucket(Bucket=bucket_name)


@pytest.mark.skipif(
    os.getenv("RUN_INGESTION_SMOKE") != "1",
    reason="Set RUN_INGESTION_SMOKE=1 to run real ingestion smoke test",
)
async def test_document_upload_is_processed_by_ingestion_pipeline():
    pdf_path = _smoke_pdf_path()

    postgres_engine = build_postgres_engine()
    db_sessionmaker = build_sessionmaker(postgres_engine)

    qdrant_client = build_qdrant_client()

    async with s3session.client(
        service_name="s3",
        endpoint_url=settings.S3_ENDPOINT_URL,
        aws_access_key_id=settings.S3_ACCESS_KEY,
        aws_secret_access_key=settings.S3_SECRET_KEY,
    ) as s3_client:
        await _ensure_minio_bucket(s3_client, settings.S3_BUCKET_NAME)

        s3_repo = S3StorageRepository(
            client=s3_client,
            bucket_name=settings.S3_BUCKET_NAME,
        )

        vector_repo = VectorIndexRepository(
            client=qdrant_client,
            collection_name=settings.RAG_COLLECTION_NAME,
        )

        duckdb_repo = DuckDBTableRepository(
            DuckDBManager(settings.RAG_DUCKDB_PATH)
        )

        pipeline = IngestionPipeline(
            vector_repo=vector_repo,
            duckdb_repo=duckdb_repo,
            chunker=build_chunker(),
            dense_embedder=build_dense_embedder(),
            sparse_embedder=build_sparse_embedder(),
        )

        ingestion_queue = IngestionQueue()

        document_use_case = DocumentUseCase(
            uow=SqlAlchemyUnitOfWork(db_sessionmaker),
            s3_repository=s3_repo,
            ingestion_queue=ingestion_queue,
        )

        ingest_use_case = IngestDocumentUseCase(
            uow_factory=lambda: SqlAlchemyUnitOfWork(db_sessionmaker),
            s3_repository=s3_repo,
            pipeline=pipeline,
        )

        workspace_id = uuid.uuid4()

        async with SqlAlchemyUnitOfWork(db_sessionmaker) as uow:
            workspace = await uow.workspaces.create_workspace(
                name=f"smoke-{workspace_id}",
                ws_type=WorkspaceType.PRIVATE,
            )
            workspace_id = workspace.id

        with pdf_path.open("rb") as file_obj:
            document = await document_use_case.upload_document(
                workspace_id=workspace_id,
                filename=pdf_path.name,
                file_obj=file_obj,
            )

        assert document.status == DocumentStatus.PROCESSING
        assert document.content_hash
        assert document.s3_object_key

        job = await ingestion_queue.get()
        assert isinstance(job, IngestDocumentJob)
        assert job.document_id == document.id

        try:
            await ingest_use_case.ingest(job)
        finally:
            ingestion_queue.task_done()

        async with SqlAlchemyUnitOfWork(db_sessionmaker) as uow:
            final_document = await uow.documents.get_document(document.id)

        assert final_document is not None
        assert final_document.status == DocumentStatus.READY
        assert final_document.failure_reason is None
        assert final_document.indexed_at is not None
        assert final_document.qdrant_points_count is not None
        assert final_document.qdrant_points_count > 0
        assert final_document.duckdb_tables_count is not None

    await qdrant_client.close()
    await postgres_engine.dispose()