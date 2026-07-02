from __future__ import annotations

import logging
import tempfile
from collections.abc import Callable
from pathlib import Path

from app.repositories.s3storage_repository import S3StorageRepository
from app.repositories.uow import SqlAlchemyUnitOfWork
from app.services.ingestion.pipeline import IngestionDocument, IngestionPipeline
from app.services.jobs.ingestion_queue import IngestDocumentJob

logger = logging.getLogger(__name__)


class IngestDocumentUseCase:
    def __init__(
        self,
        *,
        uow_factory: Callable[[], SqlAlchemyUnitOfWork],
        s3_repository: S3StorageRepository,
        pipeline: IngestionPipeline,
    ):
        self._uow_factory = uow_factory
        self._s3_repository = s3_repository
        self._pipeline = pipeline

    async def ingest(self, job: IngestDocumentJob) -> None:
        async with self._uow_factory() as uow:
            document = await uow.documents.get_document(job.document_id)

            if not document:
                raise ValueError(f"Document not found: {job.document_id}")

        try:
            with tempfile.TemporaryDirectory(prefix="finrag_ingest_") as temp_dir:
                pdf_path = Path(temp_dir) / document.filename
                file_bytes = await self._s3_repository.get_bytes(document.s3_object_key)
                pdf_path.write_bytes(file_bytes)

                result = await self._pipeline.ingest_document(
                    IngestionDocument(
                        pdf_path=str(pdf_path),
                        document_id=document.id,
                        workspace_id=document.workspace_id,
                        filename=document.filename,
                        doc_name=Path(document.filename).stem,
                        content_hash=document.content_hash,
                        ingestion_version=document.ingestion_version,
                    )
                )

            async with self._uow_factory() as uow:
                await uow.documents.mark_ready(
                    document_id=document.id,
                    qdrant_points_count=result.qdrant_points_count,
                    duckdb_tables_count=result.duckdb_tables_count,
                )

            logger.info(
                "Document ingestion completed: document_id=%s points=%s tables=%s",
                document.id,
                result.qdrant_points_count,
                result.duckdb_tables_count,
            )

        except Exception as exc:
            logger.exception("Document ingestion failed: document_id=%s", job.document_id)

            async with self._uow_factory() as uow:
                await uow.documents.mark_failed(
                    document_id=job.document_id,
                    failure_reason=str(exc)[:2048],
                )

            raise