from __future__ import annotations

import asyncio
import logging
import tempfile
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any
from uuid import UUID

from app.repositories.s3storage_repository import S3StorageRepository
from app.repositories.uow import SqlAlchemyUnitOfWork
from app.services.ingestion.pipeline import IngestionDocument, IngestionPipeline
from app.services.jobs.ingestion_queue import IngestDocumentJob

logger = logging.getLogger(__name__)

IngestionEventSink = Callable[[str, dict[str, Any]], Awaitable[None]]


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
        try:
            result = await self._ingest_document_by_id(
                document_id=job.document_id,
                event_sink=None,
            )

            logger.info(
                "Document ingestion completed: document_id=%s points=%s tables=%s",
                job.document_id,
                result.qdrant_points_count,
                result.duckdb_tables_count,
            )

        except Exception:
            logger.exception("Document ingestion failed: document_id=%s", job.document_id)
            raise

    async def ingest_stream(
        self,
        document_id: UUID,
    ) -> AsyncIterator[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        async def emit(event: str, data: dict[str, Any]) -> None:
            await queue.put({"event": event, "data": data})

        async def run_ingestion() -> None:
            try:
                result = await self._ingest_document_by_id(
                    document_id=document_id,
                    event_sink=emit,
                )

                await emit(
                    "ready",
                    {
                        "document_id": str(document_id),
                        "status": "READY",
                        "qdrant_points_count": result.qdrant_points_count,
                        "duckdb_tables_count": result.duckdb_tables_count,
                    },
                )

            except Exception as exc:
                logger.exception(
                    "Document stream ingestion failed: document_id=%s",
                    document_id,
                )

                await emit(
                    "error",
                    {
                        "document_id": str(document_id),
                        "message": "Document ingestion failed.",
                        "detail": str(exc),
                    },
                )

            finally:
                await emit(
                    "done",
                    {
                        "document_id": str(document_id),
                    },
                )

        task = asyncio.create_task(run_ingestion())

        try:
            while not task.done() or not queue.empty():
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=0.1)
                    yield item
                except asyncio.TimeoutError:
                    continue

            await task

        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    async def _ingest_document_by_id(
        self,
        *,
        document_id: UUID,
        event_sink: IngestionEventSink | None,
    ):
        async with self._uow_factory() as uow:
            document = await uow.documents.get_document(document_id)

            if not document:
                raise ValueError(f"Document not found: {document_id}")

            ingestion_document = IngestionDocument(
                pdf_path="",  # filled after S3 download
                document_id=document.id,
                workspace_id=document.workspace_id,
                filename=document.filename,
                doc_name=Path(document.filename).stem,
                content_hash=document.content_hash,
                ingestion_version=document.ingestion_version,
            )
            s3_object_key = document.s3_object_key
            filename = document.filename

        try:
            if event_sink is not None:
                await event_sink(
                    "stage",
                    {
                        "stage": "downloading",
                        "message": "Downloading uploaded PDF from storage",
                        "document_id": str(document_id),
                    },
                )

            with tempfile.TemporaryDirectory(prefix="finrag_ingest_") as temp_dir:
                pdf_path = Path(temp_dir) / filename

                file_bytes = await self._s3_repository.get_bytes(s3_object_key)

                await asyncio.to_thread(
                    pdf_path.write_bytes,
                    file_bytes,
                )

                if event_sink is not None:
                    await event_sink(
                        "stage",
                        {
                            "stage": "processing",
                            "message": "Processing PDF",
                            "document_id": str(document_id),
                        },
                    )

                ingestion_document = IngestionDocument(
                    pdf_path=str(pdf_path),
                    document_id=ingestion_document.document_id,
                    workspace_id=ingestion_document.workspace_id,
                    filename=ingestion_document.filename,
                    doc_name=ingestion_document.doc_name,
                    content_hash=ingestion_document.content_hash,
                    ingestion_version=ingestion_document.ingestion_version,
                )

                result = await self._pipeline.ingest_document(
                    ingestion_document,
                    event_sink=event_sink,
                )

            async with self._uow_factory() as uow:
                await uow.documents.mark_ready(
                    document_id=document_id,
                    qdrant_points_count=result.qdrant_points_count,
                    duckdb_tables_count=result.duckdb_tables_count,
                )

            return result

        except Exception as exc:
            async with self._uow_factory() as uow:
                await uow.documents.mark_failed(
                    document_id=document_id,
                    failure_reason=str(exc)[:2048],
                )

            raise