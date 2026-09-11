from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
import time
import uuid
from contextlib import AsyncExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.infrastructure.duckdb import DuckDBManager
from app.infrastructure.postgres import build_postgres_engine, build_sessionmaker
from app.infrastructure.qdrant import build_qdrant_client
from app.infrastructure.rag import (
    build_chunker,
    build_dense_embedder,
    build_sparse_embedder,
)
from app.infrastructure.s3storage import s3session
from app.models.domain import Document, DocumentStatus, Workspace, WorkspaceType
from app.repositories.duckdb_repo import DuckDBTableRepository
from app.repositories.qdrant_repo import VectorIndexRepository
from app.repositories.s3storage_repository import S3StorageRepository
from app.repositories.uow import SqlAlchemyUnitOfWork
from app.services.ingestion.pipeline import IngestionPipeline
from app.services.jobs.ingestion_queue import IngestDocumentJob, IngestionQueue
from app.use_cases.documents import DocumentUseCase
from app.use_cases.ingestion import IngestDocumentUseCase


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_run_id(run_name: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_name = run_name.strip().replace(" ", "_")
    return f"{safe_name}_{timestamp}"


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")
        file.flush()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    documents_by_name: dict[str, dict[str, Any]] = {}

    with manifest_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            raw = line.strip()
            if not raw:
                continue

            item = json.loads(raw)

            doc_name = str(item.get("doc_name") or "").strip()
            if not doc_name:
                raise ValueError(f"Missing doc_name in manifest line {line_number}")

            expected_filename = str(
                item.get("expected_filename") or f"{doc_name}.pdf"
            ).strip()

            item["doc_name"] = doc_name
            item["expected_filename"] = expected_filename

            documents_by_name[doc_name] = item

    return list(documents_by_name.values())


def require_workspace_id(raw_value: str | None) -> uuid.UUID:
    value = raw_value or settings.RAG_GLOBAL_WORKSPACE_ID

    if not value:
        raise ValueError(
            "Global workspace id is required. "
            "Set RAG_GLOBAL_WORKSPACE_ID or pass --workspace-id."
        )

    return uuid.UUID(str(value))


def public_settings_snapshot() -> dict[str, Any]:
    keys = [
        "RAG_COLLECTION_NAME",
        "RAG_DUCKDB_PATH",
        "RAG_USE_OPENAI_EMBEDDINGS",
        "RAG_EMBEDDING_MODEL",
        "RAG_EMBEDDING_DIMENSIONS",
        "RAG_EMBEDDING_BATCH_SIZE",
        "RAG_USE_RERANKER",
        "RAG_RERANKER_MODE",
        "RAG_CPU_RERANKER_MODEL",
        "RAG_GPU_RERANKER_MODEL",
        "RAG_CHUNK_MAX_TOKENS",
        "RAG_CHUNK_OVERLAP_TOKENS",
        "RAG_CHUNK_MIN_TOKENS",
        "RAG_CHUNK_MERGE_RATIO",
        "QDRANT_URL",
        "QDRANT_TIMEOUT",
        "S3_BUCKET_NAME",
        "S3_ENDPOINT_URL",
        "RAG_GLOBAL_WORKSPACE_ID",
    ]

    snapshot: dict[str, Any] = {}

    for key in keys:
        snapshot[key] = getattr(settings, key, None)

    if snapshot.get("QDRANT_URL"):
        snapshot["QDRANT_URL"] = str(snapshot["QDRANT_URL"]).split("?")[0]

    return snapshot


async def ensure_global_workspace(
    *,
    uow_factory,
    workspace_id: uuid.UUID,
    name: str,
) -> dict[str, Any]:
    async with uow_factory() as uow:
        existing = await uow.workspaces.get_workspace(workspace_id)

        if existing:
            return {
                "created": False,
                "workspace_id": str(existing.id),
                "name": existing.name,
                "type": existing.type.value,
            }

        workspace = Workspace(
            id=workspace_id,
            name=name,
            type=WorkspaceType.GLOBAL,
            company_id=None,
        )

        uow.session.add(workspace)
        await uow.session.flush()
        await uow.session.refresh(workspace)

        return {
            "created": True,
            "workspace_id": str(workspace.id),
            "name": workspace.name,
            "type": workspace.type.value,
        }


async def get_ready_document_by_hash(
    *,
    uow_factory,
    workspace_id: uuid.UUID,
    content_hash: str,
) -> Document | None:
    async with uow_factory() as uow:
        existing = await uow.documents.get_active_by_hash(
            workspace_id=workspace_id,
            content_hash=content_hash,
        )

        if existing and existing.status == DocumentStatus.READY:
            return existing

        return None


async def get_document_by_id(
    *,
    uow_factory,
    document_id: uuid.UUID,
) -> Document | None:
    async with uow_factory() as uow:
        return await uow.documents.get_document(document_id)


async def ingest_batch(args: argparse.Namespace) -> int:
    workspace_id = require_workspace_id(args.workspace_id)
    run_id = args.run_id or make_run_id(args.run_name)

    eval_run_dir = args.eval_root / run_id
    batch_run_dir = args.batch_root / run_id

    ingestion_dir = eval_run_dir / "ingestion"
    logs_dir = eval_run_dir / "logs"

    ingestion_results_path = ingestion_dir / "ingestion_results.jsonl"
    ingestion_summary_path = ingestion_dir / "ingestion_summary.json"
    document_id_map_path = batch_run_dir / "document_id_map.json"
    ingested_documents_path = batch_run_dir / "ingested_documents.jsonl"

    for directory in [eval_run_dir, batch_run_dir, ingestion_dir, logs_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    documents = load_manifest(args.manifest)

    shutil.copy2(args.manifest, eval_run_dir / "document_manifest.snapshot.jsonl")

    run_config = {
        "run_id": run_id,
        "run_name": args.run_name,
        "created_at": utc_now_iso(),
        "workspace_id": str(workspace_id),
        "manifest": str(args.manifest),
        "pdf_dir": str(args.pdf_dir),
        "eval_run_dir": str(eval_run_dir),
        "batch_run_dir": str(batch_run_dir),
        "document_count": len(documents),
        "settings": public_settings_snapshot(),
    }
    write_json(eval_run_dir / "run_config.json", run_config)

    postgres_engine = build_postgres_engine()
    db_sessionmaker = build_sessionmaker(postgres_engine)

    async with AsyncExitStack() as stack:
        stack.push_async_callback(postgres_engine.dispose)

        uow_factory = lambda: SqlAlchemyUnitOfWork(db_sessionmaker)

        s3_client = await stack.enter_async_context(
            s3session.client(
                service_name="s3",
                endpoint_url=settings.S3_ENDPOINT_URL,
                aws_access_key_id=settings.S3_ACCESS_KEY,
                aws_secret_access_key=settings.S3_SECRET_KEY,
            )
        )

        s3_repo = S3StorageRepository(
            client=s3_client,
            bucket_name=settings.S3_BUCKET_NAME,
        )

        qdrant_client = build_qdrant_client()
        stack.push_async_callback(qdrant_client.close)

        vector_repo = VectorIndexRepository(
            client=qdrant_client,
            collection_name=settings.RAG_COLLECTION_NAME,
        )

        duckdb_manager = DuckDBManager(path=settings.RAG_DUCKDB_PATH)
        duckdb_repo = DuckDBTableRepository(duckdb_manager)

        dense_embedder = build_dense_embedder()
        sparse_embedder = build_sparse_embedder()
        chunker = build_chunker()

        pipeline = IngestionPipeline(
            vector_repo=vector_repo,
            duckdb_repo=duckdb_repo,
            chunker=chunker,
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
        )

        document_use_case = DocumentUseCase(
            uow=SqlAlchemyUnitOfWork(db_sessionmaker),
            s3_repository=s3_repo,
            ingestion_queue=IngestionQueue(),
        )

        ingest_use_case = IngestDocumentUseCase(
            uow_factory=uow_factory,
            s3_repository=s3_repo,
            pipeline=pipeline,
        )

        workspace_result = await ensure_global_workspace(
            uow_factory=uow_factory,
            workspace_id=workspace_id,
            name=args.workspace_name,
        )
        write_json(eval_run_dir / "global_workspace.json", workspace_result)

        counters = {
            "total": len(documents),
            "ready": 0,
            "skipped_existing_ready": 0,
            "failed": 0,
            "missing_pdf": 0,
        }

        document_id_map: dict[str, Any] = {}

        for index, item in enumerate(documents, start=1):
            started = time.perf_counter()

            doc_name = item["doc_name"]
            expected_filename = item["expected_filename"]
            pdf_path = args.pdf_dir / expected_filename

            base_row = {
                "event": "document_ingestion",
                "run_id": run_id,
                "index": index,
                "total": len(documents),
                "doc_name": doc_name,
                "filename": expected_filename,
                "pdf_path": str(pdf_path),
                "workspace_id": str(workspace_id),
                "started_at": utc_now_iso(),
            }

            print(f"[{index}/{len(documents)}] {doc_name}")

            if not pdf_path.exists() or not pdf_path.is_file():
                row = {
                    **base_row,
                    "status": "MISSING_PDF",
                    "error": f"PDF not found: {pdf_path}",
                    "duration_seconds": round(time.perf_counter() - started, 3),
                }
                counters["missing_pdf"] += 1
                counters["failed"] += 1
                append_jsonl(ingestion_results_path, row)
                append_jsonl(ingested_documents_path, row)

                if args.stop_on_error:
                    break

                continue

            try:
                content_hash = sha256_file(pdf_path)

                existing_ready = None
                if args.skip_existing_ready:
                    existing_ready = await get_ready_document_by_hash(
                        uow_factory=uow_factory,
                        workspace_id=workspace_id,
                        content_hash=content_hash,
                    )

                if existing_ready:
                    row = {
                        **base_row,
                        "status": "SKIPPED_EXISTING_READY",
                        "document_id": str(existing_ready.id),
                        "content_hash": existing_ready.content_hash,
                        "document_status": existing_ready.status.value,
                        "qdrant_points_count": existing_ready.qdrant_points_count,
                        "duckdb_tables_count": existing_ready.duckdb_tables_count,
                        "duration_seconds": round(time.perf_counter() - started, 3),
                        "error": None,
                    }

                    document_id_map[doc_name] = {
                        "document_id": str(existing_ready.id),
                        "filename": existing_ready.filename,
                        "status": existing_ready.status.value,
                        "content_hash": existing_ready.content_hash,
                    }

                    counters["skipped_existing_ready"] += 1
                    counters["ready"] += 1

                    append_jsonl(ingestion_results_path, row)
                    append_jsonl(ingested_documents_path, row)
                    write_json(document_id_map_path, document_id_map)
                    continue

                with pdf_path.open("rb") as file_obj:
                    document = await document_use_case.upload_document(
                        workspace_id=workspace_id,
                        filename=expected_filename,
                        file_obj=file_obj,
                        enqueue=False,
                    )

                await ingest_use_case.ingest(
                    IngestDocumentJob(document_id=document.id)
                )

                final_document = await get_document_by_id(
                    uow_factory=uow_factory,
                    document_id=document.id,
                )

                if final_document is None:
                    raise RuntimeError(f"Document disappeared after ingestion: {document.id}")

                row = {
                    **base_row,
                    "status": final_document.status.value,
                    "document_id": str(final_document.id),
                    "s3_object_key": final_document.s3_object_key,
                    "content_hash": final_document.content_hash,
                    "qdrant_points_count": final_document.qdrant_points_count,
                    "duckdb_tables_count": final_document.duckdb_tables_count,
                    "indexed_at": (
                        final_document.indexed_at.isoformat()
                        if final_document.indexed_at
                        else None
                    ),
                    "duration_seconds": round(time.perf_counter() - started, 3),
                    "error": final_document.failure_reason,
                }

                document_id_map[doc_name] = {
                    "document_id": str(final_document.id),
                    "filename": final_document.filename,
                    "status": final_document.status.value,
                    "content_hash": final_document.content_hash,
                }

                if final_document.status == DocumentStatus.READY:
                    counters["ready"] += 1
                else:
                    counters["failed"] += 1

                append_jsonl(ingestion_results_path, row)
                append_jsonl(ingested_documents_path, row)
                write_json(document_id_map_path, document_id_map)

            except Exception as exc:
                row = {
                    **base_row,
                    "status": "FAILED",
                    "duration_seconds": round(time.perf_counter() - started, 3),
                    "error": repr(exc),
                }

                counters["failed"] += 1
                append_jsonl(ingestion_results_path, row)
                append_jsonl(ingested_documents_path, row)

                print(f"FAILED {doc_name}: {exc!r}")

                if args.stop_on_error:
                    break

        summary = {
            "run_id": run_id,
            "finished_at": utc_now_iso(),
            "workspace_id": str(workspace_id),
            "counters": counters,
            "ingestion_results_path": str(ingestion_results_path),
            "document_id_map_path": str(document_id_map_path),
        }
        write_json(ingestion_summary_path, summary)

        print(json.dumps(summary, ensure_ascii=False, indent=2))

        return 1 if counters["failed"] else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload and ingest PDFs for a global evaluation workspace."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pdf-dir", type=Path, required=True)
    parser.add_argument("--workspace-id", default=None)
    parser.add_argument(
        "--workspace-name",
        default="Global FinanceBench Workspace",
    )
    parser.add_argument("--run-name", default="subset20_prodlike")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--eval-root", type=Path, default=Path("/data/evals"))
    parser.add_argument("--batch-root", type=Path, default=Path("/data/batches"))
    parser.add_argument(
        "--skip-existing-ready",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return asyncio.run(ingest_batch(args))


if __name__ == "__main__":
    raise SystemExit(main())