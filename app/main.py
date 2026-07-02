from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager, suppress

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.infrastructure.duckdb import DuckDBManager
from app.infrastructure.postgres import build_postgres_engine, build_sessionmaker
from app.infrastructure.qdrant import build_qdrant_client
from app.infrastructure.s3storage import s3session
from app.repositories.duckdb_repo import DuckDBTableRepository
from app.repositories.qdrant_repo import VectorIndexRepository
from app.repositories.s3storage_repository import S3StorageRepository
from app.repositories.uow import SqlAlchemyUnitOfWork
from app.services.ingestion.pipeline import IngestionPipeline
from app.services.jobs.ingestion_queue import IngestionQueue
from app.services.jobs.ingestion_runner import IngestionRunner
from app.use_cases.ingestion import IngestDocumentUseCase
from app.infrastructure.rag import build_chunker, build_dense_embedder, build_sparse_embedder




@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncExitStack() as stack:
        postgres_engine = build_postgres_engine()
        db_sessionmaker = build_sessionmaker(postgres_engine)
        stack.push_async_callback(postgres_engine.dispose)

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
            collection_name=getattr(settings, "RAG_COLLECTION_NAME", "finance_documents"),
        )

        duckdb_manager = DuckDBManager(
            path=getattr(settings, "RAG_DUCKDB_PATH", "/data/finance.duckdb")
        )
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

        ingestion_queue = IngestionQueue(
            maxsize=int(getattr(settings, "RAG_INGESTION_QUEUE_MAXSIZE", 0))
        )

        ingest_use_case = IngestDocumentUseCase(
            uow_factory=lambda: SqlAlchemyUnitOfWork(db_sessionmaker),
            s3_repository=s3_repo,
            pipeline=pipeline,
        )

        ingestion_runner = IngestionRunner(
            queue=ingestion_queue,
            ingest_use_case=ingest_use_case,
        )

        ingestion_runner_task = asyncio.create_task(
            ingestion_runner.run_forever()
        )

        async def shutdown_ingestion_runner() -> None:
            ingestion_runner.stop()
            ingestion_runner_task.cancel()

            with suppress(asyncio.CancelledError):
                await ingestion_runner_task

        stack.push_async_callback(shutdown_ingestion_runner)

        app.state.db_sessionmaker = db_sessionmaker
        app.state.s3_repo = s3_repo
        app.state.qdrant_client = qdrant_client
        app.state.vector_repo = vector_repo
        app.state.duckdb_repo = duckdb_repo
        app.state.ingestion_queue = ingestion_queue
        app.state.ingestion_runner = ingestion_runner
        app.state.ingestion_pipeline = pipeline

        yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["health"])
async def health_check():
    return {"status": "ok"}