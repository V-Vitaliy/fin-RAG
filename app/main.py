from fastapi import FastAPI
from contextlib import asynccontextmanager, AsyncExitStack
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.infrastructure.qdrant import build_qdrant_client
from app.infrastructure.s3storage import s3session
from app.repositories.s3storage_repository import S3StorageRepository
from app.infrastructure.postgres import build_postgres_engine, build_sessionmaker


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

        app.state.db_sessionmaker = db_sessionmaker
        app.state.s3_repo = s3_repo
        app.state.qdrant_client = qdrant_client

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
