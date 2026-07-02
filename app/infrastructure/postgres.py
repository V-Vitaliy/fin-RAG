from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import settings


def build_postgres_engine() -> AsyncEngine:
    if not settings.POSTGRES_URL:
        raise RuntimeError("POSTGRES_URL is not set in the environment variables.")

    return create_async_engine(
        settings.POSTGRES_URL,
        echo=False,
        pool_pre_ping=True,
    )


def build_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        autoflush=False,
        expire_on_commit=False,
    )