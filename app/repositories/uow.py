from types import TracebackType
from typing import Optional, Type

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.repositories.pg_repo import (
    DocumentRepository,
    UserRepository,
    WorkspaceRepository,
)


class SqlAlchemyUnitOfWork:
    def __init__(self, sessionmaker: async_sessionmaker[AsyncSession]):
        self._sessionmaker = sessionmaker
        self.session: AsyncSession | None = None
        self.workspaces: WorkspaceRepository | None = None
        self.users: UserRepository | None = None
        self.documents: DocumentRepository | None = None

    async def __aenter__(self) -> "SqlAlchemyUnitOfWork":
        self.session = self._sessionmaker()

        self.workspaces = WorkspaceRepository(self.session)
        self.users = UserRepository(self.session)
        self.documents = DocumentRepository(self.session)

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self.session is None:
            return

        try:
            if exc_type is None:
                await self.session.commit()
            else:
                await self.session.rollback()
        finally:
            await self.session.close()
            self.session = None
            self.workspaces = None
            self.users = None
            self.documents = None