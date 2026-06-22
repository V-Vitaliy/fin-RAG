import uuid
from typing import List, Optional
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.domain import User, Workspace, Document, DocumentStatus, WorkspaceType


class WorkspaceRepository:
    """Async repository for workspace management."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_workspace(self, name: str, ws_type: WorkspaceType = WorkspaceType.PRIVATE) -> Workspace:
        """Creates a new workspace."""
        db_workspace = Workspace(name=name, type=ws_type)
        self.db.add(db_workspace)
        await self.db.commit()
        await self.db.refresh(db_workspace)
        return db_workspace

    async def get_workspace(self, workspace_id: uuid.UUID) -> Optional[Workspace]:
        """Retrieves a workspace by its ID."""
        result = await self.db.execute(
            select(Workspace).where(Workspace.id == workspace_id)
        )
        return result.scalars().first()


class UserRepository:
    """Async repository for user management."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Finds a user by email."""
        result = await self.db.execute(
            select(User).where(User.email == email)
        )
        return result.scalars().first()

    async def get_user_by_id(self, user_id: uuid.UUID) -> Optional[User]:
        """Finds a user by ID."""
        result = await self.db.execute(
            select(User).where(User.id == user_id)
        )
        return result.scalars().first()

    async def create_user(self, email: str, password_hash: str, workspace_id: uuid.UUID) -> User:
        """Creates a new user and links them to a workspace."""
        db_user = User(
            email=email,
            password_hash=password_hash,
            workspace_id=workspace_id
        )
        self.db.add(db_user)
        await self.db.commit()
        await self.db.refresh(db_user)
        return db_user


class DocumentRepository:
    """Async repository for document metadata management."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_document(self, workspace_id: uuid.UUID, filename: str) -> Document:
        """Creates a new document record with UPLOADING status."""
        db_doc = Document(
            workspace_id=workspace_id,
            filename=filename,
            status=DocumentStatus.UPLOADING
        )
        self.db.add(db_doc)
        await self.db.commit()
        await self.db.refresh(db_doc)
        return db_doc

    async def update_status(self, document_id: uuid.UUID, status: DocumentStatus) -> Optional[Document]:
        """Updates the processing status of a document."""
        result = await self.db.execute(
            select(Document).where(Document.id == document_id)
        )
        db_doc = result.scalars().first()
        if db_doc:
            db_doc.status = status
            await self.db.commit()
            await self.db.refresh(db_doc)
        return db_doc

    async def get_by_workspace(self, workspace_id: uuid.UUID) -> List[Document]:
        """Retrieves all documents belonging to a specific workspace."""
        result = await self.db.execute(
            select(Document).where(Document.workspace_id == workspace_id)
        )
        return list(result.scalars().all())

    async def delete_document(self, document_id: uuid.UUID) -> Optional[Document]:
        """Deletes a document record from the database."""
        result = await self.db.execute(
            select(Document).where(Document.id == document_id)
        )
        db_doc = result.scalars().first()
        if db_doc:
            await self.db.delete(db_doc)
            await self.db.commit()
            return db_doc
        return None