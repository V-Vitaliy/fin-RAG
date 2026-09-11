import uuid
from datetime import datetime
from typing import List, Optional, Sequence
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.domain import (
                                User,
                                Workspace,
                                Document,
                                DocumentStatus,
                                WorkspaceType,
                                Company,
                                CompanyEmailDomain,
                            )



class WorkspaceRepository:
    """Repository for workspace management."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_workspace(
            self,
            name: str,
            ws_type: WorkspaceType = WorkspaceType.PRIVATE,
            company_id: uuid.UUID | None = None,
    ) -> Workspace:
        db_workspace = Workspace(
            name=name,
            type=ws_type,
            company_id=company_id,
        )
        self.db.add(db_workspace)
        await self.db.flush()
        await self.db.refresh(db_workspace)
        return db_workspace

    async def get_workspace(self, workspace_id: uuid.UUID) -> Optional[Workspace]:
        """Retrieves a workspace by its ID."""
        result = await self.db.execute(
            select(Workspace).where(Workspace.id == workspace_id)
        )
        return result.scalars().first()

class CompanyRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_workspace_by_email_domain(
        self,
        domain: str,
    ) -> Optional[Workspace]:
        normalized = str(domain or "").strip().lower()

        if not normalized:
            return None

        result = await self.db.execute(
            select(Workspace)
            .join(Company, Workspace.company_id == Company.id)
            .join(CompanyEmailDomain, CompanyEmailDomain.company_id == Company.id)
            .where(CompanyEmailDomain.domain == normalized)
            .order_by(Workspace.created_at.asc())
        )
        return result.scalars().first()

    async def create_company_workspace_for_domain(
        self,
        *,
        company_name: str,
        domain: str,
    ) -> Workspace:
        normalized = str(domain or "").strip().lower()

        if not normalized:
            raise ValueError("Company email domain is required")

        company = Company(name=company_name)
        self.db.add(company)
        await self.db.flush()

        email_domain = CompanyEmailDomain(
            company_id=company.id,
            domain=normalized,
        )
        self.db.add(email_domain)

        workspace = Workspace(
            name=company_name,
            type=WorkspaceType.PRIVATE,
            company_id=company.id,
        )
        self.db.add(workspace)

        await self.db.flush()
        await self.db.refresh(workspace)

        return workspace


class UserRepository:
    """Repository for user management."""

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
            workspace_id=workspace_id,
        )
        self.db.add(db_user)
        await self.db.flush()
        await self.db.refresh(db_user)
        return db_user


class DocumentRepository:
    """Repository for document metadata management."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_document(
            self,
            workspace_id: uuid.UUID,
            filename: str,
            s3_object_key: str,
            document_id: uuid.UUID | None = None,
            content_hash: str | None = None,
            status: DocumentStatus = DocumentStatus.UPLOADING,
            ingestion_version: int = 1,
    ) -> Document:
        """Creates a new document record."""
        db_doc = Document(
            id=document_id or uuid.uuid4(),
            workspace_id=workspace_id,
            filename=filename,
            s3_object_key=s3_object_key,
            status=status,
            content_hash=content_hash,
            is_active=True,
            ingestion_version=ingestion_version,
        )
        self.db.add(db_doc)
        await self.db.flush()
        await self.db.refresh(db_doc)
        return db_doc

    async def get_document(self, document_id: uuid.UUID) -> Optional[Document]:
        """Retrieves a document by ID."""
        result = await self.db.execute(
            select(Document).where(Document.id == document_id)
        )
        return result.scalars().first()

    async def get_active_by_hash(
        self,
        workspace_id: uuid.UUID,
        content_hash: str | None,
    ) -> Optional[Document]:
        """
        Finds an active document with the same content hash in the same workspace.

        Used for upload deduplication.
        """
        if not content_hash:
            return None

        result = await self.db.execute(
            select(Document)
            .where(
                Document.workspace_id == workspace_id,
                Document.content_hash == content_hash,
                Document.is_active.is_(True),
            )
            .order_by(Document.created_at.desc())
        )
        return result.scalars().first()

    async def update_status(
            self,
            document_id: uuid.UUID,
            status: DocumentStatus,
            failure_reason: str | None = None,
    ) -> Optional[Document]:
        """Updates the processing status of a document."""
        result = await self.db.execute(
            select(Document).where(Document.id == document_id)
        )
        db_doc = result.scalars().first()

        if db_doc:
            db_doc.status = status
            if status == DocumentStatus.FAILED:
                db_doc.failure_reason = failure_reason
            elif status in {DocumentStatus.PROCESSING, DocumentStatus.READY}:
                db_doc.failure_reason = None
            await self.db.flush()
            await self.db.refresh(db_doc)

        return db_doc

    async def mark_ready(
        self,
        document_id: uuid.UUID,
        qdrant_points_count: int | None = None,
        duckdb_tables_count: int | None = None,
    ) -> Optional[Document]:
        """Marks document as fully indexed and ready for retrieval."""
        db_doc = await self.get_document(document_id)

        if db_doc:
            db_doc.status = DocumentStatus.READY
            db_doc.indexed_at = datetime.utcnow()
            db_doc.failure_reason = None
            db_doc.qdrant_points_count = qdrant_points_count
            db_doc.duckdb_tables_count = duckdb_tables_count
            db_doc.is_active = True

            await self.db.flush()
            await self.db.refresh(db_doc)

        return db_doc

    async def mark_failed(
        self,
        document_id: uuid.UUID,
        failure_reason: str | None = None,
    ) -> Optional[Document]:
        """Marks document ingestion as failed."""
        db_doc = await self.get_document(document_id)

        if db_doc:
            db_doc.status = DocumentStatus.FAILED
            db_doc.failure_reason = failure_reason
            await self.db.flush()
            await self.db.refresh(db_doc)

        return db_doc

    async def mark_inactive(
        self,
        document_id: uuid.UUID,
    ) -> Optional[Document]:
        """Soft-deactivates a document."""
        db_doc = await self.get_document(document_id)

        if db_doc:
            db_doc.is_active = False
            await self.db.flush()
            await self.db.refresh(db_doc)

        return db_doc

    async def get_by_workspace(self, workspace_id: uuid.UUID) -> List[Document]:
        """Retrieves active documents belonging to a specific workspace."""
        result = await self.db.execute(
            select(Document)
            .where(
                Document.workspace_id == workspace_id,
                Document.is_active.is_(True),
            )
            .order_by(Document.created_at.desc())
        )
        return list(result.scalars().all())

    async def get_ready_accessible_documents(
        self,
        workspace_id: uuid.UUID,
        global_workspace_id: uuid.UUID | None = None,
        requested_document_ids: Sequence[uuid.UUID] | None = None,
    ) -> list[Document]:
        """
        Returns READY active documents accessible by a workspace.

        Access model:
        - own workspace READY docs
        - global workspace READY docs
        - optional narrowing by requested document IDs
        """
        workspace_conditions = [Document.workspace_id == workspace_id]

        if global_workspace_id is not None:
            workspace_conditions.append(Document.workspace_id == global_workspace_id)

        stmt = (
            select(Document)
            .where(
                or_(*workspace_conditions),
                Document.status == DocumentStatus.READY,
                Document.is_active.is_(True),
            )
            .order_by(Document.created_at.desc())
        )

        if requested_document_ids is not None:
            requested_ids = list(requested_document_ids)
            if not requested_ids:
                return []
            stmt = stmt.where(Document.id.in_(requested_ids))

        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def delete_document(self, document_id: uuid.UUID) -> Optional[Document]:
        """Deletes a document record from the database."""
        result = await self.db.execute(
            select(Document).where(Document.id == document_id)
        )
        db_doc = result.scalars().first()

        if db_doc:
            await self.db.delete(db_doc)
            await self.db.flush()
            return db_doc

        return None