import enum
import uuid
from datetime import datetime
from sqlalchemy import Column, String, Enum, ForeignKey, DateTime, Uuid
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


# ENUMS
class WorkspaceType(str, enum.Enum):
    """Defines the visibility and type of the workspace."""
    PRIVATE = "PRIVATE"
    GLOBAL = "GLOBAL"


class UserRole(str, enum.Enum):
    """Defines the access level of a user."""
    ADMIN = "ADMIN"
    VIEWER = "VIEWER"


class DocumentStatus(str, enum.Enum):
    """Tracks the async processing status of a document."""
    UPLOADING = "UPLOADING"
    PROCESSING = "PROCESSING"
    READY = "READY"
    FAILED = "FAILED"


# MODELS
class Workspace(Base):
    __tablename__ = "workspaces"

    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    type = Column(Enum(WorkspaceType), default=WorkspaceType.PRIVATE, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    users = relationship("User", back_populates="workspace", cascade="all, delete-orphan")
    documents = relationship("Document", back_populates="workspace", cascade="all, delete-orphan")


class User(Base):
    __tablename__ = "users"

    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    workspace_id = Column(Uuid, ForeignKey("workspaces.id"), nullable=False)

    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(Enum(UserRole), default=UserRole.VIEWER, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    workspace = relationship("Workspace", back_populates="users")


class Document(Base):
    __tablename__ = "documents"

    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    workspace_id = Column(Uuid, ForeignKey("workspaces.id"), nullable=False)

    filename = Column(String(255), nullable=False)
    s3_object_key = Column(String(1024), nullable=False)
    status = Column(Enum(DocumentStatus), default=DocumentStatus.UPLOADING, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    workspace = relationship("Workspace", back_populates="documents")