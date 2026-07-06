from pydantic import BaseModel, EmailStr, Field, ConfigDict
from typing import Optional, List, Any
from datetime import datetime
from uuid import UUID
from app.models.domain import WorkspaceType, UserRole, DocumentStatus


# TOKEN SCHEMAS
class Token(BaseModel):
    """Schema for the JWT access token response."""
    access_token: str
    token_type: str


class TokenPayload(BaseModel):
    """Schema for the decoded JWT payload data."""
    sub: Optional[str] = None
    workspace_id: Optional[UUID] = None

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8)
    workspace_name: str | None = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str

# WORKSPACE SCHEMAS
class WorkspaceCreate(BaseModel):
    """Schema for creating a new workspace."""
    name: str = Field(..., examples=["Alpha Investment Fund"])
    ws_type: WorkspaceType = WorkspaceType.PRIVATE


class WorkspaceResponse(BaseModel):
    """Schema for returning workspace data."""
    id: UUID
    name: str
    type: WorkspaceType
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# USER SCHEMAS
class UserCreate(BaseModel):
    """Schema for user registration."""
    email: EmailStr
    password: str = Field(..., min_length=8, examples=["strong_password_123"])
    workspace_id: UUID


class UserResponse(BaseModel):
    """Schema for returning user data (password is intentionally omitted)."""
    id: UUID
    email: EmailStr
    role: UserRole
    workspace_id: UUID
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# DOCUMENT SCHEMAS
class DocumentResponse(BaseModel):
    """Schema for returning document processing status."""
    id: UUID
    workspace_id: UUID
    filename: str
    status: DocumentStatus
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


# RAG SCHEMAS
class Citation(BaseModel):
    """Schema for a single citation grounding the LLM answer."""
    doc_name: str
    section: str
    text: str


class QueryRequest(BaseModel):
    """Schema for the user's chat message to the RAG system."""
    query: str = Field(..., examples=["What was the net income for 3M in Q3 2023?"])


class QueryResponse(BaseModel):
    """Schema for the RAG system's answer, including citations."""
    answer: str
    citations: List[Citation] = []


class AgentToolTraceResponse(BaseModel):
    name: str
    arguments: dict[str, Any]
    result_preview: str
    ok: bool = True


class RagAskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    document_ids: list[UUID] | None = None


class RagAskResponse(BaseModel):
    answer: str
    tool_calls: list[AgentToolTraceResponse] = []


class SearchTextRequest(BaseModel):
    query: str = Field(..., min_length=1)
    document_ids: list[UUID] | None = None
    top_k: int = Field(default=8, ge=1, le=30)
    expand_neighbors: bool = True
    neighbor_window: int = Field(default=1, ge=0, le=3)
    query_rewrites: list[str] | None = None


class SearchTextResponse(BaseModel):
    text: str


class SearchTablesRequest(BaseModel):
    concept: str = Field(..., min_length=1)
    document_ids: list[UUID] | None = None
    top_k: int = Field(default=8, ge=1, le=30)
    concept_rewrites: list[str] | None = None


class SearchTablesResponse(BaseModel):
    text: str


class AgentStreamEvent(BaseModel):
    event: str
    data: dict[str, Any]