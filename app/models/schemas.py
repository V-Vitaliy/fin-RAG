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

class UserResponse(BaseModel):
    """Schema for returning user data (password is intentionally omitted)."""
    id: UUID
    email: EmailStr
    role: UserRole
    workspace_id: UUID
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DocumentResponse(BaseModel):
    """Schema for returning document processing status."""
    id: UUID
    workspace_id: UUID
    filename: str
    status: DocumentStatus
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class AgentCitationResponse(BaseModel):
    marker: str
    document_id: str
    source_type: str
    doc_name: str | None = None
    filename: str | None = None
    page_number: int | None = None
    table_name: str | None = None
    evidence_id: str | None = None
    label: str | None = None


class AgentSourceDocumentResponse(BaseModel):
    document_id: str
    filename: str
    url: str
    markers: list[str] = []
    pages: list[int] = []
    expires_in_seconds: int


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
    citations: list[AgentCitationResponse] = []
    source_documents: list[AgentSourceDocumentResponse] = []


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