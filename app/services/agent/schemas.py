from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID


@dataclass(frozen=True)
class AgentRequestContext:
    workspace_id: UUID
    requested_document_ids: list[UUID | str] | None
    allowed_document_ids: list[str]


@dataclass
class AgentToolTrace:
    name: str
    arguments: dict[str, Any]
    result_preview: str
    ok: bool = True


@dataclass
class AgentAnswer:
    answer: str
    tool_calls: list[AgentToolTrace] = field(default_factory=list)