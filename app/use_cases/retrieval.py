from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from app.services.retrieval.hybrid import AgentHybridRetriever
from app.services.agent.citations import AgentCitationRegistry


NO_ACCESSIBLE_DOCUMENTS_MESSAGE = "No accessible ready documents found."


@dataclass(frozen=True)
class RetrievalDocumentAccess:
    document_ids: list[str]


class RetrieveDocumentsUseCase:
    def __init__(
        self,
        *,
        uow_factory,
        retriever: AgentHybridRetriever,
        global_workspace_id: UUID | str | None = None,
    ):
        self.uow_factory = uow_factory
        self.retriever = retriever
        self.global_workspace_id = global_workspace_id

    @staticmethod
    def _normalize_requested_document_ids(
        requested_document_ids: list[UUID | str] | None,
    ) -> list[UUID] | None:
        if not requested_document_ids:
            return None

        normalized: list[UUID] = []
        for document_id in requested_document_ids:
            normalized.append(
                document_id if isinstance(document_id, UUID) else UUID(str(document_id))
            )

        return normalized

    @staticmethod
    def _document_id(document: Any) -> str:
        return str(getattr(document, "id"))

    async def _resolve_access(
        self,
        *,
        workspace_id: UUID,
        requested_document_ids: list[UUID | str] | None = None,
    ) -> RetrievalDocumentAccess:
        normalized_requested_ids = self._normalize_requested_document_ids(
            requested_document_ids
        )

        async with self.uow_factory() as uow:
            documents = await uow.documents.get_ready_accessible_documents(
                workspace_id=workspace_id,
                global_workspace_id=self.global_workspace_id,
                requested_document_ids=normalized_requested_ids,
            )

        return RetrievalDocumentAccess(
            document_ids=[self._document_id(document) for document in documents],
        )

    async def resolve_access(
        self,
        *,
        workspace_id: UUID,
        requested_document_ids: list[UUID | str] | None = None,
    ) -> RetrievalDocumentAccess:
        return await self._resolve_access(
            workspace_id=workspace_id,
            requested_document_ids=requested_document_ids,
        )

    async def search(
        self,
        *,
        workspace_id: UUID,
        query: str,
        requested_document_ids: list[UUID | str] | None = None,
        top_k: int = 8,
        only_table_stubs: bool = False,
        extra_queries: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        access = await self._resolve_access(
            workspace_id=workspace_id,
            requested_document_ids=requested_document_ids,
        )

        if not access.document_ids:
            return []

        return await self.retriever.search(
            query=query,
            allowed_document_ids=access.document_ids,
            top_k=top_k,
            only_table_stubs=only_table_stubs,
            extra_queries=extra_queries,
        )

    async def search_text(
        self,
        *,
        workspace_id: UUID,
        query: str,
        requested_document_ids: list[UUID | str] | None = None,
        top_k: int = 8,
        expand_neighbors: bool = False,
        neighbor_window: int = 1,
        extra_queries: list[str] | None = None,
        citation_registry: AgentCitationRegistry | None = None,
    ) -> str:
        access = await self._resolve_access(
            workspace_id=workspace_id,
            requested_document_ids=requested_document_ids,
        )

        if not access.document_ids:
            return NO_ACCESSIBLE_DOCUMENTS_MESSAGE

        return await self.retriever.search_text(
            query=query,
            allowed_document_ids=access.document_ids,
            top_k=top_k,
            expand_neighbors=expand_neighbors,
            neighbor_window=neighbor_window,
            extra_queries=extra_queries,
            citation_registry=citation_registry,
        )

    async def search_tables(
        self,
        *,
        workspace_id: UUID,
        concept: str,
        requested_document_ids: list[UUID | str] | None = None,
        top_k: int = 8,
        extra_queries: list[str] | None = None,
        citation_registry: AgentCitationRegistry | None = None,
    ) -> str:
        access = await self._resolve_access(
            workspace_id=workspace_id,
            requested_document_ids=requested_document_ids,
        )

        if not access.document_ids:
            return NO_ACCESSIBLE_DOCUMENTS_MESSAGE

        return await self.retriever.search_tables(
            concept=concept,
            allowed_document_ids=access.document_ids,
            top_k=top_k,
            extra_queries=extra_queries,
            citation_registry=citation_registry,
        )