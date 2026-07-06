from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

import pytest

from app.use_cases.retrieval import (
    NO_ACCESSIBLE_DOCUMENTS_MESSAGE,
    RetrieveDocumentsUseCase,
)


@dataclass
class FakeDocument:
    id: UUID
    filename: str


class FakeDocumentRepository:
    def __init__(self, documents: list[FakeDocument]):
        self.documents = documents
        self.calls = []

    async def get_ready_accessible_documents(self, **kwargs):
        self.calls.append(kwargs)

        requested_document_ids = kwargs.get("requested_document_ids")
        if requested_document_ids:
            requested_set = {str(document_id) for document_id in requested_document_ids}
            return [
                document
                for document in self.documents
                if str(document.id) in requested_set
            ]

        return self.documents


class FakeUnitOfWork:
    def __init__(self, documents: list[FakeDocument]):
        self.documents = FakeDocumentRepository(documents)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class FakeUnitOfWorkFactory:
    def __init__(self, documents: list[FakeDocument]):
        self.uow = FakeUnitOfWork(documents)

    def __call__(self):
        return self.uow


class FakeRetriever:
    def __init__(self):
        self.search_calls = []
        self.search_text_calls = []
        self.search_tables_calls = []

    async def search(self, **kwargs) -> list[dict[str, Any]]:
        self.search_calls.append(kwargs)
        return [{"document_id": kwargs["allowed_document_ids"][0], "text": "hit"}]

    async def search_text(self, **kwargs) -> str:
        self.search_text_calls.append(kwargs)
        return "text results"

    async def search_tables(self, **kwargs) -> str:
        self.search_tables_calls.append(kwargs)
        return "table results"


def make_use_case(
    *,
    documents: list[FakeDocument],
    retriever: FakeRetriever | None = None,
    global_workspace_id: UUID | None = None,
):
    retriever = retriever or FakeRetriever()
    use_case = RetrieveDocumentsUseCase(
        uow_factory=FakeUnitOfWorkFactory(documents),
        retriever=retriever,
        global_workspace_id=global_workspace_id,
    )
    return use_case, retriever


@pytest.mark.asyncio
async def test_search_text_uses_ready_accessible_document_ids():
    workspace_id = uuid4()
    global_workspace_id = uuid4()
    document = FakeDocument(id=uuid4(), filename="PEPSICO_2023_8K.pdf")

    use_case, retriever = make_use_case(
        documents=[document],
        global_workspace_id=global_workspace_id,
    )

    result = await use_case.search_text(
        workspace_id=workspace_id,
        query="revenue",
        top_k=5,
    )

    assert result == "text results"

    uow_call = use_case.uow_factory.uow.documents.calls[0]
    assert uow_call["workspace_id"] == workspace_id
    assert uow_call["global_workspace_id"] == global_workspace_id
    assert uow_call["requested_document_ids"] is None

    retriever_call = retriever.search_text_calls[0]
    assert retriever_call["query"] == "revenue"
    assert retriever_call["allowed_document_ids"] == [str(document.id)]
    assert retriever_call["top_k"] == 5
    assert "doc_name" not in retriever_call


@pytest.mark.asyncio
async def test_search_text_passes_requested_document_ids_to_repository():
    workspace_id = uuid4()
    doc_a = FakeDocument(id=uuid4(), filename="A.pdf")
    doc_b = FakeDocument(id=uuid4(), filename="B.pdf")

    use_case, retriever = make_use_case(documents=[doc_a, doc_b])

    result = await use_case.search_text(
        workspace_id=workspace_id,
        query="revenue",
        requested_document_ids=[doc_b.id],
    )

    assert result == "text results"

    uow_call = use_case.uow_factory.uow.documents.calls[0]
    assert uow_call["requested_document_ids"] == [doc_b.id]

    retriever_call = retriever.search_text_calls[0]
    assert retriever_call["allowed_document_ids"] == [str(doc_b.id)]
    assert "doc_name" not in retriever_call


@pytest.mark.asyncio
async def test_search_text_normalizes_string_requested_document_ids():
    workspace_id = uuid4()
    document = FakeDocument(id=uuid4(), filename="PEPSICO_2023_8K.pdf")

    use_case, retriever = make_use_case(documents=[document])

    result = await use_case.search_text(
        workspace_id=workspace_id,
        query="revenue",
        requested_document_ids=[str(document.id)],
    )

    assert result == "text results"

    uow_call = use_case.uow_factory.uow.documents.calls[0]
    assert uow_call["requested_document_ids"] == [document.id]

    retriever_call = retriever.search_text_calls[0]
    assert retriever_call["allowed_document_ids"] == [str(document.id)]
    assert "doc_name" not in retriever_call


@pytest.mark.asyncio
async def test_search_text_returns_message_when_no_accessible_documents():
    workspace_id = uuid4()

    use_case, retriever = make_use_case(documents=[])

    result = await use_case.search_text(
        workspace_id=workspace_id,
        query="revenue",
    )

    assert result == NO_ACCESSIBLE_DOCUMENTS_MESSAGE
    assert retriever.search_text_calls == []


@pytest.mark.asyncio
async def test_search_tables_calls_retriever_with_concept():
    workspace_id = uuid4()
    document = FakeDocument(id=uuid4(), filename="PEPSICO_2023_8K.pdf")

    use_case, retriever = make_use_case(documents=[document])

    result = await use_case.search_tables(
        workspace_id=workspace_id,
        concept="income statement",
        top_k=3,
    )

    assert result == "table results"

    call = retriever.search_tables_calls[0]
    assert call["concept"] == "income statement"
    assert call["allowed_document_ids"] == [str(document.id)]
    assert call["top_k"] == 3
    assert "doc_name" not in call


@pytest.mark.asyncio
async def test_search_returns_structured_results():
    workspace_id = uuid4()
    document = FakeDocument(id=uuid4(), filename="PEPSICO_2023_8K.pdf")

    use_case, retriever = make_use_case(documents=[document])

    result = await use_case.search(
        workspace_id=workspace_id,
        query="revenue",
        top_k=2,
    )

    assert result == [{"document_id": str(document.id), "text": "hit"}]

    call = retriever.search_calls[0]
    assert call["query"] == "revenue"
    assert call["allowed_document_ids"] == [str(document.id)]
    assert call["top_k"] == 2
    assert "doc_name" not in call