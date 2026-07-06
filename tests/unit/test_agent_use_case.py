from __future__ import annotations

from uuid import uuid4

import pytest

from app.services.agent.schemas import AgentAnswer
from app.use_cases.agent import AskAgentUseCase, NO_ACCESSIBLE_DOCUMENTS_MESSAGE


class FakeAccess:
    def __init__(self, document_ids: list[str]):
        self.document_ids = document_ids


class FakeRetrievalUseCase:
    def __init__(self, document_ids: list[str]):
        self.document_ids = document_ids
        self.resolve_calls = []

    async def resolve_access(self, *, workspace_id, requested_document_ids=None):
        self.resolve_calls.append(
            {
                "workspace_id": workspace_id,
                "requested_document_ids": requested_document_ids,
            }
        )
        return FakeAccess(self.document_ids)


class FakeOpenAIClient:
    pass


@pytest.mark.asyncio
async def test_agent_use_case_returns_no_accessible_documents_when_access_empty():
    workspace_id = uuid4()

    use_case = AskAgentUseCase(
        openai_client=FakeOpenAIClient(),
        retrieval_use_case=FakeRetrievalUseCase([]),
        duckdb_path="/tmp/test.duckdb",
        model_name="fake-model",
    )

    result = await use_case.ask(
        workspace_id=workspace_id,
        question="What did the company announce?",
    )

    assert isinstance(result, AgentAnswer)
    assert result.answer == NO_ACCESSIBLE_DOCUMENTS_MESSAGE
    assert result.tool_calls == []


@pytest.mark.asyncio
async def test_agent_use_case_rejects_empty_question_before_running_agent():
    workspace_id = uuid4()

    use_case = AskAgentUseCase(
        openai_client=FakeOpenAIClient(),
        retrieval_use_case=FakeRetrievalUseCase(["doc-1"]),
        duckdb_path="/tmp/test.duckdb",
        model_name="fake-model",
    )

    result = await use_case.ask(
        workspace_id=workspace_id,
        question="   ",
    )

    assert result.answer == "Question is empty."
    assert result.tool_calls == []