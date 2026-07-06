from __future__ import annotations

from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.dependencies import (
    get_agent_use_case,
    get_retrieval_use_case,
    get_workspace_id,
)
from app.api.routes.router import api_router
from app.services.agent.schemas import AgentAnswer, AgentToolTrace


class FakeAgentUseCase:
    def __init__(self):
        self.calls = []

    async def ask(self, *, workspace_id, question, requested_document_ids=None):
        self.calls.append(
            {
                "workspace_id": workspace_id,
                "question": question,
                "requested_document_ids": requested_document_ids,
            }
        )

        return AgentAnswer(
            answer="Fake grounded answer [C1].",
            tool_calls=[
                AgentToolTrace(
                    name="search_text",
                    arguments={"query": "fake"},
                    result_preview="[C1] fake evidence",
                    ok=True,
                )
            ],
        )

    async def ask_stream(self, *, workspace_id, question, requested_document_ids=None):
        self.calls.append(
            {
                "workspace_id": workspace_id,
                "question": question,
                "requested_document_ids": requested_document_ids,
                "stream": True,
            }
        )

        yield {
            "event": "stage",
            "data": {"stage": "planning", "message": "Planning search strategy"},
        }
        yield {
            "event": "tool_call",
            "data": {"name": "search_text", "arguments": {"query": "fake"}},
        }
        yield {
            "event": "tool_result",
            "data": {"name": "search_text", "ok": True, "preview": "[C1] fake"},
        }
        yield {
            "event": "token",
            "data": {"text": "Fake grounded answer "},
        }
        yield {
            "event": "token",
            "data": {"text": "[C1]."},
        }
        yield {
            "event": "done",
            "data": {
                "answer": "Fake grounded answer [C1].",
                "tool_calls": [],
            },
        }


class FakeRetrievalUseCase:
    def __init__(self):
        self.text_calls = []
        self.table_calls = []

    async def search_text(
        self,
        *,
        workspace_id,
        query,
        requested_document_ids=None,
        top_k=8,
        expand_neighbors=False,
        neighbor_window=1,
        extra_queries=None,
    ):
        self.text_calls.append(
            {
                "workspace_id": workspace_id,
                "query": query,
                "requested_document_ids": requested_document_ids,
                "top_k": top_k,
                "expand_neighbors": expand_neighbors,
                "neighbor_window": neighbor_window,
                "extra_queries": extra_queries,
            }
        )
        return "[C1] fake text evidence"

    async def search_tables(
        self,
        *,
        workspace_id,
        concept,
        requested_document_ids=None,
        top_k=8,
        extra_queries=None,
    ):
        self.table_calls.append(
            {
                "workspace_id": workspace_id,
                "concept": concept,
                "requested_document_ids": requested_document_ids,
                "top_k": top_k,
                "extra_queries": extra_queries,
            }
        )
        return "[T1] fake table evidence"


def build_client():
    app = FastAPI()
    app.include_router(api_router)

    workspace_id = uuid4()
    fake_agent = FakeAgentUseCase()
    fake_retrieval = FakeRetrievalUseCase()

    async def override_workspace_id():
        return workspace_id

    def override_agent_use_case():
        return fake_agent

    def override_retrieval_use_case():
        return fake_retrieval

    app.dependency_overrides[get_workspace_id] = override_workspace_id
    app.dependency_overrides[get_agent_use_case] = override_agent_use_case
    app.dependency_overrides[get_retrieval_use_case] = override_retrieval_use_case

    return TestClient(app), workspace_id, fake_agent, fake_retrieval


def test_rag_ask_route_calls_agent_use_case():
    client, workspace_id, fake_agent, _fake_retrieval = build_client()

    response = client.post(
        "/api/v1/rag/ask",
        json={"question": "What did the company announce?"},
    )

    assert response.status_code == 200
    body = response.json()

    assert body["answer"] == "Fake grounded answer [C1]."
    assert body["tool_calls"][0]["name"] == "search_text"

    assert fake_agent.calls[0]["workspace_id"] == workspace_id
    assert fake_agent.calls[0]["question"] == "What did the company announce?"


def test_rag_ask_stream_route_emits_sse_events():
    client, _workspace_id, fake_agent, _fake_retrieval = build_client()

    with client.stream(
        "POST",
        "/api/v1/rag/ask/stream",
        json={"question": "What did the company announce?"},
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        raw = "".join(response.iter_text())

    assert "event: stage" in raw
    assert "event: tool_call" in raw
    assert "event: tool_result" in raw
    assert "event: token" in raw
    assert "event: done" in raw
    assert "Fake grounded answer" in raw

    assert fake_agent.calls[0]["stream"] is True


def test_rag_search_text_route_calls_retrieval_use_case():
    client, workspace_id, _fake_agent, fake_retrieval = build_client()
    document_id = uuid4()

    response = client.post(
        "/api/v1/rag/search/text",
        json={
            "query": "announcement",
            "document_ids": [str(document_id)],
            "top_k": 5,
            "expand_neighbors": True,
            "neighbor_window": 1,
            "query_rewrites": ["company announcement"],
        },
    )

    assert response.status_code == 200
    assert response.json() == {"text": "[C1] fake text evidence"}

    call = fake_retrieval.text_calls[0]
    assert call["workspace_id"] == workspace_id
    assert call["query"] == "announcement"
    assert call["requested_document_ids"] == [document_id]
    assert call["top_k"] == 5
    assert call["expand_neighbors"] is True
    assert call["extra_queries"] == ["company announcement"]


def test_rag_search_tables_route_calls_retrieval_use_case():
    client, workspace_id, _fake_agent, fake_retrieval = build_client()

    response = client.post(
        "/api/v1/rag/search/tables",
        json={
            "concept": "income statement",
            "top_k": 3,
            "concept_rewrites": ["statement of operations"],
        },
    )

    assert response.status_code == 200
    assert response.json() == {"text": "[T1] fake table evidence"}

    call = fake_retrieval.table_calls[0]
    assert call["workspace_id"] == workspace_id
    assert call["concept"] == "income statement"
    assert call["top_k"] == 3
    assert call["extra_queries"] == ["statement of operations"]