from __future__ import annotations

import json
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.dependencies import (
    get_agent_use_case,
    get_retrieval_use_case,
    get_workspace_id,
)
from app.models.schemas import (
    AgentToolTraceResponse,
    RagAskRequest,
    RagAskResponse,
    SearchTablesRequest,
    SearchTablesResponse,
    SearchTextRequest,
    SearchTextResponse,
)
from app.use_cases.agent import AskAgentUseCase
from app.use_cases.retrieval import RetrieveDocumentsUseCase

router = APIRouter(prefix="/rag", tags=["rag"])


def _sse(event: str, data: dict) -> str:
    return (
        f"event: {event}\n"
        f"data: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"
    )


@router.post("/ask", response_model=RagAskResponse)
async def ask_rag(
    payload: RagAskRequest,
    workspace_id: Annotated[UUID, Depends(get_workspace_id)],
    agent_use_case: Annotated[AskAgentUseCase, Depends(get_agent_use_case)],
) -> RagAskResponse:
    result = await agent_use_case.ask(
        workspace_id=workspace_id,
        question=payload.question,
        requested_document_ids=payload.document_ids,
    )

    return RagAskResponse(
        answer=result.answer,
        tool_calls=[
            AgentToolTraceResponse(
                name=call.name,
                arguments=call.arguments,
                result_preview=call.result_preview,
                ok=call.ok,
            )
            for call in result.tool_calls
        ],
        citations=result.citations,
        source_documents=result.source_documents,
    )


@router.post("/ask/stream")
async def ask_rag_stream(
    payload: RagAskRequest,
    workspace_id: Annotated[UUID, Depends(get_workspace_id)],
    agent_use_case: Annotated[AskAgentUseCase, Depends(get_agent_use_case)],
) -> StreamingResponse:
    async def event_generator():
        async for item in agent_use_case.ask_stream(
            workspace_id=workspace_id,
            question=payload.question,
            requested_document_ids=payload.document_ids,
        ):
            yield _sse(item["event"], item.get("data", {}))

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/search/text", response_model=SearchTextResponse)
async def search_text(
    payload: SearchTextRequest,
    workspace_id: Annotated[UUID, Depends(get_workspace_id)],
    retrieval_use_case: Annotated[RetrieveDocumentsUseCase, Depends(get_retrieval_use_case)],
) -> SearchTextResponse:
    text = await retrieval_use_case.search_text(
        workspace_id=workspace_id,
        query=payload.query,
        requested_document_ids=payload.document_ids,
        top_k=payload.top_k,
        expand_neighbors=payload.expand_neighbors,
        neighbor_window=payload.neighbor_window,
        extra_queries=payload.query_rewrites,
    )

    return SearchTextResponse(text=text)


@router.post("/search/tables", response_model=SearchTablesResponse)
async def search_tables(
    payload: SearchTablesRequest,
    workspace_id: Annotated[UUID, Depends(get_workspace_id)],
    retrieval_use_case: Annotated[RetrieveDocumentsUseCase, Depends(get_retrieval_use_case)],
) -> SearchTablesResponse:
    text = await retrieval_use_case.search_tables(
        workspace_id=workspace_id,
        concept=payload.concept,
        requested_document_ids=payload.document_ids,
        top_k=payload.top_k,
        extra_queries=payload.concept_rewrites,
    )

    return SearchTablesResponse(text=text)