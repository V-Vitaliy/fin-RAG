from __future__ import annotations

import logging
from typing import Any
from uuid import UUID
import asyncio
from collections.abc import AsyncIterator

from app.services.agent.runner import AgentRunner
from app.services.agent.schemas import AgentAnswer
from app.services.agent.tools import build_agent_tools
from app.use_cases.retrieval import RetrieveDocumentsUseCase

logger = logging.getLogger(__name__)


NO_ACCESSIBLE_DOCUMENTS_MESSAGE = "No accessible ready documents found."


class AskAgentUseCase:
    def __init__(
        self,
        *,
        openai_client: Any,
        retrieval_use_case: RetrieveDocumentsUseCase,
        duckdb_path: str,
        model_name: str,
        max_tool_turns: int = 15,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        trace_path: str | None = None,
        max_tool_output_chars: int = 12000,
        retrieval_top_k: int = 8,
    ):
        self.openai_client = openai_client
        self.retrieval_use_case = retrieval_use_case
        self.duckdb_path = str(duckdb_path)
        self.model_name = str(model_name)
        self.max_tool_turns = int(max_tool_turns)
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.trace_path = trace_path
        self.max_tool_output_chars = int(max_tool_output_chars)
        self.retrieval_top_k = int(retrieval_top_k)

    async def ask(
        self,
        *,
        workspace_id: UUID,
        question: str,
        requested_document_ids: list[UUID | str] | None = None,
    ) -> AgentAnswer:
        question = str(question or "").strip()

        if not question:
            return AgentAnswer(
                answer="Question is empty.",
                tool_calls=[],
            )

        access = await self.retrieval_use_case.resolve_access(
            workspace_id=workspace_id,
            requested_document_ids=requested_document_ids,
        )

        if not access.document_ids:
            return AgentAnswer(
                answer=NO_ACCESSIBLE_DOCUMENTS_MESSAGE,
                tool_calls=[],
            )

        tools_registry = build_agent_tools(
            retrieval_use_case=self.retrieval_use_case,
            workspace_id=workspace_id,
            requested_document_ids=requested_document_ids,
            allowed_document_ids=access.document_ids,
            duckdb_path=self.duckdb_path,
            retrieval_top_k=self.retrieval_top_k,
        )

        runner = AgentRunner(
            openai_client=self.openai_client,
            tools_registry=tools_registry,
            model_name=self.model_name,
            max_tool_turns=self.max_tool_turns,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            trace_path=self.trace_path,
            max_tool_output_chars=self.max_tool_output_chars,
        )

        logger.info(
            "Running RAG agent question=%r workspace_id=%s requested_document_ids=%s allowed_document_count=%s",
            question,
            workspace_id,
            requested_document_ids,
            len(access.document_ids),
        )

        return await runner.generate_answer(question=question)

    @staticmethod
    def _answer_chunks(answer: str, chunk_size: int = 24) -> list[str]:
        words = str(answer or "").split(" ")
        chunks: list[str] = []
        current: list[str] = []

        for word in words:
            current.append(word)
            if len(" ".join(current)) >= chunk_size:
                chunks.append(" ".join(current) + " ")
                current = []

        if current:
            chunks.append(" ".join(current))

        return chunks

    async def ask_stream(
        self,
        *,
        workspace_id: UUID,
        question: str,
        requested_document_ids: list[UUID | str] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        question = str(question or "").strip()

        if not question:
            yield {
                "event": "error",
                "data": {"message": "Question is empty."},
            }
            yield {"event": "done", "data": {}}
            return

        yield {
            "event": "stage",
            "data": {
                "stage": "resolving_access",
                "message": "Checking accessible documents",
            },
        }

        access = await self.retrieval_use_case.resolve_access(
            workspace_id=workspace_id,
            requested_document_ids=requested_document_ids,
        )

        if not access.document_ids:
            yield {
                "event": "error",
                "data": {"message": NO_ACCESSIBLE_DOCUMENTS_MESSAGE},
            }
            yield {"event": "done", "data": {}}
            return

        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        async def event_sink(event: str, data: dict[str, Any]) -> None:
            await queue.put({"event": event, "data": data})

        tools_registry = build_agent_tools(
            retrieval_use_case=self.retrieval_use_case,
            workspace_id=workspace_id,
            requested_document_ids=requested_document_ids,
            allowed_document_ids=access.document_ids,
            duckdb_path=self.duckdb_path,
            retrieval_top_k=self.retrieval_top_k,
        )

        runner = AgentRunner(
            openai_client=self.openai_client,
            tools_registry=tools_registry,
            model_name=self.model_name,
            max_tool_turns=self.max_tool_turns,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            trace_path=self.trace_path,
            max_tool_output_chars=self.max_tool_output_chars,
            event_sink=event_sink,
        )

        async def run_agent():
            return await runner.generate_answer(question=question)

        task = asyncio.create_task(run_agent())

        while not task.done() or not queue.empty():
            try:
                item = await asyncio.wait_for(queue.get(), timeout=0.1)
                yield item
            except asyncio.TimeoutError:
                continue

        result = await task

        yield {
            "event": "stage",
            "data": {
                "stage": "answer_stream",
                "message": "Streaming final answer",
            },
        }

        for chunk in self._answer_chunks(result.answer):
            yield {
                "event": "token",
                "data": {"text": chunk},
            }

        yield {
            "event": "done",
            "data": {
                "answer": result.answer,
                "tool_calls": [
                    {
                        "name": call.name,
                        "arguments": call.arguments,
                        "result_preview": call.result_preview,
                        "ok": call.ok,
                    }
                    for call in result.tool_calls
                ],
            },
        }