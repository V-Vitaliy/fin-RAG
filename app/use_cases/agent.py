from __future__ import annotations

import logging
from typing import Any
from uuid import UUID
import asyncio
from collections.abc import AsyncIterator
from collections import defaultdict

from app.services.agent.runner import AgentRunner
from app.services.agent.schemas import AgentAnswer
from app.services.agent.tools import build_agent_tools
from app.use_cases.retrieval import RetrieveDocumentsUseCase
from app.services.agent.citations import AgentCitationRegistry, AgentSourceDocument

logger = logging.getLogger(__name__)


NO_ACCESSIBLE_DOCUMENTS_MESSAGE = "No accessible ready documents found."


class AskAgentUseCase:
    def __init__(
        self,
        *,
        openai_client: Any,
        retrieval_use_case: RetrieveDocumentsUseCase,
        reasoning_effort: str = "low",
        citation_url_expires_seconds: int = 600,
        duckdb_path: str,
        model_name: str,
        max_tool_turns: int = 15,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        trace_path: str | None = None,
        max_tool_output_chars: int = 12000,
        retrieval_top_k: int = 8,
        uow_factory: Any,
        s3_repository: Any,
    ):
        self.openai_client = openai_client
        self.retrieval_use_case = retrieval_use_case
        self.citation_url_expires_seconds = int(citation_url_expires_seconds)
        self.duckdb_path = str(duckdb_path)
        self.model_name = str(model_name)
        self.max_tool_turns = int(max_tool_turns)
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.trace_path = trace_path
        self.max_tool_output_chars = int(max_tool_output_chars)
        self.retrieval_top_k = int(retrieval_top_k)
        self.uow_factory = uow_factory
        self.s3_repository = s3_repository
        self.reasoning_effort = str(reasoning_effort or "low")

    @staticmethod
    def _tool_calls_payload(result: AgentAnswer) -> list[dict[str, Any]]:
        return [
            {
                "name": call.name,
                "arguments": call.arguments,
                "result_preview": call.result_preview,
                "ok": call.ok,
            }
            for call in result.tool_calls
        ]

    async def _attach_citation_links(
            self,
            *,
            result: AgentAnswer,
            citation_registry: AgentCitationRegistry,
            allowed_document_ids: list[str],
    ) -> AgentAnswer:
        citations = citation_registry.citations_for_answer(result.answer)
        result.citations = [citation.to_dict() for citation in citations]

        if not citations:
            result.source_documents = []
            return result

        allowed = {str(document_id) for document_id in allowed_document_ids}
        grouped: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"markers": set(), "pages": set(), "filename": None}
        )

        for citation in citations:
            document_id = str(citation.document_id)
            if document_id not in allowed:
                continue

            group = grouped[document_id]
            group["markers"].add(citation.marker)

            if citation.page_number is not None:
                group["pages"].add(int(citation.page_number))

            if citation.filename:
                group["filename"] = citation.filename

        source_documents: list[dict[str, Any]] = []

        async with self.uow_factory() as uow:
            for document_id, group in grouped.items():
                try:
                    db_document = await uow.documents.get_document(UUID(document_id))
                except Exception:
                    db_document = None

                if not db_document or str(db_document.id) not in allowed:
                    continue

                object_key = getattr(db_document, "s3_object_key", None)
                if not object_key:
                    continue

                url = await self.s3_repository.create_presigned_get_url(
                    object_key,
                    expires_in_seconds=self.citation_url_expires_seconds,
                )

                filename = str(
                    getattr(db_document, "filename", None)
                    or group.get("filename")
                    or document_id
                )

                source_documents.append(
                    AgentSourceDocument(
                        document_id=document_id,
                        filename=filename,
                        url=url,
                        markers=sorted(group["markers"]),
                        pages=sorted(group["pages"]),
                        expires_in_seconds=self.citation_url_expires_seconds,
                    ).to_dict()
                )

        result.source_documents = source_documents
        return result


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

        citation_registry = AgentCitationRegistry()

        tools_registry = build_agent_tools(
            retrieval_use_case=self.retrieval_use_case,
            workspace_id=workspace_id,
            requested_document_ids=requested_document_ids,
            allowed_document_ids=access.document_ids,
            duckdb_path=self.duckdb_path,
            retrieval_top_k=self.retrieval_top_k,
            citation_registry=citation_registry,
        )

        runner = AgentRunner(
            openai_client=self.openai_client,
            tools_registry=tools_registry,
            reasoning_effort=self.reasoning_effort,
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

        result = await runner.generate_answer(question=question)

        return await self._attach_citation_links(
            result=result,
            citation_registry=citation_registry,
            allowed_document_ids=access.document_ids,
        )

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
            yield {
                "event": "done",
                "data": {
                    "answer": "",
                    "tool_calls": [],
                    "citations": [],
                    "source_documents": [],
                },
            }
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
            yield {
                "event": "done",
                "data": {
                    "answer": "",
                    "tool_calls": [],
                    "citations": [],
                    "source_documents": [],
                },
            }
            return

        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        async def event_sink(event: str, data: dict[str, Any]) -> None:
            await queue.put({"event": event, "data": data})

        citation_registry = AgentCitationRegistry()

        tools_registry = build_agent_tools(
            retrieval_use_case=self.retrieval_use_case,
            workspace_id=workspace_id,
            requested_document_ids=requested_document_ids,
            allowed_document_ids=access.document_ids,
            duckdb_path=self.duckdb_path,
            retrieval_top_k=self.retrieval_top_k,
            citation_registry=citation_registry,
        )

        runner = AgentRunner(
            openai_client=self.openai_client,
            tools_registry=tools_registry,
            model_name=self.model_name,
            reasoning_effort=self.reasoning_effort,
            max_tool_turns=self.max_tool_turns,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            trace_path=self.trace_path,
            max_tool_output_chars=self.max_tool_output_chars,
            event_sink=event_sink,
        )

        async def run_agent() -> AgentAnswer:
            return await runner.generate_answer(question=question)

        task = asyncio.create_task(run_agent())

        try:
            while not task.done() or not queue.empty():
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=0.1)
                    yield item
                except asyncio.TimeoutError:
                    continue

            try:
                result = await task

            except Exception as exc:
                logger.exception("RAG agent stream failed")

                yield {
                    "event": "error",
                    "data": {
                        "message": "Agent failed while generating the answer.",
                        "detail": str(exc),
                    },
                }
                yield {
                    "event": "done",
                    "data": {
                        "answer": "",
                        "tool_calls": [],
                        "citations": [],
                        "source_documents": [],
                    },
                }
                return

            yield {
                "event": "stage",
                "data": {
                    "stage": "finalizing",
                    "message": "Preparing final answer and source links",
                },
            }

            try:
                result = await self._attach_citation_links(
                    result=result,
                    citation_registry=citation_registry,
                    allowed_document_ids=access.document_ids,
                )

            except Exception as exc:
                logger.exception("Failed to attach citation links")

                # Do not lose the already generated answer because link generation failed.
                result.citations = []
                result.source_documents = []

                yield {
                    "event": "error",
                    "data": {
                        "message": "Answer was generated, but citation links could not be prepared.",
                        "detail": str(exc),
                    },
                }

            final_payload = {
                "answer": result.answer,
                "citations": result.citations,
                "source_documents": result.source_documents,
            }

            yield {
                "event": "final_answer",
                "data": final_payload,
            }

            yield {
                "event": "done",
                "data": {
                    **final_payload,
                    "tool_calls": self._tool_calls_payload(result),
                },
            }

        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    logger.exception("Agent task failed while being cancelled")