from __future__ import annotations

from app.core.config import settings
from app.use_cases.agent import AskAgentUseCase
from app.use_cases.retrieval import RetrieveDocumentsUseCase


def build_agent_use_case(
    *,
    openai_client,
    retrieval_use_case: RetrieveDocumentsUseCase,
) -> AskAgentUseCase:
    return AskAgentUseCase(
        openai_client=openai_client,
        retrieval_use_case=retrieval_use_case,
        duckdb_path=settings.RAG_DUCKDB_PATH,
        model_name=settings.RAG_AGENT_MODEL,
        max_tool_turns=settings.RAG_AGENT_MAX_TOOL_TURNS,
        max_tokens=settings.RAG_AGENT_MAX_TOKENS,
        temperature=settings.RAG_AGENT_TEMPERATURE,
        trace_path=settings.RAG_AGENT_TRACE_PATH,
        max_tool_output_chars=settings.RAG_AGENT_MAX_TOOL_OUTPUT_CHARS,
        retrieval_top_k=settings.RAG_RETRIEVAL_TOP_K,
    )