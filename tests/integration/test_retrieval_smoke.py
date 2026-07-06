from __future__ import annotations

import logging
import os
import re
from textwrap import shorten

import pytest

from app.core.config import settings
from app.infrastructure.qdrant import build_qdrant_client
from app.infrastructure.rag import (
    build_dense_embedder,
    build_reranker,
    build_sparse_embedder,
)
from app.repositories.qdrant_repo import VectorIndexRepository
from app.services.retrieval.hybrid import AgentHybridRetriever

pytestmark = [
    pytest.mark.smoke,
    pytest.mark.slow,
    pytest.mark.asyncio,
]

logger = logging.getLogger(__name__)


def _preview_tool_output(output: str, *, max_blocks: int = 4) -> str:
    blocks = re.split(r"\n(?=\[[CT]\d+\s+\|)", output.strip())
    preview: list[str] = []

    for block in blocks[:max_blocks]:
        lines = block.splitlines()
        if not lines:
            continue

        header = lines[0]
        body = " ".join(lines[1:]).strip()

        preview.append(
            header + "\n" + shorten(body, width=600, placeholder=" ...")
        )

    return "\n\n".join(preview) if preview else shorten(
        output,
        width=1500,
        placeholder=" ...",
    )


def _extract_document_ids_from_output(output: str) -> set[str]:
    return set(re.findall(r"document_id=([0-9a-fA-F-]{36})", output))


@pytest.mark.skipif(
    os.getenv("RUN_RETRIEVAL_SMOKE") != "1",
    reason="Set RUN_RETRIEVAL_SMOKE=1 to run real retrieval smoke test",
)
async def test_agent_hybrid_retriever_smoke_returns_text_and_tables(caplog):
    caplog.set_level(logging.INFO)

    document_id = os.getenv("RETRIEVAL_SMOKE_DOCUMENT_ID")
    assert document_id, "Set RETRIEVAL_SMOKE_DOCUMENT_ID to a READY indexed document id"

    text_query = os.getenv(
        "RETRIEVAL_SMOKE_TEXT_QUERY",
        "What did PepsiCo announce?",
    )
    table_query = os.getenv(
        "RETRIEVAL_SMOKE_TABLE_QUERY",
        "financial table income statement",
    )

    top_k = int(os.getenv("RETRIEVAL_SMOKE_TOP_K", "8"))

    logger.info("Retrieval smoke started")
    logger.info("Collection: %s", settings.RAG_COLLECTION_NAME)
    logger.info("Document ID: %s", document_id)
    logger.info("Text query: %s", text_query)
    logger.info("Table query: %s", table_query)
    logger.info("Top K: %s", top_k)

    logger.info("RAG_USE_OPENAI_EMBEDDINGS=%s", settings.RAG_USE_OPENAI_EMBEDDINGS)
    logger.info("RAG_USE_RERANKER=%s", settings.RAG_USE_RERANKER)
    logger.info("RAG_RERANKER_MODE=%s", settings.RAG_RERANKER_MODE)

    qdrant_client = build_qdrant_client()

    try:
        vector_repo = VectorIndexRepository(
            client=qdrant_client,
            collection_name=settings.RAG_COLLECTION_NAME,
        )
        
        logger.info("Ensuring Qdrant payload indexes...")
        await vector_repo.ensure_payload_indexes()
        logger.info("Qdrant payload indexes are ready")

        point_count = await vector_repo.count_points_by_document_id(document_id)
        logger.info("Qdrant points for document_id=%s: %s", document_id, point_count)
        assert point_count > 0, "Expected indexed Qdrant points for smoke document"

        logger.info("Building dense embedder...")
        dense_embedder = build_dense_embedder()
        logger.info(
            "Dense embedder ready: class=%s vector_size=%s",
            dense_embedder.__class__.__name__,
            getattr(dense_embedder, "vector_size", "unknown"),
        )

        logger.info("Building sparse embedder...")
        sparse_embedder = build_sparse_embedder()
        logger.info("Sparse embedder ready: class=%s", sparse_embedder.__class__.__name__)

        logger.info("Building reranker...")
        reranker = build_reranker()
        logger.info(
            "Reranker ready: %s",
            reranker.__class__.__name__ if reranker else "disabled",
        )

        retriever = AgentHybridRetriever(
            vector_repo=vector_repo,
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
            reranker=reranker,
            prefetch_min=settings.RAG_PREFETCH_MIN,
            fusion_min=settings.RAG_FUSION_MIN,
            first_stage_multiplier=settings.RAG_FIRST_STAGE_MULTIPLIER,
            fusion_multiplier=settings.RAG_FUSION_MULTIPLIER,
        )

        logger.info("Running structured search() for text query...")
        structured_results = await retriever.search(
            query=text_query,
            allowed_document_ids=[document_id],
            top_k=top_k,
            only_table_stubs=False,
        )

        logger.info("Structured text result count: %s", len(structured_results))

        for idx, result in enumerate(structured_results[:top_k], 1):
            logger.info(
                "TEXT_RESULT #%s rank=%s score=%s doc_id=%s doc=%s page=%s "
                "table=%s table_stub=%s section=%s text=%s",
                idx,
                result.get("rank"),
                result.get("score"),
                result.get("document_id"),
                result.get("doc_name"),
                result.get("page_number"),
                result.get("table_name"),
                result.get("is_table_stub"),
                result.get("section_path"),
                shorten(str(result.get("text") or ""), width=450, placeholder=" ..."),
            )

        assert structured_results, "Expected at least one text retrieval result"
        assert all(str(result.get("document_id")) == document_id for result in structured_results)

        logger.info("Running search_text() tool output with neighbor expansion...")
        text_output = await retriever.search_text(
            query=text_query,
            allowed_document_ids=[document_id],
            top_k=top_k,
            expand_neighbors=True,
            neighbor_window=1,
        )

        logger.info("search_text output preview:\n%s", _preview_tool_output(text_output))

        assert "[C" in text_output or "[T" in text_output
        text_doc_ids = _extract_document_ids_from_output(text_output)
        assert not text_doc_ids or text_doc_ids == {document_id}

        logger.info("Running structured search() for table query...")
        structured_tables = await retriever.search(
            query=table_query,
            allowed_document_ids=[document_id],
            top_k=top_k,
            only_table_stubs=True,
        )

        logger.info("Structured table result count: %s", len(structured_tables))

        for idx, result in enumerate(structured_tables[:top_k], 1):
            logger.info(
                "TABLE_RESULT #%s rank=%s score=%s doc_id=%s doc=%s page=%s "
                "table=%s type=%s text=%s",
                idx,
                result.get("rank"),
                result.get("score"),
                result.get("document_id"),
                result.get("doc_name"),
                result.get("page_number"),
                result.get("table_name"),
                result.get("statement_type"),
                shorten(str(result.get("text") or ""), width=450, placeholder=" ..."),
            )

        assert structured_tables, "Expected at least one table retrieval result"
        assert all(str(result.get("document_id")) == document_id for result in structured_tables)
        assert any(result.get("is_table_stub") is True for result in structured_tables)
        assert any(result.get("table_name") for result in structured_tables)

        logger.info("Running search_tables() tool output...")
        table_output = await retriever.search_tables(
            concept=table_query,
            allowed_document_ids=[document_id],
            top_k=top_k,
        )

        logger.info("search_tables output preview:\n%s", _preview_tool_output(table_output))

        assert "[T" in table_output
        assert "table=" in table_output

        table_doc_ids = _extract_document_ids_from_output(table_output)
        assert not table_doc_ids or table_doc_ids == {document_id}

        logger.info("Retrieval smoke finished successfully")

    finally:
        await qdrant_client.close()