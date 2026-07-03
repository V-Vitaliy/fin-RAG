from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from app.services.retrieval.hybrid import AgentHybridRetriever


@dataclass
class FakePoint:
    id: str
    score: float
    payload: dict[str, Any]


class FakeSparseVector:
    def __init__(self):
        self.indices = [1, 2, 3]
        self.values = [0.3, 0.2, 0.1]


class FakeDenseEmbedder:
    def embed_query(self, query: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class FakeSparseEmbedder:
    def embed(self, texts: list[str]):
        return [FakeSparseVector()]


class FakeReranker:
    def __init__(self, scores: list[float]):
        self.scores = scores
        self.calls = []

    def predict(self, pairs: list[list[str]]) -> list[float]:
        self.calls.append(pairs)
        return self.scores[: len(pairs)]


class FakeVectorRepo:
    def __init__(
        self,
        points: list[FakePoint] | None = None,
        neighbor_points: list[FakePoint] | None = None,
    ):
        self.points = points or []
        self.neighbor_points = neighbor_points or []
        self.hybrid_calls = []
        self.neighbor_calls = []

    async def hybrid_search(self, **kwargs):
        self.hybrid_calls.append(kwargs)
        return self.points

    async def scroll_text_neighbors(self, **kwargs):
        self.neighbor_calls.append(kwargs)
        return self.neighbor_points


def make_retriever(
    *,
    vector_repo: FakeVectorRepo | None = None,
    reranker=None,
) -> AgentHybridRetriever:
    return AgentHybridRetriever(
        vector_repo=vector_repo or FakeVectorRepo(),
        dense_embedder=FakeDenseEmbedder(),
        sparse_embedder=FakeSparseEmbedder(),
        reranker=reranker,
        prefetch_min=10,
        fusion_min=8,
        first_stage_multiplier=2,
        fusion_multiplier=2,
    )


def test_clean_extra_queries_removes_unsafe_rewrites():
    retriever = make_retriever()

    cleaned = retriever._clean_extra_queries(
        [
            "normal revenue search",
            "use tbl_abc123",
            "page 55 says revenue",
            "p. 10 has the answer",
            "answer: revenue was 123",
            "x" * 181,
        ]
    )

    assert cleaned == ["normal revenue search"]


def test_rewrite_query_adds_financial_synonym_query():
    retriever = make_retriever()

    rewrites = retriever._rewrite_query("What was net income in FY2023?")

    assert "What was net income in FY2023?" in rewrites
    assert any("net earnings profit loss attributable" in item for item in rewrites)


def test_rewrite_query_adds_driver_narrative_query():
    retriever = make_retriever()

    rewrites = retriever._rewrite_query("What drove revenue change in 2023?")

    assert len(rewrites) <= 4
    assert any("management discussion results of operations" in item for item in rewrites)


def test_metadata_boost_prefers_narrative_sections_for_driver_queries():
    retriever = make_retriever()

    mdna_boost = retriever._metadata_boost(
        {
            "is_table_stub": False,
            "source_type": "text",
            "section_path": ["Management Discussion", "Results of Operations"],
        },
        "What drove revenue change?",
    )

    exhibit_boost = retriever._metadata_boost(
        {
            "is_table_stub": False,
            "source_type": "text",
            "section_path": ["Exhibits"],
        },
        "What drove revenue change?",
    )

    assert mdna_boost > exhibit_boost


def test_format_results_for_tool_uses_c_and_t_markers():
    retriever = make_retriever()

    text = retriever._format_results_for_tool(
        [
            {
                "rank": 1,
                "score": 0.9,
                "text": "Narrative evidence",
                "doc_name": "PEPSICO_2023_8K",
                "document_id": "doc-1",
                "page_number": 1,
                "is_table_stub": False,
                "source_type": "text",
                "local_chunk_index": 0,
            },
            {
                "rank": 2,
                "score": 0.8,
                "text": "[FINANCIAL TABLE METADATA]",
                "doc_name": "PEPSICO_2023_8K",
                "document_id": "doc-1",
                "page_number": 2,
                "is_table_stub": True,
                "source_type": "table_stub",
                "table_name": "tbl_doc_p002_t000",
                "statement_type": "financial_notes",
            },
        ]
    )

    assert "[C1 |" in text
    assert "[T2 |" in text
    assert "table=tbl_doc_p002_t000" in text
    assert "Narrative evidence" in text


@pytest.mark.asyncio
async def test_search_text_returns_no_results_for_empty_allowed_documents():
    retriever = make_retriever()

    result = await retriever.search_text(
        query="revenue",
        allowed_document_ids=[],
    )

    assert result == "No relevant results found."


@pytest.mark.asyncio
async def test_search_calls_vector_repo_with_allowed_document_ids():
    repo = FakeVectorRepo(
        points=[
            FakePoint(
                id="point-1",
                score=0.7,
                payload={
                    "text": "Revenue increased.",
                    "document_id": "doc-1",
                    "doc_name": "PEPSICO_2023_8K",
                    "is_table_stub": False,
                    "source_type": "text",
                    "local_chunk_index": 1,
                },
            )
        ]
    )
    retriever = make_retriever(vector_repo=repo)

    results = await retriever.search(
        query="revenue",
        allowed_document_ids=["doc-1"],
        top_k=3,
    )

    assert len(results) == 1
    assert results[0]["document_id"] == "doc-1"

    assert repo.hybrid_calls
    first_call = repo.hybrid_calls[0]
    assert first_call["allowed_document_ids"] == ["doc-1"]
    assert first_call["dense_vector"] == [0.1, 0.2, 0.3]
    assert first_call["sparse_indices"] == [1, 2, 3]
    assert first_call["sparse_values"] == [0.3, 0.2, 0.1]


@pytest.mark.asyncio
async def test_search_tables_uses_only_table_stubs_filter():
    repo = FakeVectorRepo(
        points=[
            FakePoint(
                id="table-point-1",
                score=0.8,
                payload={
                    "text": "[FINANCIAL TABLE METADATA]",
                    "document_id": "doc-1",
                    "doc_name": "PEPSICO_2023_8K",
                    "is_table_stub": True,
                    "source_type": "table_stub",
                    "table_name": "tbl_doc_p002_t000",
                },
            )
        ]
    )
    retriever = make_retriever(vector_repo=repo)

    result = await retriever.search_tables(
        concept="income statement",
        allowed_document_ids=["doc-1"],
    )

    assert "[T1 |" in result
    assert "table=tbl_doc_p002_t000" in result

    assert repo.hybrid_calls
    assert all(call["only_table_stubs"] is True for call in repo.hybrid_calls)


@pytest.mark.asyncio
async def test_search_uses_reranker_when_available():
    repo = FakeVectorRepo(
        points=[
            FakePoint(
                id="point-low",
                score=0.9,
                payload={
                    "text": "Lower reranker score",
                    "document_id": "doc-1",
                    "doc_name": "PEPSICO_2023_8K",
                    "is_table_stub": False,
                    "source_type": "text",
                },
            ),
            FakePoint(
                id="point-high",
                score=0.1,
                payload={
                    "text": "Higher reranker score",
                    "document_id": "doc-1",
                    "doc_name": "PEPSICO_2023_8K",
                    "is_table_stub": False,
                    "source_type": "text",
                },
            ),
        ]
    )
    reranker = FakeReranker(scores=[0.1, 0.9])
    retriever = make_retriever(vector_repo=repo, reranker=reranker)

    results = await retriever.search(
        query="revenue",
        allowed_document_ids=["doc-1"],
        top_k=2,
    )

    assert reranker.calls
    assert results[0]["text"] == "Higher reranker score"


@pytest.mark.asyncio
async def test_search_text_expands_neighbor_chunks():
    repo = FakeVectorRepo(
        points=[
            FakePoint(
                id="hit",
                score=0.9,
                payload={
                    "text": "Main hit",
                    "document_id": "doc-1",
                    "doc_name": "PEPSICO_2023_8K",
                    "is_table_stub": False,
                    "source_type": "text",
                    "local_chunk_index": 5,
                },
            )
        ],
        neighbor_points=[
            FakePoint(
                id="prev",
                score=0.0,
                payload={
                    "text": "Previous neighbor",
                    "document_id": "doc-1",
                    "doc_name": "PEPSICO_2023_8K",
                    "is_table_stub": False,
                    "source_type": "text",
                    "local_chunk_index": 4,
                },
            ),
            FakePoint(
                id="hit-neighbor-copy",
                score=0.0,
                payload={
                    "text": "Main hit duplicate by index",
                    "document_id": "doc-1",
                    "doc_name": "PEPSICO_2023_8K",
                    "is_table_stub": False,
                    "source_type": "text",
                    "local_chunk_index": 5,
                },
            ),
            FakePoint(
                id="next",
                score=0.0,
                payload={
                    "text": "Next neighbor",
                    "document_id": "doc-1",
                    "doc_name": "PEPSICO_2023_8K",
                    "is_table_stub": False,
                    "source_type": "text",
                    "local_chunk_index": 6,
                },
            ),
        ],
    )
    retriever = make_retriever(vector_repo=repo)

    result = await retriever.search_text(
        query="revenue",
        allowed_document_ids=["doc-1"],
        expand_neighbors=True,
        neighbor_window=1,
    )

    assert repo.neighbor_calls
    assert repo.neighbor_calls[0]["document_id"] == "doc-1"
    assert repo.neighbor_calls[0]["center_index"] == 5
    assert repo.neighbor_calls[0]["window"] == 1

    assert "Main hit" in result
    assert "Previous neighbor" in result
    assert "Next neighbor" in result
    assert "previous_neighbor" in result
    assert "next_neighbor" in result