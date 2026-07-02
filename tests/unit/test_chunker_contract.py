import pytest

from app.services.ingestion.chunker import ChunkerConfig, HierarchicalChunker


class FakeTokenizer:
    def count(self, text: str) -> int:
        return len(str(text or "").split())

    def encode(self, text: str) -> list[int]:
        return list(range(self.count(text)))

    def decode(self, ids: list[int]) -> str:
        return " ".join(f"tok{i}" for i in ids)


def _chunker_without_real_tokenizer(config: ChunkerConfig | None = None) -> HierarchicalChunker:
    chunker = object.__new__(HierarchicalChunker)
    chunker.cfg = config or ChunkerConfig(
        max_tokens=40,
        min_tokens=3,
        overlap_tokens=5,
        tokenizer_type="tiktoken",
        tokenizer_name="cl100k_base",
    )
    chunker._tokenizer = FakeTokenizer()
    return chunker


def test_chunker_config_validates_overlap_smaller_than_max_tokens():
    with pytest.raises(ValueError, match="overlap_tokens"):
        ChunkerConfig(max_tokens=10, overlap_tokens=10)


def test_chunker_config_validates_merge_ratio():
    with pytest.raises(ValueError, match="merge_ratio"):
        ChunkerConfig(merge_ratio=0)


def test_chunk_document_emits_only_semantic_metadata():
    chunker = _chunker_without_real_tokenizer()

    chunks = chunker.chunk_document(
        markdown=(
            "# Item 7\n\n"
            "Revenue increased because of higher customer demand and pricing improvements.\n\n"
            "Net income increased by 12%."
        ),
        doc_name="AAPL_2023_10K",
        page_number=5,
    )

    assert chunks

    first = chunks[0]

    assert "local_chunk_index" in first
    assert "chunk_id" not in first
    assert "workspace_id" not in first
    assert "document_id" not in first
    assert "content_hash" not in first
    assert "ingestion_version" not in first

    assert first["doc_name"] == "AAPL_2023_10K"
    assert first["page_number"] == 5
    assert first["section"] == "Item 7"
    assert first["section_path"] == "Item 7"
    assert first["is_table_stub"] is False
    assert first["text"]
    assert first["text_bm25"]


def test_chunk_document_marks_markdown_table_reference():
    chunker = _chunker_without_real_tokenizer()

    chunks = chunker.chunk_document(
        markdown=(
            "# Financial Statements\n\n"
            "| Line Item | 2023 |\n"
            "|---|---|\n"
            "| Revenue | 100 |"
        ),
        doc_name="AAPL_2023_10K",
        page_number=10,
    )

    assert chunks
    assert any(chunk["is_table_stub"] is True for chunk in chunks)
    assert any("[TABLE ON PAGE 10:" in chunk["text"] for chunk in chunks)


def test_truncate_text_uses_configured_tokenizer():
    chunker = _chunker_without_real_tokenizer()

    result = chunker.truncate_text("one two three four five", max_tokens=3)

    assert result == "tok0 tok1 tok2"