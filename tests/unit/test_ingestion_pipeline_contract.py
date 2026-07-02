import uuid

import pytest

from app.services.ingestion.pipeline import (
    IngestionDocument,
    IngestionPipeline,
    _fix_markdown_tables_split_across_pages,
)


def _pipeline_without_init() -> IngestionPipeline:
    """
    Build pipeline instance without calling __init__.

    We only test pure contract/private helper logic here,
    not Docling/Qdrant/DuckDB/embeddings.
    """
    return object.__new__(IngestionPipeline)


def test_coerce_document_requires_document_id(tmp_path):
    pipeline = _pipeline_without_init()
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    with pytest.raises(ValueError, match="document_id"):
        pipeline._coerce_document(
            {
                "pdf_path": str(pdf_path),
                "workspace_id": str(uuid.uuid4()),
                "filename": "doc.pdf",
                "doc_name": "doc",
            }
        )


def test_coerce_document_requires_workspace_id(tmp_path):
    pipeline = _pipeline_without_init()
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    with pytest.raises(ValueError, match="workspace_id"):
        pipeline._coerce_document(
            {
                "pdf_path": str(pdf_path),
                "document_id": str(uuid.uuid4()),
                "filename": "doc.pdf",
                "doc_name": "doc",
            }
        )


def test_coerce_document_builds_ingestion_document(tmp_path):
    pipeline = _pipeline_without_init()
    pdf_path = tmp_path / "Apple 2023 10-K.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    document_id = uuid.uuid4()
    workspace_id = uuid.uuid4()

    document = pipeline._coerce_document(
        {
            "pdf_path": str(pdf_path),
            "document_id": document_id,
            "workspace_id": workspace_id,
            "filename": "Apple 2023 10-K.pdf",
            "doc_name": "apple_2023_10k",
            "content_hash": "a" * 64,
            "ingestion_version": 3,
        }
    )

    assert isinstance(document, IngestionDocument)
    assert document.pdf_path == str(pdf_path)
    assert document.document_id == document_id
    assert document.workspace_id == workspace_id
    assert document.filename == "Apple 2023 10-K.pdf"
    assert document.doc_name == "apple_2023_10k"
    assert document.content_hash == "a" * 64
    assert document.ingestion_version == 3


def test_fix_markdown_tables_split_across_pages_joins_page_break_inside_table():
    page_break = "---PAGE_BREAK---"

    markdown = (
        "# Page 1\n\n"
        "Intro text.\n\n"
        "| Line Item | 2023 |\n"
        "|---|---|\n"
        "| Revenue | 100 |\n"
        f"{page_break}\n"
        "| Net income | 20 |\n\n"
        "# Page 2\n\n"
        "More text."
    )

    pages = _fix_markdown_tables_split_across_pages(markdown, page_break)

    assert len(pages) == 1
    assert page_break not in pages[0]
    assert "| Revenue | 100 |" in pages[0]
    assert "| Net income | 20 |" in pages[0]

def test_fix_markdown_tables_split_across_pages_moves_hanging_table_to_next_page():
    page_break = "---PAGE_BREAK---"

    markdown = (
        "# Page 1\n\n"
        "Intro text.\n\n"
        "| Line Item | 2023 |\n"
        "|---|---|\n"
        "| Revenue | 100 |\n\n"
        f"{page_break}\n"
        "# Page 2\n\n"
        "| Net income | 20 |\n"
        "| Operating income | 30 |\n\n"
        "More text."
    )

    pages = _fix_markdown_tables_split_across_pages(markdown, page_break)

    assert len(pages) == 2
    assert "| Revenue | 100 |" not in pages[0]
    assert "| Revenue | 100 |" in pages[1]
    assert "| Net income | 20 |" in pages[1]
    assert "| Operating income | 30 |" in pages[1]
    
def test_assign_chunk_metadata_marks_sql_table_stub():
    pipeline = _pipeline_without_init()

    document = IngestionDocument(
        pdf_path="/tmp/doc.pdf",
        document_id=uuid.uuid4(),
        workspace_id=uuid.uuid4(),
        filename="Apple 2023 10-K.pdf",
        doc_name="apple_2023_10k",
        content_hash="a" * 64,
        ingestion_version=2,
    )

    chunks = [
        {
            "text": "Table metadata text",
            "is_table_stub": True,
            "table_name": "tbl_abcd1234_apple_2023_10k_p001_t001",
            "page_number": 1,
        }
    ]

    pipeline._assign_chunk_metadata(document=document, chunks=chunks)

    chunk = chunks[0]

    assert chunk["document_id"] == str(document.document_id)
    assert chunk["workspace_id"] == str(document.workspace_id)
    assert chunk["filename"] == "Apple 2023 10-K.pdf"
    assert chunk["doc_name"] == "apple_2023_10k"
    assert chunk["chunk_index"] == 0
    assert chunk["source_type"] == "table_stub"
    assert chunk["is_table_stub"] is True
    assert chunk["chunk_id"] == "tbl_abcd1234_apple_2023_10k_p001_t001:stub"
    assert chunk["evidence_id"] == "tbl_abcd1234_apple_2023_10k_p001_t001"
    assert "citation_label" in chunk
    assert chunk["content_hash"] == "a" * 64
    assert chunk["ingestion_version"] == 2


def test_assign_chunk_metadata_marks_markdown_table_reference_as_text_evidence():
    pipeline = _pipeline_without_init()

    document = IngestionDocument(
        pdf_path="/tmp/doc.pdf",
        document_id=uuid.uuid4(),
        workspace_id=uuid.uuid4(),
        filename="Apple 2023 10-K.pdf",
        doc_name="apple_2023_10k",
        content_hash=None,
        ingestion_version=1,
    )

    chunks = [
        {
            "text": "| Line Item | 2023 |\n| Revenue | 100 |",
            "is_table_stub": True,
            "page_number": 2,
        }
    ]

    pipeline._assign_chunk_metadata(document=document, chunks=chunks)

    chunk = chunks[0]

    assert chunk["is_table_stub"] is False
    assert chunk["source_type"] == "text_table_reference"
    assert chunk["document_id"] == str(document.document_id)
    assert chunk["workspace_id"] == str(document.workspace_id)
    assert chunk["chunk_index"] == 0
    assert chunk["chunk_id"] == f"{document.document_id}:chunk:0"
    assert chunk["evidence_id"] == chunk["chunk_id"]