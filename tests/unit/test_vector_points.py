import uuid

from qdrant_client.http import models

from app.services.ingestion.vector_points import (
    build_qdrant_payload,
    build_qdrant_point,
)


def test_build_qdrant_payload_contains_document_and_chunk_identity():
    document_id = uuid.uuid4()
    workspace_id = uuid.uuid4()

    payload = build_qdrant_payload(
        document_id=document_id,
        workspace_id=workspace_id,
        filename="Apple 2023 10-K.pdf",
        doc_name="apple_2023_10k",
        chunk_id="chunk-001",
        chunk_index=1,
        source_type="text",
        text="Revenue increased year over year.",
        page_number=42,
        section_path=["MD&A", "Results of Operations"],
        content_hash="a" * 64,
        ingestion_version=2,
    )

    assert payload["document_id"] == str(document_id)
    assert payload["workspace_id"] == str(workspace_id)
    assert payload["filename"] == "Apple 2023 10-K.pdf"
    assert payload["doc_name"] == "apple_2023_10k"

    assert payload["chunk_id"] == "chunk-001"
    assert payload["chunk_index"] == 1
    assert payload["source_type"] == "text"

    assert payload["text"] == "Revenue increased year over year."
    assert payload["text_bm25"] == "Revenue increased year over year."

    assert payload["page_number"] == 42
    assert payload["section_path"] == ["MD&A", "Results of Operations"]
    assert payload["content_hash"] == "a" * 64
    assert payload["ingestion_version"] == 2


def test_build_qdrant_payload_removes_none_but_keeps_false():
    payload = build_qdrant_payload(
        document_id=uuid.uuid4(),
        workspace_id=uuid.uuid4(),
        filename="Doc.pdf",
        doc_name="doc",
        chunk_id="chunk-001",
        chunk_index=0,
        source_type="text",
        text="Some text.",
        page_number=None,
        is_table_stub=False,
        table_name=None,
    )

    assert "page_number" not in payload
    assert "table_name" not in payload
    assert payload["is_table_stub"] is False


def test_build_qdrant_point_uses_stable_uuid_and_vectors():
    document_id = uuid.uuid4()
    workspace_id = uuid.uuid4()

    point_1 = build_qdrant_point(
        document_id=document_id,
        workspace_id=workspace_id,
        filename="Apple 2023 10-K.pdf",
        doc_name="apple_2023_10k",
        chunk_id="chunk-001",
        chunk_index=5,
        source_type="text",
        text="Revenue increased.",
        dense_vector=[0.1, 0.2, 0.3],
        sparse_indices=[1, 5, 9],
        sparse_values=[0.7, 0.2, 0.1],
    )

    point_2 = build_qdrant_point(
        document_id=document_id,
        workspace_id=workspace_id,
        filename="Apple 2023 10-K.pdf",
        doc_name="apple_2023_10k",
        chunk_id="chunk-001",
        chunk_index=5,
        source_type="text",
        text="Revenue increased.",
        dense_vector=[0.1, 0.2, 0.3],
        sparse_indices=[1, 5, 9],
        sparse_values=[0.7, 0.2, 0.1],
    )

    assert point_1.id == point_2.id
    assert str(uuid.UUID(point_1.id)) == point_1.id

    assert point_1.vector["dense"] == [0.1, 0.2, 0.3]
    assert isinstance(point_1.vector["bm25"], models.SparseVector)
    assert point_1.vector["bm25"].indices == [1, 5, 9]
    assert point_1.vector["bm25"].values == [0.7, 0.2, 0.1]

    assert point_1.payload["document_id"] == str(document_id)
    assert point_1.payload["chunk_index"] == 5


def test_build_qdrant_point_for_table_stub_contains_table_metadata():
    document_id = uuid.uuid4()
    workspace_id = uuid.uuid4()

    point = build_qdrant_point(
        document_id=document_id,
        workspace_id=workspace_id,
        filename="Apple 2023 10-K.pdf",
        doc_name="apple_2023_10k",
        chunk_id="table-stub-001",
        chunk_index=10,
        source_type="table_stub",
        text="Table: Consolidated Statements of Operations",
        dense_vector=[0.1, 0.2, 0.3],
        is_table_stub=True,
        table_name="tbl_abcd1234_apple_2023_10k_p130_t001",
        statement_type="income_statement",
        citation_label="Apple 2023 10-K p.130",
        evidence_id="tbl_abcd1234_apple_2023_10k_p130_t001",
        unit_scale="millions",
        is_primary_statement=True,
    )

    assert point.payload["source_type"] == "table_stub"
    assert point.payload["is_table_stub"] is True
    assert point.payload["table_name"] == "tbl_abcd1234_apple_2023_10k_p130_t001"
    assert point.payload["statement_type"] == "income_statement"
    assert point.payload["citation_label"] == "Apple 2023 10-K p.130"
    assert point.payload["evidence_id"] == "tbl_abcd1234_apple_2023_10k_p130_t001"
    assert point.payload["unit_scale"] == "millions"
    assert point.payload["is_primary_statement"] is True