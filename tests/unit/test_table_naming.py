import re
import uuid

from app.services.ingestion.naming import (
    make_safe_doc_slug,
    make_table_name,
    make_qdrant_point_id,
)


def test_safe_doc_slug_is_duckdb_safe():
    slug = make_safe_doc_slug("Johnson & Johnson 2022 10-K.pdf")

    assert slug == "johnson_johnson_2022_10_k"
    assert re.fullmatch(r"[a-z0-9_]+", slug)


def test_table_name_is_unique_and_duckdb_safe():
    doc_a = uuid.uuid4()
    doc_b = uuid.uuid4()

    name_a = make_table_name(
        document_id=doc_a,
        doc_name="FOOTLOCKER_2022_8K_dated_2022-08-19.pdf",
        table_index=37,
        page_number=130,
    )
    name_b = make_table_name(
        document_id=doc_b,
        doc_name="AES_2022_10K.pdf",
        table_index=37,
        page_number=130,
    )

    assert name_a != name_b
    assert name_a.startswith("tbl_")
    assert "_footlocker_2022_8k_" in name_a
    assert "_p130_t037" in name_a
    assert re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name_a)

def test_table_name_strips_sec_dated_suffix():
    document_id = "e177aaae-0000-4000-8000-000000000000"

    table_name = make_table_name(
        document_id=document_id,
        doc_name="JOHNSON_JOHNSON_2023_8K_dated-2023-08-30",
        table_index=37,
        page_number=130,
    )

    assert "_johnson_johnson_2023_8k_" in table_name
    assert "dated" not in table_name
    assert "2022_08_19" not in table_name
    assert table_name.endswith("_p130_t037")

def test_qdrant_point_id_is_stable_uuid():
    document_id = uuid.uuid4()

    point_1 = make_qdrant_point_id(
        document_id=document_id,
        chunk_index=5,
        source_type="text",
    )
    point_2 = make_qdrant_point_id(
        document_id=document_id,
        chunk_index=5,
        source_type="text",
    )
    point_3 = make_qdrant_point_id(
        document_id=document_id,
        chunk_index=6,
        source_type="text",
    )

    assert point_1 == point_2
    assert point_1 != point_3
    assert str(uuid.UUID(point_1)) == point_1