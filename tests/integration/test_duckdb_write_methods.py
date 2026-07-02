import uuid

import pandas as pd
import pytest

from app.infrastructure.duckdb import DuckDBManager
from app.repositories.duckdb_repo import DuckDBCatalogRow, DuckDBTableRepository


@pytest.mark.asyncio
async def test_write_dataframe_table_and_insert_catalog_row(tmp_path):
    db_path = tmp_path / "finance_test.duckdb"
    repo = DuckDBTableRepository(DuckDBManager(str(db_path)))

    document_id = uuid.uuid4()
    workspace_id = uuid.uuid4()
    table_name = "tbl_abcd1234_test_doc_p001_t001"

    await repo.ensure_catalog_schema()

    rows_written = await repo.write_dataframe_table(
        table_name=table_name,
        dataframe=pd.DataFrame(
            [
                {"Line Item": "Revenue", "2023": 100},
                {"Line Item": "Net income", "2023": 20},
            ]
        ),
    )

    await repo.insert_catalog_row(
        DuckDBCatalogRow(
            table_name=table_name,
            doc_name="test_doc",
            col_names="Line Item|2023",
            row_labels="Revenue|Net income",
            section_hint="Financial Statements",
            page_number=1,
            statement_type="income_statement",
            first_col="Line Item",
            year_columns="2023",
            evidence_id=table_name,
            citation_label="test_doc p.1",
            unit_scale="millions",
            is_primary_statement=True,
            document_id=document_id,
            workspace_id=workspace_id,
            content_hash="a" * 64,
            ingestion_version=1,
            is_active=True,
        )
    )

    assert rows_written == 2

    df = await repo.execute_select(
        f'SELECT * FROM "{table_name}"',
        allowed_document_ids=[document_id],
    )

    assert len(df) == 2
    assert list(df["Line Item"]) == ["Revenue", "Net income"]
    assert list(df["2023"]) == [100, 20]


@pytest.mark.asyncio
async def test_deactivate_document_tables_blocks_future_access(tmp_path):
    db_path = tmp_path / "finance_test.duckdb"
    repo = DuckDBTableRepository(DuckDBManager(str(db_path)))

    document_id = uuid.uuid4()
    workspace_id = uuid.uuid4()
    table_name = "tbl_abcd1234_test_doc_p001_t001"

    await repo.ensure_catalog_schema()

    await repo.write_dataframe_table(
        table_name=table_name,
        dataframe=pd.DataFrame([{"Line Item": "Revenue", "2023": 100}]),
    )

    await repo.insert_catalog_row(
        DuckDBCatalogRow(
            table_name=table_name,
            doc_name="test_doc",
            col_names="Line Item|2023",
            row_labels="Revenue",
            section_hint=None,
            page_number=1,
            statement_type="income_statement",
            first_col="Line Item",
            year_columns="2023",
            evidence_id=table_name,
            citation_label="test_doc p.1",
            unit_scale="millions",
            is_primary_statement=True,
            document_id=document_id,
            workspace_id=workspace_id,
            content_hash="a" * 64,
            ingestion_version=1,
            is_active=True,
        )
    )

    affected = await repo.deactivate_document_tables(document_id)

    assert affected == 1

    entry = await repo.get_catalog_entry(table_name)
    assert entry is not None
    assert entry["is_active"] is False