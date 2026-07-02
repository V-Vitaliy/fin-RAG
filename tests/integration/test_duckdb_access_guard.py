import uuid

import asyncio
import duckdb
import pytest

from app.infrastructure.duckdb import DuckDBManager
from app.repositories.duckdb_repo import DuckDBAccessError, DuckDBTableRepository


def _create_test_duckdb(path):
    allowed_doc_id = str(uuid.uuid4())
    blocked_doc_id = str(uuid.uuid4())

    with duckdb.connect(str(path)) as conn:
        conn.execute(
            """
            CREATE TABLE _table_catalog
            (
                table_name VARCHAR,
                doc_name VARCHAR,
                page_number INTEGER,
                statement_type VARCHAR,
                document_id VARCHAR,
                workspace_id VARCHAR,
                citation_label VARCHAR,
                evidence_id VARCHAR,
                unit_scale VARCHAR,
                is_primary_statement BOOLEAN,
                is_active BOOLEAN
            )
            """
        )

        conn.execute('CREATE TABLE "tbl_allowed" ("Line Item" VARCHAR, "2023" INTEGER)')
        conn.execute('INSERT INTO "tbl_allowed" VALUES (?, ?)', ["Revenue", 100])

        conn.execute('CREATE TABLE "tbl_blocked" ("Line Item" VARCHAR, "2023" INTEGER)')
        conn.execute('INSERT INTO "tbl_blocked" VALUES (?, ?)', ["Revenue", 200])

        conn.execute('CREATE TABLE "tbl_inactive" ("Line Item" VARCHAR, "2023" INTEGER)')
        conn.execute('INSERT INTO "tbl_inactive" VALUES (?, ?)', ["Revenue", 300])

        conn.execute(
            """
            INSERT INTO _table_catalog
            VALUES
                ('tbl_allowed', 'AllowedDoc', 10, 'income_statement', ?, 'workspace-a',
                 'AllowedDoc p.10, table=tbl_allowed', 'ev1', 'millions', true, true),
                ('tbl_blocked', 'BlockedDoc', 11, 'income_statement', ?, 'workspace-b',
                 'BlockedDoc p.11, table=tbl_blocked', 'ev2', 'millions', true, true),
                ('tbl_inactive', 'InactiveDoc', 12, 'income_statement', ?, 'workspace-a',
                 'InactiveDoc p.12, table=tbl_inactive', 'ev3', 'millions', true, false)
            """,
            [allowed_doc_id, blocked_doc_id, allowed_doc_id],
        )

    return allowed_doc_id, blocked_doc_id


@pytest.mark.asyncio
async def test_duckdb_repo_allows_table_from_allowed_document(tmp_path):
    db_path = tmp_path / "finance_test.duckdb"
    allowed_doc_id, _ = _create_test_duckdb(db_path)

    repo = DuckDBTableRepository(DuckDBManager(str(db_path)))

    df = await repo.execute_select(
        'SELECT * FROM "tbl_allowed"',
        allowed_document_ids=[allowed_doc_id],
    )

    assert len(df) == 1
    assert df.iloc[0]["2023"] == 100


@pytest.mark.asyncio
async def test_duckdb_repo_blocks_table_from_other_document(tmp_path):
    db_path = tmp_path / "finance_test.duckdb"
    allowed_doc_id, _ = _create_test_duckdb(db_path)

    repo = DuckDBTableRepository(DuckDBManager(str(db_path)))

    with pytest.raises(DuckDBAccessError):
        await repo.execute_select(
            'SELECT * FROM "tbl_blocked"',
            allowed_document_ids=[allowed_doc_id],
        )

@pytest.mark.asyncio
async def test_duckdb_repo_blocks_inactive_table(tmp_path):
    db_path = tmp_path / "finance_test.duckdb"
    allowed_doc_id, _ = _create_test_duckdb(db_path)

    repo = DuckDBTableRepository(DuckDBManager(str(db_path)))

    with pytest.raises(DuckDBAccessError):
        await repo.execute_select(
            'SELECT * FROM "tbl_inactive"',
            allowed_document_ids=[allowed_doc_id],
        )