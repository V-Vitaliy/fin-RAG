from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import duckdb
from app.infrastructure.duckdb import DuckDBManager


class DuckDBAccessError(PermissionError):
    """Raised when an agent/tool tries to access a table outside its document scope."""

@dataclass(frozen=True)
class DuckDBCatalogRow:
    table_name: str
    doc_name: str
    col_names: str
    row_labels: str
    section_hint: str | None
    page_number: int | None
    statement_type: str | None
    first_col: str | None
    year_columns: str | None
    evidence_id: str | None
    citation_label: str | None
    unit_scale: str | None
    is_primary_statement: bool | None
    document_id: UUID | str
    workspace_id: UUID | str
    content_hash: str | None
    ingestion_version: int = 1
    is_active: bool = True


class DuckDBTableRepository:
    """
    Repository for DuckDB financial tables and _table_catalog.

    This repository is the security boundary for SQL table access:
    tools should not execute SQL directly against DuckDB without checking
    document_id/is_active in _table_catalog.
    """

    def __init__(self, manager: DuckDBManager):
        self._manager = manager

    @staticmethod
    def quote_ident(name: str) -> str:
        return '"' + str(name).replace('"', '""') + '"'

    @staticmethod
    def validate_table_name(table_name:str) -> None:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*",str(table_name) or ""):
            raise ValueError(f"Unsafe DuckDB table name: {table_name!r}")

    @staticmethod
    def extract_table_name(query: str | None) -> str | None:
        if not query:
            return None

        match = re.search(
            r'\bFROM\s+["\'`]?([A-Za-z_][A-Za-z0-9_]*)["\'`]?',
            str(query),
            re.IGNORECASE,
        )
        return match.group(1) if match else None

    @staticmethod
    def normalize_document_ids(document_ids: list[UUID | str]) -> list[str]:
        return [str(item) for item in document_ids]

    async def ensure_catalog_schema(self) -> None:
        await asyncio.to_thread(self._ensure_catalog_schema_sync)

    def _ensure_catalog_schema_sync(self) -> None:
        """
        Creates/updates _table_catalog with production metadata fields.
        """
        with self._manager.connect_writer() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS _table_catalog
                (
                    table_name
                    VARCHAR,
                    doc_name
                    VARCHAR,
                    col_names
                    VARCHAR,
                    row_labels
                    VARCHAR,
                    section_hint
                    VARCHAR,
                    page_number
                    INTEGER,
                    statement_type
                    VARCHAR,
                    first_col
                    VARCHAR,
                    year_columns
                    VARCHAR,
                    evidence_id
                    VARCHAR,
                    citation_label
                    VARCHAR,
                    unit_scale
                    VARCHAR,
                    is_primary_statement
                    BOOLEAN,
                    document_id
                    VARCHAR,
                    workspace_id
                    VARCHAR,
                    content_hash
                    VARCHAR,
                    ingestion_version
                    INTEGER,
                    is_active
                    BOOLEAN
                )
                """
            )

            ddl_statements = [
                "ALTER TABLE _table_catalog ADD COLUMN IF NOT EXISTS evidence_id VARCHAR",
                "ALTER TABLE _table_catalog ADD COLUMN IF NOT EXISTS citation_label VARCHAR",
                "ALTER TABLE _table_catalog ADD COLUMN IF NOT EXISTS unit_scale VARCHAR",
                "ALTER TABLE _table_catalog ADD COLUMN IF NOT EXISTS is_primary_statement BOOLEAN",
                "ALTER TABLE _table_catalog ADD COLUMN IF NOT EXISTS document_id VARCHAR",
                "ALTER TABLE _table_catalog ADD COLUMN IF NOT EXISTS workspace_id VARCHAR",
                "ALTER TABLE _table_catalog ADD COLUMN IF NOT EXISTS content_hash VARCHAR",
                "ALTER TABLE _table_catalog ADD COLUMN IF NOT EXISTS ingestion_version INTEGER",
                "ALTER TABLE _table_catalog ADD COLUMN IF NOT EXISTS is_active BOOLEAN",
            ]

            for ddl in ddl_statements:
                conn.execute(ddl)

    async def get_catalog_entry(self, table_name: str) -> dict[str, Any] | None:
        return await asyncio.to_thread(self._get_catalog_entry_sync, table_name)

    def _get_catalog_entry_sync(self, table_name: str) -> dict[str, Any] | None:
        with self._manager.connect_readonly() as conn:
            row = conn.execute(
                """
                SELECT
                    table_name,
                    doc_name,
                    page_number,
                    statement_type,
                    document_id,
                    workspace_id,
                    citation_label,
                    evidence_id,
                    unit_scale,
                    is_primary_statement,
                    is_active
                FROM _table_catalog
                WHERE table_name = ?
                LIMIT 1
                """,
                [table_name],
            ).fetchone()

        if not row:
            return None

        keys = [
            "table_name",
            "doc_name",
            "page_number",
            "statement_type",
            "document_id",
            "workspace_id",
            "citation_label",
            "evidence_id",
            "unit_scale",
            "is_primary_statement",
            "is_active",
        ]
        return dict(zip(keys, row))

    async def validate_table_access(
        self,
        table_name: str,
        allowed_document_ids: list[UUID | str],
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self._validate_table_access_sync,
            table_name,
            allowed_document_ids,
        )

    def _validate_table_access_sync(
        self,
        table_name: str,
        allowed_document_ids: list[UUID | str],
    ) -> dict[str, Any]:
        """
        Validate that table belongs to an allowed active document.

        Returns catalog entry when access is allowed.
        Raises DuckDBAccessError when blocked.
        """
        if not table_name or table_name == "_table_catalog":
            raise DuckDBAccessError("Access to this table is not allowed.")

        allowed = set(self.normalize_document_ids(allowed_document_ids))
        entry = self._get_catalog_entry_sync(table_name)

        if not entry:
            raise DuckDBAccessError(
                f"Table '{table_name}' is not registered in _table_catalog."
            )

        if not entry.get("is_active"):
            raise DuckDBAccessError(
                f"Table '{table_name}' belongs to an inactive document version."
            )

        document_id = str(entry.get("document_id") or "")
        if document_id not in allowed:
            raise DuckDBAccessError(
                f"Table '{table_name}' is outside the allowed document scope."
            )

        return entry

    async def execute_select(
        self,
        query: str,
        allowed_document_ids: list[UUID | str],
    ):
        return await asyncio.to_thread(
            self._execute_select_sync,
            query,
            allowed_document_ids,
        )

    def _execute_select_sync(
        self,
        query: str,
        allowed_document_ids: list[UUID | str],
    ):
        """
        Execute a SELECT query after validating the referenced table.

        Returns a pandas DataFrame.
        """
        query = str(query or "").strip()

        if not query.lower().startswith("select"):
            raise ValueError("Only SELECT queries are allowed.")

        table_name = self.extract_table_name(query)
        if not table_name:
            raise ValueError("Could not detect table name from SQL query.")

        self._validate_table_access_sync(
            table_name=table_name,
            allowed_document_ids=allowed_document_ids,
        )

        with self._manager.connect_readonly() as conn:
            return conn.execute(query).fetchdf()

    async def describe_table(
        self,
        table_name: str,
        allowed_document_ids: list[UUID | str],
        include_sample: bool = True,
        sample_rows: int = 25,
    ) -> str:
        return await asyncio.to_thread(
            self._describe_table_sync,
            table_name,
            allowed_document_ids,
            include_sample,
            sample_rows,
        )

    def _describe_table_sync(
        self,
        table_name: str,
        allowed_document_ids: list[UUID | str],
        include_sample: bool = True,
        sample_rows: int = 25,
    ) -> str:
        """
        Return schema and optional sample for an allowed table.
        This is intentionally text-formatted because tools will pass it to the agent.
        """
        entry = self._validate_table_access_sync(
            table_name=table_name,
            allowed_document_ids=allowed_document_ids,
        )

        sample_rows = max(1, min(int(sample_rows or 25), 120))

        with self._manager.connect_readonly() as conn:
            schema_df = conn.execute(
                f"DESCRIBE {self.quote_ident(table_name)}"
            ).fetchdf()

            out = [
                f"Source: [{entry.get('citation_label') or table_name}]",
                f"Table: {table_name}",
                f"Document ID: {entry.get('document_id')}",
                f"Statement type: {entry.get('statement_type')}",
                "",
                "Schema:",
                schema_df.to_markdown(index=False),
            ]

            if include_sample:
                sample_df = conn.execute(
                    f"SELECT * FROM {self.quote_ident(table_name)} LIMIT {sample_rows}"
                ).fetchdf()
                out.extend(
                    [
                        "",
                        f"Sample: showing {len(sample_df)} rows",
                        sample_df.to_markdown(index=False),
                    ]
                )

        return "\n".join(out)

    async def write_dataframe_table(
            self,
            table_name: str,
            dataframe: Any,
            replace: bool = True,
    ) -> int:
        async with self._manager.write_lock:
            return await asyncio.to_thread(
                self._write_dataframe_table_sync,
                table_name,
                dataframe,
                replace,
            )

    def _write_dataframe_table_sync(
        self,
        table_name: str,
        dataframe: Any,
        replace: bool = True,
    ) -> int:
        self.validate_table_name(table_name)

        with self._manager.connect_writer() as conn:
            conn.register("_df_to_write", dataframe)
            try:
                if replace:
                    conn.execute(
                        f"""
                        CREATE OR REPLACE TABLE {self.quote_ident(table_name)}
                        AS SELECT * FROM _df_to_write
                        """
                    )
                else:
                    conn.execute(
                        f"""
                        CREATE TABLE {self.quote_ident(table_name)}
                        AS SELECT * FROM _df_to_write
                        """
                    )

                row_count = conn.execute(
                    f"SELECT COUNT(*) FROM {self.quote_ident(table_name)}"
                ).fetchone()[0]

                return int(row_count)
            finally:
                conn.unregister("_df_to_write")

    async def insert_catalog_row(self, row: DuckDBCatalogRow) -> None:
        """
        Insert one row into _table_catalog.
        Assumes ensure_catalog_schema() was called earlier.
        """
        async with self._manager.write_lock:
            await asyncio.to_thread(self._insert_catalog_row_sync, row)

    def _insert_catalog_row_sync(self, row: DuckDBCatalogRow) -> None:
        self.validate_table_name(row.table_name)

        with self._manager.connect_writer() as conn:
            conn.execute(
                """
                INSERT INTO _table_catalog
                (
                    table_name,
                    doc_name,
                    col_names,
                    row_labels,
                    section_hint,
                    page_number,
                    statement_type,
                    first_col,
                    year_columns,
                    evidence_id,
                    citation_label,
                    unit_scale,
                    is_primary_statement,
                    document_id,
                    workspace_id,
                    content_hash,
                    ingestion_version,
                    is_active
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    row.table_name,
                    row.doc_name,
                    row.col_names,
                    row.row_labels,
                    row.section_hint,
                    row.page_number,
                    row.statement_type,
                    row.first_col,
                    row.year_columns,
                    row.evidence_id,
                    row.citation_label,
                    row.unit_scale,
                    row.is_primary_statement,
                    str(row.document_id),
                    str(row.workspace_id),
                    row.content_hash,
                    int(row.ingestion_version),
                    bool(row.is_active),
                ],
            )

    async def deactivate_document_tables(
        self,
        document_id: UUID | str,
    ) -> int:
        """
        Mark all catalog rows for a document as inactive.

        Returns number of affected active catalog rows.
        """
        async with self._manager.write_lock:
            return await asyncio.to_thread(
                self._deactivate_document_tables_sync,
                document_id,
            )

    def _deactivate_document_tables_sync(
        self,
        document_id: UUID | str,
    ) -> int:
        with self._manager.connect_writer() as conn:
            active_count = conn.execute(
                """
                SELECT COUNT(*)
                FROM _table_catalog
                WHERE document_id = ?
                  AND is_active = true
                """,
                [str(document_id)],
            ).fetchone()[0]

            conn.execute(
                """
                UPDATE _table_catalog
                SET is_active = false
                WHERE document_id = ?
                  AND is_active = true
                """,
                [str(document_id)],
            )

            return int(active_count)