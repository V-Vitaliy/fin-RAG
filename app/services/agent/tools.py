from __future__ import annotations

import asyncio
import ast
import json
import logging
import operator
import re
from typing import Any
from uuid import UUID

import duckdb

from app.services.agent.metric_contracts import get_metric_contract, compact_contract
from app.use_cases.retrieval import RetrieveDocumentsUseCase

logger = logging.getLogger(__name__)


class BaseAgentTool:
    name: str

    def get_schema(self) -> dict[str, Any]:
        raise NotImplementedError

    async def execute(self, **kwargs) -> str:
        raise NotImplementedError


class SearchTextTool(BaseAgentTool):
    name = "search_text"

    def __init__(
        self,
        *,
        retrieval_use_case: RetrieveDocumentsUseCase,
        workspace_id: UUID,
        requested_document_ids: list[UUID | str] | None,
        top_k: int = 8,
    ):
        self.retrieval_use_case = retrieval_use_case
        self.workspace_id = workspace_id
        self.requested_document_ids = requested_document_ids
        self.top_k = int(top_k)

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Search filing text for narrative evidence, explanations, reasons, drivers, legal/proceedings text, "
                    "and nearby references to financial tables. Use this for qualitative questions and driver questions."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The text concept, fact, explanation, or driver to search for.",
                        },
                        "query_rewrites": {
                            "type": "array",
                            "items": {"type": "string"},
                            "maxItems": 3,
                            "description": (
                                "Optional paraphrases of the same search intent. Use generic financial/MD&A wording only. "
                                "Do not include expected answers, exact table IDs, page numbers, or unsupported facts."
                            ),
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    async def execute(
        self,
        *,
        query: str,
        query_rewrites: list[str] | None = None,
        **_: Any,
    ) -> str:
        return await self.retrieval_use_case.search_text(
            workspace_id=self.workspace_id,
            query=query,
            requested_document_ids=self.requested_document_ids,
            top_k=self.top_k,
            expand_neighbors=True,
            neighbor_window=1,
            extra_queries=query_rewrites,
        )


class SearchTablesTool(BaseAgentTool):
    name = "search_tables"

    def __init__(
        self,
        *,
        retrieval_use_case: RetrieveDocumentsUseCase,
        workspace_id: UUID,
        requested_document_ids: list[UUID | str] | None,
        top_k: int = 8,
    ):
        self.retrieval_use_case = retrieval_use_case
        self.workspace_id = workspace_id
        self.requested_document_ids = requested_document_ids
        self.top_k = int(top_k)

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Find relevant extracted financial tables. Returns [T#] table metadata blocks with exact SQL table names, "
                    "columns, row-label guidance, statement type, source page, and evidence IDs."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "concept": {
                            "type": "string",
                            "description": "Financial table concept to find, e.g. income statement, balance sheet, cash flow, revenue, liabilities.",
                        },
                        "concept_rewrites": {
                            "type": "array",
                            "items": {"type": "string"},
                            "maxItems": 3,
                            "description": (
                                "Optional paraphrases of the same table-search intent. Use generic statement/metric/table synonyms only. "
                                "Do not include exact table IDs, row IDs, page numbers, or answer facts."
                            ),
                        },
                    },
                    "required": ["concept"],
                },
            },
        }

    async def execute(
        self,
        *,
        concept: str,
        concept_rewrites: list[str] | None = None,
        **_: Any,
    ) -> str:
        return await self.retrieval_use_case.search_tables(
            workspace_id=self.workspace_id,
            concept=concept,
            requested_document_ids=self.requested_document_ids,
            top_k=self.top_k,
            extra_queries=concept_rewrites,
        )


class GetFinancialFormulaTool(BaseAgentTool):
    name = "get_financial_formula"

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Provides financial metric formulas and structured retrieval contracts. "
                    "Use this for ratios and metrics such as ROA, Quick Ratio, Inventory Turnover, "
                    "Capital Intensity, margins, FCF, and Debt-to-Equity."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "metric_name": {"type": "string"},
                        "structured": {
                            "type": "boolean",
                            "description": "If true, return a JSON metric contract instead of prose.",
                        },
                    },
                    "required": ["metric_name"],
                },
            },
        }

    async def execute(
        self,
        *,
        metric_name: str,
        structured: bool = False,
        **_: Any,
    ) -> str:
        contract = get_metric_contract(metric_name)
        if contract:
            compact = compact_contract(contract)

            if structured:
                return json.dumps(compact, ensure_ascii=False, indent=2)

            lines = [
                f"Formula for {compact['display_name'].upper()}:",
                f"Formula: {compact['formula']}",
            ]

            if compact.get("calculation_steps"):
                lines.append("Calculation steps:")
                lines.extend(f"- {step}" for step in compact["calculation_steps"])

            required = compact.get("required_components", [])
            if required:
                lines.append("Required components to retrieve from the filing:")
                for comp in required:
                    optional = " (optional if not present)" if comp.get("optional") else ""
                    aliases = ", ".join(comp.get("aliases", [])[:5])
                    stmt = comp.get("preferred_statement", "relevant statement")
                    lines.append(
                        f"- {comp.get('name')}{optional}: aliases [{aliases}], preferred source: {stmt}"
                    )

            risky = compact.get("forbidden_or_risky_components", [])
            if risky:
                lines.append("Forbidden or risky substitutions:")
                for item in risky:
                    lines.append(f"- {item.get('term')}: {item.get('reason')}")

            searches = compact.get("suggested_searches", [])
            if searches:
                lines.append("Suggested searches:")
                lines.extend(f"- {s}" for s in searches[:5])

            output = compact.get("output_contract", {})
            if output:
                lines.append(f"Output contract: {json.dumps(output, ensure_ascii=False)}")

            return "\n".join(lines)

        return (
            "Formula not in library. Use standard accounting definitions:\n"
            "- Ratios: numerator / denominator from the appropriate financial statement.\n"
            "- Always check for negative numbers: parentheses in financial tables usually mean negative values.\n"
            "- Retrieve all components before calculating; do not substitute broader metrics unless explicitly stated."
        )


class CalculateTool(BaseAgentTool):
    name = "calculate"

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Safe math calculator. Call this for arithmetic. "
                    "Financial values in parentheses like (546) mean NEGATIVE; write them as -546."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Arithmetic expression using numbers and + - * / parentheses.",
                        }
                    },
                    "required": ["expression"],
                },
            },
        }

    @staticmethod
    def _normalize_financial_expression(expr: str) -> str:
        expr = str(expr or "")
        expr = expr.replace("$", "").replace("£", "").replace("€", "")

        expr = re.sub(r"([\d.]+)\s*%", r"(\1/100)", expr)

        def replace_finance_negative(match: re.Match) -> str:
            inner = match.group(1).replace(",", "")
            return f"-{inner}"

        expr = re.sub(r"\((\d[\d,]*(?:\.\d+)?)\)", replace_finance_negative, expr)
        expr = re.sub(r"(\d),(\d)", r"\1\2", expr)
        expr = re.sub(r"(\d),(\d)", r"\1\2", expr)
        expr = re.sub(r"(\d)\s*[MB]\b", r"\1", expr, flags=re.IGNORECASE)

        return expr.strip()

    @staticmethod
    def _safe_eval(expr: str) -> float:
        allowed_binary = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
        }
        allowed_unary = {
            ast.UAdd: operator.pos,
            ast.USub: operator.neg,
        }

        def eval_node(node: ast.AST) -> float:
            if isinstance(node, ast.Expression):
                return eval_node(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
                return float(node.value)
            if isinstance(node, ast.BinOp) and type(node.op) in allowed_binary:
                return float(allowed_binary[type(node.op)](eval_node(node.left), eval_node(node.right)))
            if isinstance(node, ast.UnaryOp) and type(node.op) in allowed_unary:
                return float(allowed_unary[type(node.op)](eval_node(node.operand)))
            raise ValueError(f"Unsupported expression node: {type(node).__name__}")

        tree = ast.parse(expr, mode="eval")
        return eval_node(tree)

    async def execute(self, *, expression: str, **_: Any) -> str:
        original = expression
        try:
            normalized = self._normalize_financial_expression(expression)

            allowed_chars = set("0123456789+-*/(). ")
            if not all(char in allowed_chars for char in normalized):
                bad = sorted({char for char in normalized if char not in allowed_chars})
                return (
                    f"Error: unexpected characters after normalization: {bad}. "
                    f"Original: {original!r}, normalized: {normalized!r}."
                )

            result = self._safe_eval(normalized)
            return str(round(float(result), 4))

        except ZeroDivisionError:
            return "Error: Division by zero. The denominator is 0; percentage change is undefined."
        except Exception as exc:
            return (
                f"Calculation error: {exc}. "
                f"Original expression: {original!r}, after normalization: {self._normalize_financial_expression(original)!r}."
            )


class DuckDBReadOnlyTool(BaseAgentTool):
    name = "execute_sql"

    FORBIDDEN_SQL = re.compile(
        r"\b(insert|update|delete|drop|alter|create|copy|attach|detach|install|load|pragma|export|import|call)\b",
        re.IGNORECASE,
    )

    def __init__(
        self,
        *,
        duckdb_path: str,
        allowed_document_ids: list[str],
        max_rows: int = 80,
    ):
        self.duckdb_path = duckdb_path
        self.allowed_document_ids = {str(document_id) for document_id in allowed_document_ids}
        self.max_rows = int(max_rows)

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Executes a read-only SELECT query on DuckDB. Use exact table names and columns returned by search_tables or describe_table. "
                    "Only SELECT / WITH ... SELECT is allowed."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                    },
                    "required": ["query"],
                },
            },
        }

    @staticmethod
    def _quote_ident(name: str) -> str:
        return '"' + str(name).replace('"', '""') + '"'

    @staticmethod
    def _extract_table_names(query: str) -> list[str]:
        pattern = re.compile(
            r"\b(?:from|join)\s+[\"'`]?([A-Za-z_][A-Za-z0-9_]*)[\"'`]?",
            re.IGNORECASE,
        )
        return list(dict.fromkeys(match.group(1) for match in pattern.finditer(query or "")))

    @classmethod
    def _validate_read_only_sql(cls, query: str) -> str | None:
        q = str(query or "").strip()

        if not q:
            return "Error: empty SQL query."

        if ";" in q.rstrip(";"):
            return "Error: multiple SQL statements are not allowed."

        q_no_tail_semicolon = q[:-1].strip() if q.endswith(";") else q

        if cls.FORBIDDEN_SQL.search(q_no_tail_semicolon):
            return "Error: only read-only SELECT queries are allowed."

        lowered = q_no_tail_semicolon.lower()
        if not (lowered.startswith("select") or lowered.startswith("with")):
            return "Error: only SELECT or WITH ... SELECT queries are allowed."

        return None

    def _catalog_rows_for_tables(self, conn, table_names: list[str]) -> dict[str, dict[str, Any]]:
        if not table_names:
            return {}

        placeholders = ", ".join("?" for _ in table_names)
        rows = conn.execute(
            f"""
            SELECT table_name, document_id, doc_name, page_number, statement_type,
                   citation_label, evidence_id, unit_scale, is_primary_statement
            FROM _table_catalog
            WHERE table_name IN ({placeholders})
              AND is_active = true
            """,
            table_names,
        ).fetchall()

        keys = [
            "table_name",
            "document_id",
            "doc_name",
            "page_number",
            "statement_type",
            "citation_label",
            "evidence_id",
            "unit_scale",
            "is_primary_statement",
        ]
        return {str(row[0]): dict(zip(keys, row)) for row in rows}

    def _guard_tables(self, conn, table_names: list[str]) -> str | None:
        if not table_names:
            return "Error: SQL query must reference at least one table from search_tables."

        if "_table_catalog" in {name.lower() for name in table_names}:
            return "Error: querying _table_catalog directly is not allowed."

        catalog = self._catalog_rows_for_tables(conn, table_names)

        missing = [table for table in table_names if table not in catalog]
        if missing:
            return (
                f"Error: table(s) not found in active table catalog: {', '.join(missing)}. "
                "Use search_tables first and query an exact table name from the [T#] result."
            )

        denied = [
            table
            for table, row in catalog.items()
            if str(row.get("document_id")) not in self.allowed_document_ids
        ]
        if denied:
            return (
                f"Error: table(s) not accessible for this request: {', '.join(denied)}. "
                "Use search_tables for documents allowed in the current request."
            )

        return None

    @staticmethod
    def _source_line(catalog_row: dict[str, Any] | None, table_name: str | None) -> str:
        if not catalog_row:
            return f"Source: table={table_name}" if table_name else ""

        citation = catalog_row.get("citation_label")
        if not citation:
            doc = catalog_row.get("doc_name")
            page = catalog_row.get("page_number")
            citation = f"{doc} p.{page}, table={catalog_row.get('table_name')}"

        parts = [
            f"Source: [{citation}]",
            f"table={catalog_row.get('table_name')}",
            f"document_id={catalog_row.get('document_id')}",
            f"statement_type={catalog_row.get('statement_type')}",
        ]
        if catalog_row.get("evidence_id"):
            parts.append(f"evidence_id={catalog_row.get('evidence_id')}")
        if catalog_row.get("unit_scale"):
            parts.append(f"unit_scale={catalog_row.get('unit_scale')}")
        if catalog_row.get("is_primary_statement") is not None:
            parts.append(f"primary_statement={catalog_row.get('is_primary_statement')}")

        return " | ".join(parts)

    def _format_sql_error_help(self, query: str, table_name: str | None, error: Exception) -> str:
        base = f"SQL Error: {error}\n"

        if not table_name:
            return base + "Fix your query and try again. Use exact table names from search_tables."

        try:
            with duckdb.connect(self.duckdb_path, read_only=True) as conn:
                guard = self._guard_tables(conn, [table_name])
                if guard:
                    return base + guard

                schema = conn.execute(f"DESCRIBE {self._quote_ident(table_name)}").fetchdf()
                columns = [str(col) for col in schema["column_name"].tolist()]

                first_col = columns[0] if columns else None
                try:
                    row = conn.execute(
                        "SELECT first_col FROM _table_catalog WHERE table_name = ? LIMIT 1",
                        [table_name],
                    ).fetchone()
                    if row and row[0]:
                        first_col = row[0]
                except Exception:
                    pass

                sample_labels = ""
                if first_col and first_col in columns:
                    try:
                        labels = conn.execute(
                            f"""
                            SELECT DISTINCT {self._quote_ident(first_col)}
                            FROM {self._quote_ident(table_name)}
                            WHERE {self._quote_ident(first_col)} IS NOT NULL
                            LIMIT 25
                            """
                        ).fetchdf()
                        if not labels.empty:
                            sample_labels = "\nAvailable row labels from first column:\n" + labels.to_string(index=False)
                    except Exception:
                        pass

                return (
                    base
                    + "Available columns in this table:\n  - "
                    + "\n  - ".join(columns[:60])
                    + ("\n  - ..." if len(columns) > 60 else "")
                    + sample_labels
                    + "\n\nRecovery rule: do not retry guessed column names. "
                      "Call describe_table(table_name=..., include_sample=true), or use one of the exact column names above."
                )
        except Exception:
            return base + "Fix your query and try again. Call describe_table to inspect exact columns."

    def _execute_sync(self, query: str) -> str:
        query = str(query or "").strip()
        query = re.sub(r"[\}\)]+\s*,?\s*[\}\)]*\s*$", "", query).strip()

        for _ in range(5):
            new_query = query.replace('\\"', '"')
            if new_query == query:
                break
            query = new_query

        validation_error = self._validate_read_only_sql(query)
        if validation_error:
            return validation_error

        table_names = self._extract_table_names(query)

        logger.info("Agent SQL tool executing query=%s", query)

        try:
            with duckdb.connect(self.duckdb_path, read_only=True) as conn:
                guard_error = self._guard_tables(conn, table_names)
                if guard_error:
                    return guard_error

                catalog = self._catalog_rows_for_tables(conn, table_names)
                source_lines = [
                    self._source_line(catalog.get(table), table)
                    for table in table_names
                    if catalog.get(table)
                ]

                df = conn.execute(query).fetchdf()

                if len(df) > self.max_rows:
                    df = df.head(self.max_rows)
                    truncated_note = f"\n\nResult truncated to first {self.max_rows} rows."
                else:
                    truncated_note = ""

                if df.empty:
                    return (
                        ("\n".join(source_lines) + "\n" if source_lines else "")
                        + "Query returned 0 rows. Check exact column names and WHERE conditions."
                    )

                return (
                    ("\n".join(source_lines) + "\n" if source_lines else "")
                    + df.to_markdown(index=False)
                    + truncated_note
                )

        except Exception as exc:
            first_table = table_names[0] if table_names else None
            logger.warning("Agent SQL tool error: %s", exc)
            return self._format_sql_error_help(query, first_table, exc)

    async def execute(self, *, query: str, **_: Any) -> str:
        return await asyncio.to_thread(self._execute_sync, query)


class DescribeTableTool(BaseAgentTool):
    name = "describe_table"

    DEFAULT_SAMPLE_ROWS = 80
    MAX_SAMPLE_ROWS = 120
    MAX_SAMPLE_COLS = 14
    LARGE_TABLE_PREVIEW_ROWS = 25

    def __init__(
        self,
        *,
        duckdb_path: str,
        allowed_document_ids: list[str],
    ):
        self.duckdb_path = duckdb_path
        self.allowed_document_ids = {str(document_id) for document_id in allowed_document_ids}

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Get schema, citation metadata, row/column counts, and a bounded sample/full snapshot of a table. "
                    "Use this before issuing multiple row-by-row SQL queries, after a column-not-found error, "
                    "or when comparing several rows/segments/years from one table."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "table_name": {"type": "string", "description": "Exact table name from search_tables."},
                        "include_sample": {
                            "type": "boolean",
                            "description": "If true, include bounded table snapshot/sample. Default: true.",
                        },
                        "sample_rows": {
                            "type": "integer",
                            "description": "Maximum rows to include. Default: 80.",
                        },
                    },
                    "required": ["table_name"],
                },
            },
        }

    @staticmethod
    def _quote_ident(name: str) -> str:
        return '"' + str(name).replace('"', '""') + '"'

    def _catalog_row(self, conn, table_name: str) -> dict[str, Any] | None:
        row = conn.execute(
            """
            SELECT table_name, document_id, doc_name, page_number, statement_type,
                   citation_label, evidence_id, unit_scale, is_primary_statement
            FROM _table_catalog
            WHERE table_name = ?
              AND is_active = true
            LIMIT 1
            """,
            [table_name],
        ).fetchone()

        if not row:
            return None

        keys = [
            "table_name",
            "document_id",
            "doc_name",
            "page_number",
            "statement_type",
            "citation_label",
            "evidence_id",
            "unit_scale",
            "is_primary_statement",
        ]
        return dict(zip(keys, row))

    def _guard_table(self, conn, table_name: str) -> str | None:
        row = self._catalog_row(conn, table_name)
        if not row:
            return (
                f"Error describing table: table {table_name!r} is not in active _table_catalog. "
                "Use search_tables first."
            )

        if str(row.get("document_id")) not in self.allowed_document_ids:
            return (
                f"Error describing table: table {table_name!r} is not accessible for this request. "
                "Use search_tables for allowed documents."
            )

        return None

    @staticmethod
    def _source_line(row: dict[str, Any]) -> str:
        citation = row.get("citation_label")
        if not citation:
            citation = f"{row.get('doc_name')} p.{row.get('page_number')}, table={row.get('table_name')}"

        parts = [
            f"Source: [{citation}]",
            f"table={row.get('table_name')}",
            f"document_id={row.get('document_id')}",
            f"statement_type={row.get('statement_type')}",
        ]
        if row.get("evidence_id"):
            parts.append(f"evidence_id={row.get('evidence_id')}")
        if row.get("unit_scale"):
            parts.append(f"unit_scale={row.get('unit_scale')}")
        if row.get("is_primary_statement") is not None:
            parts.append(f"primary_statement={row.get('is_primary_statement')}")

        return " | ".join(parts)

    def _execute_sync(
        self,
        *,
        table_name: str,
        include_sample: bool = True,
        sample_rows: int = DEFAULT_SAMPLE_ROWS,
    ) -> str:
        table_name = str(table_name or "").strip().strip('"`')

        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table_name):
            return "Error describing table: invalid table name. Use an exact SQL table ID from search_tables."

        try:
            sample_rows = int(sample_rows or self.DEFAULT_SAMPLE_ROWS)
        except Exception:
            sample_rows = self.DEFAULT_SAMPLE_ROWS

        sample_rows = max(1, min(sample_rows, self.MAX_SAMPLE_ROWS))

        try:
            with duckdb.connect(self.duckdb_path, read_only=True) as conn:
                guard_error = self._guard_table(conn, table_name)
                if guard_error:
                    return guard_error

                catalog_row = self._catalog_row(conn, table_name)
                assert catalog_row is not None

                schema_df = conn.execute(f"DESCRIBE {self._quote_ident(table_name)}").fetchdf()
                columns = [str(col) for col in schema_df["column_name"].tolist()]
                row_count = conn.execute(
                    f"SELECT COUNT(*) FROM {self._quote_ident(table_name)}"
                ).fetchone()[0]
                col_count = len(columns)

                out = [
                    self._source_line(catalog_row),
                    f"Table: {table_name} | rows={row_count} | columns={col_count}",
                    "\nSchema:",
                    schema_df.to_markdown(index=False),
                ]

                if include_sample:
                    preview_rows = min(sample_rows, row_count)
                    if row_count > sample_rows:
                        preview_rows = min(self.LARGE_TABLE_PREVIEW_ROWS, preview_rows)

                    sample_cols = columns[: self.MAX_SAMPLE_COLS]
                    select_expr = ", ".join(self._quote_ident(col) for col in sample_cols) or "*"

                    sample_df = conn.execute(
                        f"SELECT {select_expr} FROM {self._quote_ident(table_name)} LIMIT {preview_rows}"
                    ).fetchdf()

                    label = "Full bounded table snapshot" if row_count <= sample_rows else "Table preview"
                    out.append(
                        f"\n{label}: showing {len(sample_df)} of {row_count} rows"
                        + (
                            f", first {len(sample_cols)} of {col_count} columns"
                            if col_count > len(sample_cols)
                            else ""
                        )
                    )

                    if col_count > len(sample_cols):
                        out.append("Columns not shown in preview: " + ", ".join(columns[len(sample_cols):]))

                    out.append(sample_df.to_markdown(index=False))
                    out.append(
                        "\nGuidance: if the question compares multiple rows/segments/years in this table, "
                        "use this snapshot to build one batched SELECT or calculate directly. "
                        "Avoid many separate row-by-row queries against this same table."
                    )

                return "\n".join(part for part in out if part)

        except Exception as exc:
            logger.warning("Describe table tool error: %s", exc)
            return f"Error describing table: {exc}"

    async def execute(
        self,
        *,
        table_name: str,
        include_sample: bool = True,
        sample_rows: int = DEFAULT_SAMPLE_ROWS,
        **_: Any,
    ) -> str:
        return await asyncio.to_thread(
            self._execute_sync,
            table_name=table_name,
            include_sample=include_sample,
            sample_rows=sample_rows,
        )


def build_agent_tools(
    *,
    retrieval_use_case: RetrieveDocumentsUseCase,
    workspace_id: UUID,
    requested_document_ids: list[UUID | str] | None,
    allowed_document_ids: list[str],
    duckdb_path: str,
    retrieval_top_k: int = 8,
) -> dict[str, BaseAgentTool]:
    return {
        "search_text": SearchTextTool(
            retrieval_use_case=retrieval_use_case,
            workspace_id=workspace_id,
            requested_document_ids=requested_document_ids,
            top_k=retrieval_top_k,
        ),
        "search_tables": SearchTablesTool(
            retrieval_use_case=retrieval_use_case,
            workspace_id=workspace_id,
            requested_document_ids=requested_document_ids,
            top_k=retrieval_top_k,
        ),
        "describe_table": DescribeTableTool(
            duckdb_path=duckdb_path,
            allowed_document_ids=allowed_document_ids,
        ),
        "execute_sql": DuckDBReadOnlyTool(
            duckdb_path=duckdb_path,
            allowed_document_ids=allowed_document_ids,
        ),
        "get_financial_formula": GetFinancialFormulaTool(),
        "calculate": CalculateTool(),
    }