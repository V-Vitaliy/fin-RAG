from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

import pandas as pd
from collections.abc import Awaitable, Callable
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from app.repositories.duckdb_repo import DuckDBCatalogRow, DuckDBTableRepository
from app.repositories.qdrant_repo import VectorIndexRepository
from app.services.ingestion.naming import make_table_name
from app.services.ingestion.vector_points import build_qdrant_point

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*export_to_dataframe.*")
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class IngestionDocument:
    pdf_path: str
    document_id: UUID | str
    workspace_id: UUID | str
    filename: str
    doc_name: str
    content_hash: str | None = None
    ingestion_version: int = 1


@dataclass(frozen=True)
class DocumentIngestionResult:
    document_id: str
    doc_name: str
    filename: str
    qdrant_points_count: int
    duckdb_tables_count: int


@dataclass(frozen=True)
class IngestionResult:
    documents_count: int
    qdrant_points_count: int
    duckdb_tables_count: int
    documents: list[DocumentIngestionResult]

IngestionEventSink = Callable[[str, dict[str, Any]], Awaitable[None]]



def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _year_columns(columns: list[str]) -> list[str]:
    result = []
    for col in columns:
        col_s = str(col)
        if re.search(r"(19|20)\d{2}", col_s) or re.search(
            r"december|january|june|september|fiscal|year",
            col_s,
            re.I,
        ):
            result.append(col_s)
    return result


def _text_hash(text: str) -> str:
    return hashlib.sha1(str(text or "").encode("utf-8", errors="ignore")).hexdigest()[:16]


def _citation_label(
    doc_name: str,
    page_number: int | None,
    table_name: str | None = None,
) -> str:
    page_part = f" p.{page_number}" if page_number is not None else ""
    table_part = f", table={table_name}" if table_name else ""
    return f"{doc_name}{page_part}{table_part}"


def _dense_metadata_prefix(meta: dict[str, Any]) -> str:
    """
    Build a short contextual prefix embedded together with chunk text.

    BM25 still uses raw/sparse text, so exact lexical matching is not drowned
    in metadata boilerplate.
    """
    fields = [
        ("Document", meta.get("doc_name")),
        ("Filename", meta.get("filename")),
        ("Page", meta.get("page_number")),
        ("Source type", meta.get("source_type")),
        ("Section path", meta.get("section_path") or meta.get("section")),
        ("Table", meta.get("table_name")),
        ("Statement type", meta.get("statement_type")),
        ("Unit scale", meta.get("unit_scale")),
        ("Primary statement", meta.get("is_primary_statement")),
        ("Citation", meta.get("citation_label")),
    ]

    lines = []
    for label, value in fields:
        if value is None or value == "":
            continue
        lines.append(f"{label}: {value}")

    return "\n".join(lines)


def _infer_unit_scale(*texts: str) -> str | None:
    text = " ".join(str(t or "") for t in texts).lower()
    text = re.sub(r"[_\-]+", " ", text)

    if re.search(r"\b(in|amounts? in|dollars? in|usd)\s+billions?\b|\$\s*in\s*billions?", text):
        return "billions"
    if re.search(r"\b(in|amounts? in|dollars? in|usd)\s+millions?\b|\$\s*in\s*millions?", text):
        return "millions"
    if re.search(r"\b(in|amounts? in|dollars? in|usd)\s+thousands?\b|\$\s*in\s*thousands?", text):
        return "thousands"

    return None


def _infer_is_primary_statement(
    statement_type: str,
    row_labels: str,
    col_names: str,
    table_label: str = "",
) -> bool:
    """
    Conservative soft flag.

    False does not mean "irrelevant"; it only means "not confidently primary".
    """
    st = (statement_type or "").lower()
    text = f"{row_labels} {col_names} {table_label}".lower()

    if st == "balance_sheet":
        return (
            "total assets" in text
            and (
                "total liabilities" in text
                or "stockholders" in text
                or "shareholders" in text
                or "total equity" in text
            )
            and not any(
                x in text
                for x in [
                    "derivative",
                    "fair value hierarchy",
                    "level 1",
                    "level 2",
                    "level 3",
                ]
            )
        )

    if st == "income_statement":
        return (
            any(
                x in text
                for x in [
                    "net sales",
                    "net revenue",
                    "total revenue",
                    "sales to customers",
                    "revenues",
                ]
            )
            and any(x in text for x in ["net income", "net earnings", "net loss"])
            and not any(x in text for x in ["reconciliation", "non-gaap", "adjusted"])
        )

    if st == "cash_flow_statement":
        return (
            any(
                x in text
                for x in [
                    "operating activities",
                    "net cash provided by operating",
                    "cash flows from operating",
                ]
            )
            and any(x in text for x in ["investing activities", "financing activities"])
        )

    return False


def _extract_table_page_number(table: Any) -> int | None:
    try:
        if hasattr(table, "prov") and table.prov:
            return int(table.prov[0].page_no)
    except Exception:
        return None

    return None


def _export_table_dataframe(table: Any, docling_document: Any) -> pd.DataFrame:
    try:
        return table.export_to_dataframe(doc=docling_document)
    except TypeError:
        return table.export_to_dataframe()


def _safe_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    raw_cols = [_normalize_text(c) for c in df.columns.tolist()]
    seen_exact: dict[str, int] = {}
    safe_cols: list[str] = []

    for idx, col in enumerate(raw_cols):
        name = col if col else f"column_{idx}"
        name = re.sub(r"[\s.,()]+", "_", str(name)).strip("_")
        name = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")

        if not name:
            name = f"column_{idx}"

        if re.match(r"^\d", name):
            name = f"col_{name}"

        if name in seen_exact:
            seen_exact[name] += 1
            name = f"{name}_{seen_exact[name]}"
        else:
            seen_exact[name] = 0

        safe_cols.append(name)

    result = df.copy()
    result.columns = safe_cols
    return result


def _fix_markdown_tables_split_across_pages(
    markdown: str,
    page_break: str,
) -> list[str]:
    """
    Repair markdown tables that Docling may split around page breaks.

    Handles two common cases:
    1. The page break is directly inside a markdown table.
    2. The previous page ends with a table and the next page starts with
       optional headings/blank lines followed by table rows.
    """
    md_text = re.sub(
        r"(\|\s*)\n+" + re.escape(page_break) + r"\n+(\s*\|)",
        r"\1\n\2",
        markdown,
    )
    pages = [page.strip() for page in md_text.split(page_break)]

    def is_table_row(line: str) -> bool:
        stripped = line.strip()
        return stripped.startswith("|") and stripped.endswith("|") and "|" in stripped[1:-1]

    def is_markdown_heading(line: str) -> bool:
        return bool(re.match(r"^#{1,6}\s+", line.strip()))

    def find_first_table_row_index(lines: list[str]) -> int | None:
        for idx, line in enumerate(lines):
            if not line.strip():
                continue
            if is_markdown_heading(line):
                continue
            if is_table_row(line):
                return idx
            return None
        return None

    for i in range(len(pages) - 1):
        lines_current = pages[i].strip().split("\n")
        lines_next = pages[i + 1].strip().split("\n")

        if not lines_current or not lines_next:
            continue

        first_next_table_idx = find_first_table_row_index(lines_next)
        if first_next_table_idx is None:
            continue

        if not is_table_row(lines_current[-1]):
            continue

        table_start_idx = len(lines_current) - 1
        while table_start_idx >= 0 and is_table_row(lines_current[table_start_idx]):
            table_start_idx -= 1
        table_start_idx += 1

        hanging_table = "\n".join(lines_current[table_start_idx:]).strip()
        before_table = "\n".join(lines_current[:table_start_idx]).strip()

        next_prefix = "\n".join(lines_next[:first_next_table_idx]).strip()
        next_table_and_rest = "\n".join(lines_next[first_next_table_idx:]).strip()

        pages[i] = before_table

        merged_next_parts = [
            part
            for part in [next_prefix, hanging_table, next_table_and_rest]
            if part
        ]
        pages[i + 1] = "\n".join(merged_next_parts)

    return [page for page in pages if page.strip()]


class IngestionPipeline:
    """
    Backend-oriented ingestion pipeline.

    This class coordinates:
    - Docling PDF parsing
    - financial table extraction
    - DuckDB table/catalog writes through DuckDBTableRepository
    - narrative chunking
    - dense/sparse embeddings
    - Qdrant point building/upsert through VectorIndexRepository

    Blocking libraries are isolated with asyncio.to_thread().
    """

    def __init__(
        self,
        *,
        vector_repo: VectorIndexRepository,
        duckdb_repo: DuckDBTableRepository,
        chunker: Any,
        dense_embedder: Any,
        sparse_embedder: Any,
        qdrant_batch_size: int = 300,
    ):
        self.vector_repo = vector_repo
        self.duckdb_repo = duckdb_repo
        self.chunker = chunker
        self.dense_embedder = dense_embedder
        self.sparse_embedder = sparse_embedder
        self.qdrant_batch_size = int(qdrant_batch_size or 300)

        opts = PdfPipelineOptions()
        opts.do_ocr = False
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=opts),
            }
        )

    async def run(
        self,
        docs: list[dict[str, Any] | IngestionDocument],
        page_break: str = "---PAGE_BREAK---",
    ) -> IngestionResult:
        logger.info("Starting backend ingestion for %s documents.", len(docs))

        await self._ensure_storage_ready()

        document_results: list[DocumentIngestionResult] = []

        for raw_doc in docs:
            document = self._coerce_document(raw_doc)
            result = await self.ingest_document(document, page_break=page_break)
            document_results.append(result)

        return IngestionResult(
            documents_count=len(document_results),
            qdrant_points_count=sum(item.qdrant_points_count for item in document_results),
            duckdb_tables_count=sum(item.duckdb_tables_count for item in document_results),
            documents=document_results,
        )

    async def _ensure_storage_ready(self) -> None:
        await self.vector_repo.ensure_collection(
            dense_vector_size=int(self.dense_embedder.vector_size)
        )
        await self.vector_repo.ensure_payload_indexes()
        await self.duckdb_repo.ensure_catalog_schema()


    async def ingest_document(
        self,
        document: IngestionDocument,
        event_sink: IngestionEventSink | None = None,
        page_break: str = "---PAGE_BREAK---",
    ) -> DocumentIngestionResult:

        if not os.path.exists(document.pdf_path):
            raise FileNotFoundError(f"PDF file does not exist: {document.pdf_path}")

        async def emit(stage: str, message: str, **extra: Any) -> None:
            if event_sink is not None:
                await event_sink(
                    "stage",
                    {
                        "stage": stage,
                        "message": message,
                        **extra,
                    },
                )

        await emit("storage_ready", "Preparing vector/table storage")
        await self._ensure_storage_ready()

        logger.info(
            "Processing document_id=%s doc_name=%s filename=%s",
            document.document_id,
            document.doc_name,
            document.filename,
        )

        await emit("cleanup_previous_index", "Cleaning previous index state")
        await self.vector_repo.delete_points_by_document_id(document.document_id)
        await self.duckdb_repo.deactivate_document_tables(document.document_id)

        await emit("parsing", "Parsing PDF")
        doc_res = await asyncio.to_thread(self.converter.convert, document.pdf_path)

        doc_chunks: list[dict[str, Any]] = []
        await emit("tables", "Extracting financial tables")
        table_count = await self._extract_tables_to_duckdb_and_stubs(
            doc_res=doc_res,
            document=document,
            doc_chunks=doc_chunks,
        )

        await emit("chunking", "Extracting narrative chunks")
        narrative_chunks = await self._extract_narrative_chunks(
            doc_res=doc_res,
            document=document,
            page_break=page_break,
        )
        doc_chunks.extend(narrative_chunks)

        if not doc_chunks:
            logger.warning("No chunks produced for document_id=%s", document.document_id)
            return DocumentIngestionResult(
                document_id=str(document.document_id),
                doc_name=document.doc_name,
                filename=document.filename,
                qdrant_points_count=0,
                duckdb_tables_count=table_count,
            )

        await emit("metadata", "Assigning chunk metadata")
        self._assign_chunk_metadata(document=document, chunks=doc_chunks)

        await emit(
            "embedding_indexing",
            "Creating embeddings and writing vector index",
            chunks_count=len(doc_chunks),
        )
        points_count = await self._embed_and_upsert_chunks(
            document=document,
            chunks=doc_chunks,
        )

        logger.info(
            "Completed ingestion for document_id=%s: tables=%s qdrant_points=%s",
            document.document_id,
            table_count,
            points_count,
        )

        await emit(
            "ingestion_completed",
            "Document ingestion completed",
            qdrant_points_count=points_count,
            duckdb_tables_count=table_count,
        )

        return DocumentIngestionResult(
            document_id=str(document.document_id),
            doc_name=document.doc_name,
            filename=document.filename,
            qdrant_points_count=points_count,
            duckdb_tables_count=table_count,
        )

    def _coerce_document(
        self,
        raw_doc: dict[str, Any] | IngestionDocument,
    ) -> IngestionDocument:
        if isinstance(raw_doc, IngestionDocument):
            return raw_doc

        pdf_path = str(raw_doc.get("pdf_path") or "")
        if not pdf_path:
            raise ValueError("Ingestion document is missing required field: pdf_path")

        document_id = raw_doc.get("document_id")
        if not document_id:
            raise ValueError("Ingestion document is missing required field: document_id")

        workspace_id = raw_doc.get("workspace_id")
        if not workspace_id:
            raise ValueError("Ingestion document is missing required field: workspace_id")

        filename = str(raw_doc.get("filename") or Path(pdf_path).name)
        doc_name = str(raw_doc.get("doc_name") or Path(filename).stem)

        return IngestionDocument(
            pdf_path=pdf_path,
            document_id=document_id,
            workspace_id=workspace_id,
            filename=filename,
            doc_name=doc_name,
            content_hash=raw_doc.get("content_hash"),
            ingestion_version=int(raw_doc.get("ingestion_version") or 1),
        )

    def _classify_statement_type(
        self,
        row_labels: str,
        col_names: str,
        table_label: str = "",
    ) -> str:
        """
        Robust scoring heuristic for table classification.
        """
        text = f"{row_labels} {col_names} {table_label}".lower()

        bridge_signals = [
            "before segment change",
            "after segment change",
            "before reportable segment",
            "after reportable segment",
            "adjustments and reassignment",
            "reclassification",
            "segment realignment",
            "restated segment",
        ]

        has_bridge_text = any(sig in text for sig in bridge_signals)
        has_acquisition_adjustment_cols = "acquisition" in text and "adjustment" in text

        if has_bridge_text or has_acquisition_adjustment_cols:
            return "segment_reconciliation_bridge"

        scores = {
            "cash_flow_statement": 0,
            "balance_sheet": 0,
            "income_statement": 0,
            "segment_revenue": 0,
            "debt_schedule": 0,
            "fair_value_measurements": 0,
        }

        if any(
            x in text
            for x in [
                "cash flow",
                "cash flows",
                "operating activities",
                "investing activities",
            ]
        ):
            scores["cash_flow_statement"] += 3

        if any(
            x in text
            for x in [
                "balance sheet",
                "assets",
                "liabilities",
                "stockholders",
                "financial position",
            ]
        ):
            scores["balance_sheet"] += 3

        if any(
            x in text
            for x in [
                "income statement",
                "statement of operations",
                "net revenue",
                "gross profit",
                "net income",
                "operating income",
            ]
        ):
            scores["income_statement"] += 3

        if any(x in text for x in ["segment", "geographic", "revenue by", "sales by"]):
            scores["segment_revenue"] += 2

        if "debt" in text and any(x in text for x in ["maturity", "principal", "interest"]):
            scores["debt_schedule"] += 2

        if "fair value" in text:
            scores["fair_value_measurements"] += 2

        best_match = max(scores, key=scores.get)
        return best_match if scores[best_match] > 0 else "financial_notes"

    async def _extract_tables_to_duckdb_and_stubs(
        self,
        *,
        doc_res: Any,
        document: IngestionDocument,
        doc_chunks: list[dict[str, Any]],
    ) -> int:
        table_count = 0
        tables = getattr(doc_res.document, "tables", []) or []

        for table_index, table in enumerate(tables):
            try:
                df = await asyncio.to_thread(
                    _export_table_dataframe,
                    table,
                    doc_res.document,
                )

                if df is None or df.empty:
                    continue

                df = _safe_dataframe_columns(df)

                if df.empty or not df.columns.tolist():
                    continue

                page_no = _extract_table_page_number(table)
                table_name = make_table_name(
                    document_id=document.document_id,
                    doc_name=document.doc_name,
                    table_index=table_index,
                    page_number=page_no,
                )

                await self.duckdb_repo.write_dataframe_table(
                    table_name=table_name,
                    dataframe=df,
                    replace=True,
                )

                col_names_list = [str(col) for col in df.columns.tolist()]
                col_names = ", ".join(col_names_list)
                first_col = col_names_list[0]
                year_cols = _year_columns(col_names_list)

                row_labels_list = (
                    df[first_col]
                    .dropna()
                    .astype(str)
                    .map(_normalize_text)
                    .drop_duplicates()
                    .head(200)
                    .tolist()
                )
                row_labels_str = ", ".join(row_labels_list)
                row_labels_bullets = "\n- ".join(row_labels_list)

                raw_label = str(table.label) if hasattr(table, "label") else ""
                statement_type = self._classify_statement_type(
                    row_labels_str,
                    col_names,
                    raw_label,
                )
                unit_scale = _infer_unit_scale(raw_label, col_names, row_labels_str)
                is_primary_statement = _infer_is_primary_statement(
                    statement_type,
                    row_labels_str,
                    col_names,
                    raw_label,
                )
                evidence_id = table_name
                citation_label = _citation_label(document.doc_name, page_no, table_name)

                await self.duckdb_repo.insert_catalog_row(
                    DuckDBCatalogRow(
                        table_name=table_name,
                        doc_name=document.doc_name,
                        col_names=col_names,
                        row_labels=row_labels_str,
                        section_hint=raw_label,
                        page_number=page_no,
                        statement_type=statement_type,
                        first_col=first_col,
                        year_columns=", ".join(year_cols),
                        evidence_id=evidence_id,
                        citation_label=citation_label,
                        unit_scale=unit_scale,
                        is_primary_statement=is_primary_statement,
                        document_id=document.document_id,
                        workspace_id=document.workspace_id,
                        content_hash=document.content_hash,
                        ingestion_version=document.ingestion_version,
                        is_active=True,
                    )
                )

                table_stub = self._build_table_stub_chunk(
                    document=document,
                    table_name=table_name,
                    page_no=page_no,
                    statement_type=statement_type,
                    first_col=first_col,
                    col_names=col_names,
                    value_col_names=[col for col in col_names_list if col != first_col],
                    row_labels_list=row_labels_list,
                    row_labels_bullets=row_labels_bullets,
                    citation_label=citation_label,
                    evidence_id=evidence_id,
                    unit_scale=unit_scale,
                    is_primary_statement=is_primary_statement,
                )
                doc_chunks.append(table_stub)

                table_count += 1

            except Exception as exc:
                logger.warning("Failed to process table %s: %s", table_index, exc)

        return table_count

    def _build_table_stub_chunk(
        self,
        *,
        document: IngestionDocument,
        table_name: str,
        page_no: int | None,
        statement_type: str,
        first_col: str,
        col_names: str,
        value_col_names: list[str],
        row_labels_list: list[str],
        row_labels_bullets: str,
        citation_label: str,
        evidence_id: str,
        unit_scale: str | None,
        is_primary_statement: bool,
    ) -> dict[str, Any]:
        rows_count = len(row_labels_list)
        is_small = rows_count <= 25

        if is_small:
            select_hint = "Use SELECT * without WHERE (small table, ≤25 rows)"
            example_sql = f'SELECT * FROM "{table_name}"'
        else:
            select_hint = f'Use WHERE "{first_col}" ILIKE \'%<pattern>%\''
            sample_pattern = row_labels_list[0][:20] if row_labels_list else "Total"
            example_sql = (
                f'SELECT * FROM "{table_name}" '
                f'WHERE "{first_col}" ILIKE \'%{sample_pattern}%\''
            )

        value_cols_display = ", ".join(value_col_names) if value_col_names else "(none)"

        col_select_example = ""
        if value_col_names:
            sample_col = value_col_names[0]
            row_sample = row_labels_list[0][:25] if row_labels_list else "Total"
            col_select_example = (
                f"\nExample selecting one value column: "
                f'SELECT "{first_col}", "{sample_col}" FROM "{table_name}" '
                f'WHERE "{first_col}" ILIKE \'%{row_sample}%\''
            )

        stub_text = (
            f"[FINANCIAL TABLE METADATA]\n"
            f"Table ID (SQL name): {table_name}\n"
            f"Document: {document.doc_name} | Filename: {document.filename} | Page: {page_no}\n"
            f"Citation: {citation_label} | Evidence ID: {evidence_id}\n"
            f"Type: {statement_type} | Primary Statement: {is_primary_statement} | "
            f"Unit Scale: {unit_scale or 'unknown'}\n"
            f"Row Labels Column: \"{first_col}\" ← use this EXACT column name in WHERE clause with ILIKE\n"
            f"Value Columns (use these EXACT names in SELECT — do not abbreviate or rename):\n"
            f"  {value_cols_display}\n"
            f"Rows: {rows_count} | Query hint: {select_hint}\n"
            f"Example query: {example_sql}"
            f"{col_select_example}\n"
            f"Categories in this table:\n- {row_labels_bullets}"
        )

        if statement_type == "segment_reconciliation_bridge":
            stub_text += (
                "\n\n⚠️ WARNING — THIS IS A RECONCILIATION/BRIDGE TABLE, NOT A "
                "YEAR-OVER-YEAR COMPARISON.\n"
                "The columns represent transformation steps, not independent fiscal "
                "years to compare. Do NOT use this table to answer questions about "
                "year-over-year segment growth or YoY revenue change. For actual "
                "annual segment revenue by fiscal year, search_tables for a different "
                "table explicitly showing fiscal year columns."
            )

        key_row_keywords = [
            "total",
            "net",
            "operating",
            "gross",
            "income",
            "revenue",
            "sales",
            "provided by",
            "used in",
            "investing",
            "financing",
            "assets",
            "liabilities",
            "equity",
            "earnings",
            "expense",
            "margin",
            "depreciation",
            "amortization",
            "capital expenditure",
        ]
        key_rows = [
            row
            for row in row_labels_list
            if any(keyword in row.lower() for keyword in key_row_keywords)
        ]
        if key_rows:
            stub_text += (
                "\nKey rows — use these EXACT strings in ILIKE:\n"
                + "\n".join(f'  → "{row}"' for row in key_rows[:10])
            )

        return {
            "text": stub_text,
            "text_bm25": stub_text,
            "is_table_stub": True,
            "source_type": "table_stub",
            "table_name": table_name,
            "statement_type": statement_type,
            "col_names": col_names,
            "doc_name": document.doc_name,
            "filename": document.filename,
            "document_id": str(document.document_id),
            "workspace_id": str(document.workspace_id),
            "page_number": page_no,
            "citation_label": citation_label,
            "evidence_id": evidence_id,
            "unit_scale": unit_scale,
            "is_primary_statement": is_primary_statement,
            "text_hash": _text_hash(stub_text),
            "section": "Financial Table Metadata",
            "section_path": f"Financial Table Metadata > {statement_type}",
            "section_level": 0,
            "chunk_id": f"{table_name}:stub",
            "content_hash": document.content_hash,
            "ingestion_version": document.ingestion_version,
        }

    async def _extract_narrative_chunks(
        self,
        *,
        doc_res: Any,
        document: IngestionDocument,
        page_break: str,
    ) -> list[dict[str, Any]]:
        md_text = await asyncio.to_thread(
            doc_res.document.export_to_markdown,
            page_break_placeholder=page_break,
        )

        pages = _fix_markdown_tables_split_across_pages(md_text, page_break)

        chunks: list[dict[str, Any]] = []
        current_section: str | None = None

        for page_idx, page_md in enumerate(pages, start=1):
            if not page_md.strip():
                continue

            page_md = page_md.strip()
            first_line = page_md.split("\n")[0]

            if not re.match(r"^#{1,6}\s", first_line):
                if current_section:
                    page_md = f"{current_section}\n\n{page_md}"
            else:
                current_section = first_line

            page_chunks = await asyncio.to_thread(
                self.chunker.chunk_document,
                markdown=page_md,
                doc_name=document.doc_name,
                page_number=page_idx,
            )

            if page_chunks:
                chunks.extend(page_chunks)

        return chunks

    def _assign_chunk_metadata(
            self,
            *,
            document: IngestionDocument,
            chunks: list[dict[str, Any]],
    ) -> None:
    
        local_chunk_index = 0

        for chunk_index, chunk in enumerate(chunks):
            page_number = chunk.get("page_number")

            chunk["document_id"] = str(document.document_id)
            chunk["workspace_id"] = str(document.workspace_id)
            chunk["filename"] = document.filename
            chunk["doc_name"] = document.doc_name
            chunk["chunk_index"] = chunk_index
            chunk["content_hash"] = document.content_hash
            chunk["ingestion_version"] = document.ingestion_version

            is_real_table_stub = bool(chunk.get("is_table_stub") and chunk.get("table_name"))

            if is_real_table_stub:
                table_name = chunk["table_name"]

                chunk["source_type"] = "table_stub"
                chunk["chunk_id"] = f"{table_name}:stub"
                chunk["evidence_id"] = table_name
                chunk["local_chunk_index"] = None

                chunk.setdefault(
                    "citation_label",
                    _citation_label(document.doc_name, page_number, table_name),
                )

            else:
                was_table_like = bool(chunk.get("is_table_stub"))

                chunk["is_table_stub"] = False

                # Important fix:
                # override any page-local index produced by chunker.
                chunk["local_chunk_index"] = local_chunk_index
                local_chunk_index += 1

                if was_table_like:
                    chunk["source_type"] = "text_table_reference"
                else:
                    chunk.setdefault("source_type", "text")

                chunk["chunk_id"] = f"{document.document_id}:chunk:{chunk['local_chunk_index']}"
                chunk["evidence_id"] = chunk["chunk_id"]

                chunk.setdefault("citation_label", _citation_label(document.doc_name, page_number))
                chunk.setdefault("unit_scale", None)
                chunk.setdefault("is_primary_statement", False)

            chunk["text_hash"] = chunk.get("text_hash") or _text_hash(chunk.get("text", ""))

    async def _embed_and_upsert_chunks(
        self,
        *,
        document: IngestionDocument,
        chunks: list[dict[str, Any]],
    ) -> int:
        sparse_texts = [chunk.pop("text_bm25", chunk["text"]) for chunk in chunks]
        dense_texts = self._build_dense_texts(chunks)

        dense_vecs = await asyncio.to_thread(
            self.dense_embedder.embed_documents,
            dense_texts,
        )
        sparse_vecs = await asyncio.to_thread(
            lambda: list(self.sparse_embedder.embed(sparse_texts))
        )

        points = []
        total_points = 0

        for meta, sparse_text, dense_vec, sparse_vec in zip(
            chunks,
            sparse_texts,
            dense_vecs,
            sparse_vecs,
        ):
            sparse_indices = (
                sparse_vec.indices.tolist()
                if hasattr(sparse_vec.indices, "tolist")
                else sparse_vec.indices
            )
            sparse_values = (
                sparse_vec.values.tolist()
                if hasattr(sparse_vec.values, "tolist")
                else sparse_vec.values
            )

            points.append(
                build_qdrant_point(
                    document_id=meta["document_id"],
                    workspace_id=meta["workspace_id"],
                    filename=meta["filename"],
                    doc_name=meta["doc_name"],
                    chunk_id=meta["chunk_id"],
                    chunk_index=meta["chunk_index"],
                    source_type=meta["source_type"],
                    text=meta["text"],
                    text_bm25=sparse_text,
                    dense_vector=dense_vec,
                    sparse_indices=sparse_indices,
                    sparse_values=sparse_values,
                    page_number=meta.get("page_number"),
                    section_path=meta.get("section_path"),
                    is_table_stub=bool(meta.get("is_table_stub")),
                    table_name=meta.get("table_name"),
                    statement_type=meta.get("statement_type"),
                    citation_label=meta.get("citation_label"),
                    evidence_id=meta.get("evidence_id"),
                    unit_scale=meta.get("unit_scale"),
                    is_primary_statement=meta.get("is_primary_statement"),
                    content_hash=meta.get("content_hash"),
                    ingestion_version=meta.get("ingestion_version", 1),

                    extra_payload={
                        "section": meta.get("section"),
                        "section_level": meta.get("section_level"),
                        "text_hash": meta.get("text_hash"),
                        "col_names": meta.get("col_names"),
                        "local_chunk_index": meta.get("local_chunk_index"),
                    },
                )
            )

            if len(points) >= self.qdrant_batch_size:
                await self.vector_repo.upsert_points(points)
                total_points += len(points)
                points = []

        if points:
            await self.vector_repo.upsert_points(points)
            total_points += len(points)

        return total_points

    def _build_dense_texts(self, chunks: list[dict[str, Any]]) -> list[str]:
        dense_limit = self._dense_token_limit()
        dense_texts: list[str] = []

        for chunk in chunks:
            text = str(chunk.get("text") or "")

            if not chunk.get("is_table_stub"):
                prefix = _dense_metadata_prefix(chunk)
                if prefix:
                    text = f"{prefix}\n\n{text}"

            if hasattr(self.chunker, "truncate_text"):
                text = self.chunker.truncate_text(text, dense_limit)

            dense_texts.append(text)

        return dense_texts

    def _dense_token_limit(self) -> int:
        cfg = getattr(self.chunker, "cfg", None)
        max_tokens = getattr(cfg, "max_tokens", None)

        if isinstance(max_tokens, int) and max_tokens > 4:
            return max_tokens - 2

        return 1000