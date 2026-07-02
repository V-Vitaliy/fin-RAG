from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from uuid import UUID

from qdrant_client.http import models

from app.services.ingestion.naming import make_qdrant_point_id


def _clean_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Remove None values from Qdrant payload.

    Keeps False/0/empty strings because those can be meaningful.
    """
    return {key: value for key, value in payload.items() if value is not None}


def build_qdrant_payload(
    *,
    document_id: UUID | str,
    workspace_id: UUID | str,
    filename: str,
    doc_name: str,
    chunk_id: str,
    chunk_index: int,
    source_type: str,
    text: str,
    text_bm25: str | None = None,
    page_number: int | None = None,
    section_path: str | list[str] | None = None,
    is_table_stub: bool = False,
    table_name: str | None = None,
    statement_type: str | None = None,
    citation_label: str | None = None,
    evidence_id: str | None = None,
    unit_scale: str | None = None,
    is_primary_statement: bool | None = None,
    content_hash: str | None = None,
    ingestion_version: int = 1,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build Qdrant payload for a text chunk or table stub.

    Payload can be rich; Qdrant payload indexes should stay minimal and only cover
    fields used for filtering/access/routing.
    """
    payload: dict[str, Any] = {
        # Access / document identity
        "document_id": str(document_id),
        "workspace_id": str(workspace_id),
        "filename": filename,
        "doc_name": doc_name,

        # Chunk identity
        "chunk_id": chunk_id,
        "chunk_index": int(chunk_index),
        "source_type": source_type,

        # Text used by retriever/reranker/agent context
        "text": text,
        "text_bm25": text_bm25 or text,

        # Location / section metadata
        "page_number": page_number,
        "section_path": section_path,

        # Table metadata
        "is_table_stub": bool(is_table_stub),
        "table_name": table_name,
        "statement_type": statement_type,

        # Citation / evidence metadata
        "citation_label": citation_label,
        "evidence_id": evidence_id,
        "unit_scale": unit_scale,
        "is_primary_statement": is_primary_statement,

        # Versioning / dedup
        "content_hash": content_hash,
        "ingestion_version": int(ingestion_version),
    }

    if extra:
        payload.update(dict(extra))

    return _clean_payload(payload)


def build_qdrant_point(
    *,
    document_id: UUID | str,
    workspace_id: UUID | str,
    filename: str,
    doc_name: str,
    chunk_id: str,
    chunk_index: int,
    source_type: str,
    text: str,
    dense_vector: Sequence[float],
    sparse_indices: Sequence[int] | None = None,
    sparse_values: Sequence[float] | None = None,
    text_bm25: str | None = None,
    page_number: int | None = None,
    section_path: str | list[str] | None = None,
    is_table_stub: bool = False,
    table_name: str | None = None,
    statement_type: str | None = None,
    citation_label: str | None = None,
    evidence_id: str | None = None,
    unit_scale: str | None = None,
    is_primary_statement: bool | None = None,
    content_hash: str | None = None,
    ingestion_version: int = 1,
    extra_payload: Mapping[str, Any] | None = None,
) -> models.PointStruct:
    """
    Build a Qdrant PointStruct with a stable UUID ID and dense/sparse vectors.
    """
    point_id = make_qdrant_point_id(
        document_id=document_id,
        chunk_index=chunk_index,
        source_type=source_type,
    )

    payload = build_qdrant_payload(
        document_id=document_id,
        workspace_id=workspace_id,
        filename=filename,
        doc_name=doc_name,
        chunk_id=chunk_id,
        chunk_index=chunk_index,
        source_type=source_type,
        text=text,
        text_bm25=text_bm25,
        page_number=page_number,
        section_path=section_path,
        is_table_stub=is_table_stub,
        table_name=table_name,
        statement_type=statement_type,
        citation_label=citation_label,
        evidence_id=evidence_id,
        unit_scale=unit_scale,
        is_primary_statement=is_primary_statement,
        content_hash=content_hash,
        ingestion_version=ingestion_version,
        extra=extra_payload,
    )

    vector: dict[str, Any] = {
        "dense": list(dense_vector),
    }

    if sparse_indices is not None and sparse_values is not None:
        vector["bm25"] = models.SparseVector(
            indices=list(sparse_indices),
            values=list(sparse_values),
        )

    return models.PointStruct(
        id=point_id,
        vector=vector,
        payload=payload,
    )