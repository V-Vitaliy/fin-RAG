from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


_BRACKET_RE = re.compile(r"\[([^\]]+)\]")
_MARKER_IN_BRACKET_RE = re.compile(r"\b([CT])(\d+)\b", re.IGNORECASE)



@dataclass
class AgentCitation:
    marker: str
    document_id: str
    source_type: str
    doc_name: str | None = None
    filename: str | None = None
    page_number: int | None = None
    table_name: str | None = None
    evidence_id: str | None = None
    label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "marker": self.marker,
            "document_id": self.document_id,
            "source_type": self.source_type,
            "doc_name": self.doc_name,
            "filename": self.filename,
            "page_number": self.page_number,
            "table_name": self.table_name,
            "evidence_id": self.evidence_id,
            "label": self.label,
        }


@dataclass
class AgentSourceDocument:
    document_id: str
    filename: str
    url: str
    markers: list[str] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)
    expires_in_seconds: int = 600

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "url": self.url,
            "markers": self.markers,
            "pages": self.pages,
            "expires_in_seconds": self.expires_in_seconds,
        }


class AgentCitationRegistry:
    """
    Per-agent-run registry for globally unique [C#]/[T#] markers.

    Without this registry each tool call restarts from [C1]/[T1],
    so the final answer cannot be mapped safely back to real documents.
    """

    def __init__(self) -> None:
        self._c_counter = 0
        self._t_counter = 0
        self._marker_by_key: dict[tuple[Any, ...], str] = {}
        self._citation_by_marker: dict[str, AgentCitation] = {}

    @staticmethod
    def extract_markers(text: str) -> list[str]:
        markers: list[str] = []
        seen: set[str] = set()

        for bracket_content in _BRACKET_RE.findall(text or ""):
            for kind, number in _MARKER_IN_BRACKET_RE.findall(bracket_content):
                marker = f"{kind.upper()}{number}"

                if marker not in seen:
                    seen.add(marker)
                    markers.append(marker)

        return markers

    @staticmethod
    def _clean_str(value: Any) -> str | None:
        if value is None:
            return None

        text = str(value).strip()
        return text or None

    @staticmethod
    def _clean_int(value: Any) -> int | None:
        if value is None:
            return None

        try:
            return int(value)
        except Exception:
            return None

    @staticmethod
    def _is_table_source(data: dict[str, Any]) -> bool:
        source_type = str(data.get("source_type") or "").lower()

        return bool(
            data.get("is_table_stub")
            or data.get("table_name")
            or source_type in {"table", "table_stub", "sql", "financial_table"}
        )

    def _next_marker(self, *, table_source: bool) -> str:
        if table_source:
            self._t_counter += 1
            return f"T{self._t_counter}"

        self._c_counter += 1
        return f"C{self._c_counter}"

    def register(self, data: dict[str, Any]) -> str:
        document_id = self._clean_str(data.get("document_id"))
        if not document_id:
            document_id = "unknown-document"

        table_source = self._is_table_source(data)
        source_type = "table" if table_source else "text"

        table_name = self._clean_str(data.get("table_name"))
        evidence_id = self._clean_str(data.get("evidence_id"))
        chunk_id = self._clean_str(data.get("chunk_id"))
        text_hash = self._clean_str(data.get("text_hash"))
        page_number = self._clean_int(data.get("page_number"))

        if table_source:
            key = (
                "T",
                document_id,
                table_name or evidence_id or chunk_id or text_hash or page_number,
            )
        else:
            key = (
                "C",
                document_id,
                evidence_id or chunk_id or text_hash or page_number,
            )

        existing = self._marker_by_key.get(key)
        if existing:
            return existing

        marker = self._next_marker(table_source=table_source)
        self._marker_by_key[key] = marker

        label = self._clean_str(data.get("citation_label"))
        if not label:
            doc_name = self._clean_str(data.get("doc_name"))
            if page_number is not None and table_name:
                label = f"{doc_name} p.{page_number}, table={table_name}"
            elif page_number is not None:
                label = f"{doc_name} p.{page_number}"
            else:
                label = doc_name

        self._citation_by_marker[marker] = AgentCitation(
            marker=marker,
            document_id=document_id,
            source_type=source_type,
            doc_name=self._clean_str(data.get("doc_name")),
            filename=self._clean_str(data.get("filename")),
            page_number=page_number,
            table_name=table_name,
            evidence_id=evidence_id,
            label=label,
        )

        return marker

    def get(self, marker: str) -> AgentCitation | None:
        return self._citation_by_marker.get(str(marker or "").strip())

    def citations_for_answer(self, answer: str) -> list[AgentCitation]:
        citations: list[AgentCitation] = []

        for marker in self.extract_markers(answer):
            citation = self.get(marker)
            if citation:
                citations.append(citation)

        return citations