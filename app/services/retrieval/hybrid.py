from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from app.repositories.qdrant_repo import VectorIndexRepository

logger = logging.getLogger(__name__)


class AgentHybridRetriever:
    def __init__(
        self,
        *,
        vector_repo: VectorIndexRepository,
        dense_embedder,
        sparse_embedder,
        reranker=None,
        prefetch_min: int = 60,
        fusion_min: int = 40,
        first_stage_multiplier: int = 8,
        fusion_multiplier: int = 4,
    ):
        self.vector_repo = vector_repo
        self.dense_embedder = dense_embedder
        self.sparse_embedder = sparse_embedder
        self.reranker = reranker

        self.prefetch_min = int(prefetch_min)
        self.fusion_min = int(fusion_min)
        self.first_stage_multiplier = int(first_stage_multiplier)
        self.fusion_multiplier = int(fusion_multiplier)

    @staticmethod
    def _is_driver_query(query: str) -> bool:
        q = (query or "").lower()
        return any(
            phrase in q
            for phrase in [
                "what drove",
                "driven by",
                "drivers",
                "what caused",
                "why did",
                "due to",
                "primarily",
                "offset by",
                "gross margin change",
                "operating margin change",
                "revenue change",
                "sales change",
                "margin change",
            ]
        )

    @staticmethod
    def _is_narrative_like_query(query: str) -> bool:
        q = (query or "").lower()
        return AgentHybridRetriever._is_driver_query(q) or any(
            phrase in q
            for phrase in [
                "major acquisitions",
                "business combinations",
                "acquired",
                "customer concentration",
                "legal proceedings",
                "legal battles",
                "high growth",
                "growth company",
                "products and services",
                "major products",
            ]
        )

    @staticmethod
    def _clean_extra_queries(extra_queries: list[str] | None) -> list[str]:
        """
        Keep LLM-provided rewrites useful but safe:
        no table IDs, page hints, expected answers, or prompt-like long strings.
        """
        if not extra_queries:
            return []

        cleaned: list[str] = []

        for item in extra_queries[:3]:
            if not isinstance(item, str):
                continue

            query = " ".join(item.strip().split())
            if not query or len(query) > 180:
                continue

            query_lower = query.lower()
            forbidden_patterns = [
                r"\btbl_[a-z0-9_]+",
                r"\bpage\s*\d+\b",
                r"\bp\.\s*\d+\b",
                r"\brow[_\s-]?id\b",
                r"\btable[_\s-]?id\b",
                r"\banswer\s*:",
                r"\bexpected\s*:",
            ]

            if any(re.search(pattern, query_lower) for pattern in forbidden_patterns):
                continue

            cleaned.append(query)

        return cleaned

    def _rewrite_query(self, query: str, extra_queries: list[str] | None = None) -> list[str]:
        rewrites = [query]

        for extra in self._clean_extra_queries(extra_queries):
            if extra.lower() not in [rewrite.lower() for rewrite in rewrites]:
                rewrites.append(extra)

        q = query.strip()
        q = re.sub(
            r"^(what is|what was|what are|what were|how much|how many|"
            r"does|did|has|have|is|are|was|were|based on|calculate|"
            r"using|please (state|provide|calculate)|as of)[,\s]+",
            "",
            q,
            flags=re.IGNORECASE,
        ).strip()

        fy_match = re.search(r"\b(FY\s?\d{4}(?:Q\d)?|\d{4})\b", q)
        fy_tag = fy_match.group(0).replace(" ", "") if fy_match else ""

        q_short = re.sub(r"\?$", "", q).strip()
        if q_short and q_short.lower() != query.lower():
            rewrites.append(q_short)

        synonym_map = {
            "net income": "net earnings profit loss attributable",
            "revenue": "net revenue net sales total revenue",
            "cash flow from operations": "net cash provided by operating activities",
            "cash flow from operating": "net cash provided by operating activities",
            "operating activities": "net cash provided by operating activities cash from ops",
            "quick ratio": "cash short-term investments accounts receivable current liabilities liquidity",
            "inventory turnover": "cost of goods sold COGS inventory balance",
            "return on assets": "net income total assets ROA",
            "roa": "net income total assets return on assets",
            "ebitda": "operating income depreciation amortization D&A",
            "capex": "capital expenditures purchases of property plant equipment",
            "restructuring": "restructuring charges costs impairment",
            "d&a": "depreciation amortization adjustments reconcile",
            "gain": "gain on sale proceeds transaction separation",
            "proceeds": "cash received consideration received sale proceeds",
            "separation": "spin-off divestiture separation transaction",
            "legal": "litigation proceedings contingencies commitments",
            "working capital": "current assets current liabilities",
            "debt": "long-term debt borrowings notes payable",
            "segment": "business segment reporting geographic",
        }

        query_lower = query.lower()
        expanded_terms: list[str] = []

        for key, expansion in synonym_map.items():
            if key in query_lower:
                expanded_terms.append(expansion)

        if expanded_terms:
            expansion_query = f"{fy_tag} {expanded_terms[0]}".strip()
            if expansion_query.lower() not in [rewrite.lower() for rewrite in rewrites]:
                rewrites.append(expansion_query)

        narrative_expansions: list[str] = []

        if self._is_driver_query(query):
            narrative_expansions.append(
                f"{fy_tag} management discussion results of operations due to driven by primarily offset by "
                "currency inflation supply chain acquisition product mix volume price demand"
            )

        if "gross margin" in query_lower:
            narrative_expansions.append(
                f"{fy_tag} gross profit margin increase decrease cost of products sold currency commodity inflation "
                "supply chain product mix"
            )

        if "operating margin" in query_lower or "operating income" in query_lower:
            narrative_expansions.append(
                f"{fy_tag} operating income margin increase decrease amortization intangible assets acquisition "
                "restructuring operating expenses"
            )

        if "major acquisition" in query_lower or "acquisitions" in query_lower or "business combination" in query_lower:
            narrative_expansions.append(
                f"{fy_tag} business combinations acquisitions acquired outstanding shares purchase price acquired company "
                "closed acquisition"
            )

        if "high growth" in query_lower or "growth company" in query_lower:
            narrative_expansions.append(
                f"{fy_tag} sales growth net sales increase decrease consolidated sales results of operations"
            )

        for expansion_query in narrative_expansions:
            expansion_query = expansion_query.strip()
            if expansion_query and expansion_query.lower() not in [rewrite.lower() for rewrite in rewrites]:
                rewrites.append(expansion_query)

        seen_lower: set[str] = set()
        unique: list[str] = []

        for rewrite in rewrites:
            normalized = rewrite.strip().lower()
            if normalized and normalized not in seen_lower:
                seen_lower.add(normalized)
                unique.append(rewrite.strip())

        cap = 4 if self._is_narrative_like_query(query) else 3
        return unique[:cap]

    @staticmethod
    def _metadata_boost(payload: dict[str, Any], query: str, doc_name: str | None = None) -> float:
        boost = 0.0

        if doc_name:
            payload_doc_name = str(payload.get("doc_name") or "")
            if doc_name.lower().removesuffix(".pdf") == payload_doc_name.lower().removesuffix(".pdf"):
                boost += 0.5

        q = (query or "").lower()
        section = str(payload.get("section_path") or "").lower()
        source_type = str(payload.get("source_type") or "").lower()
        is_table = bool(payload.get("is_table_stub"))

        if is_table:
            boost += 0.2

        if AgentHybridRetriever._is_narrative_like_query(q):
            if not is_table and source_type != "table_stub":
                boost += 0.25

            if any(
                term in section
                for term in [
                    "management",
                    "discussion",
                    "results of operations",
                    "operating results",
                    "segment",
                    "business combinations",
                    "acquisition",
                    "acquisitions",
                    "notes to",
                    "financial review",
                    "consolidated results",
                ]
            ):
                boost += 0.35

            if any(
                term in section
                for term in [
                    "signatures",
                    "exhibits",
                    "index",
                    "controls and procedures",
                    "subsidiaries",
                    "cover",
                    "table of contents",
                ]
            ):
                boost -= 0.25

            if AgentHybridRetriever._is_driver_query(q) and is_table:
                boost -= 0.15

        if "acquisition" in q or "business combination" in q:
            if any(term in section for term in ["business combinations", "acquisition", "acquisitions", "notes to"]):
                boost += 0.35
            if "cash flow" in section or "statements of cash flows" in section:
                boost -= 0.20

        if "high growth" in q or "growth company" in q:
            if any(term in section for term in ["selected financial", "management", "results of operations", "consolidated results"]):
                boost += 0.25

        return max(boost, -0.75)

    @staticmethod
    def _deduplicate(results: list) -> list:
        seen: set[str] = set()
        output = []

        for result in results:
            payload = dict(getattr(result, "payload", None) or {})
            prefix = str(payload.get("text") or "")[:120].strip()

            if prefix and prefix in seen:
                continue

            if prefix:
                seen.add(prefix)

            output.append(result)

        return output

    @staticmethod
    def _point_id(point: object) -> str:
        return str(getattr(point, "id", ""))

    @staticmethod
    def _point_score(point: object) -> float:
        try:
            return float(getattr(point, "score", 0.0) or 0.0)
        except Exception:
            return 0.0

    @staticmethod
    def _payload(point: object) -> dict[str, Any]:
        return dict(getattr(point, "payload", None) or {})

    @staticmethod
    def _to_list(value) -> list:
        return value.tolist() if hasattr(value, "tolist") else list(value)

    async def _embed_dense_query(self, query: str) -> list[float]:
        result = await asyncio.to_thread(self.dense_embedder.embed_query, query)
        return self._to_list(result)

    async def _embed_sparse_query(self, query: str) -> tuple[list[int], list[float]]:
        sparse_results = await asyncio.to_thread(lambda: list(self.sparse_embedder.embed([query])))
        sparse = sparse_results[0]

        indices = self._to_list(sparse.indices)
        values = self._to_list(sparse.values)

        return indices, values

    async def _single_search(
        self,
        *,
        query: str,
        allowed_document_ids: list[str],
        doc_name: str | None,
        top_k: int,
        only_table_stubs: bool,
        boost_table_stubs: bool,
    ) -> list:
        dense_vector = await self._embed_dense_query(query)
        if not dense_vector:
            return []

        sparse_indices, sparse_values = await self._embed_sparse_query(query)

        prefetch_limit = max(self.prefetch_min, top_k * self.first_stage_multiplier)
        fusion_limit = max(self.fusion_min, top_k * self.fusion_multiplier)

        return await self.vector_repo.hybrid_search(
            dense_vector=dense_vector,
            sparse_indices=sparse_indices,
            sparse_values=sparse_values,
            allowed_document_ids=allowed_document_ids,
            limit=fusion_limit,
            prefetch_limit=prefetch_limit,
            only_table_stubs=only_table_stubs,
            boost_table_stubs=boost_table_stubs,
        )

    async def search(
        self,
        *,
        query: str,
        allowed_document_ids: list[str],
        doc_name: str | None = None,
        top_k: int = 8,
        only_table_stubs: bool = False,
        extra_queries: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        query = str(query or "").strip()
        if not query:
            raise ValueError("Query must not be empty")

        if not allowed_document_ids:
            return []

        logger.info(
            "Retrieval search query=%r doc_name=%s top_k=%s only_table_stubs=%s",
            query,
            doc_name,
            top_k,
            only_table_stubs,
        )

        query_lower = query.lower()
        financial_keywords = [
            "ratio",
            "revenue",
            "margin",
            "cash flow",
            "balance",
            "income",
            "assets",
            "liabilities",
            "sales",
            "debt",
            "segment",
            "financial",
            "statement",
            "acquisition",
            "discontinued",
            "restructuring",
            "depreciation",
            "amortization",
            "capex",
            "ebitda",
            "inventory",
            "turnover",
            "earnings",
            "profit",
            "equity",
            "expense",
            "gain",
            "proceeds",
            "separation",
            "legal",
            "working capital",
        ]

        boost_table_stubs = (not only_table_stubs) and any(keyword in query_lower for keyword in financial_keywords)

        queries = self._rewrite_query(query, extra_queries=extra_queries)
        logger.info("Retrieval query rewrites: %s", queries)

        best_by_id: dict[str, object] = {}

        for rewritten_query in queries:
            points = await self._single_search(
                query=rewritten_query,
                allowed_document_ids=allowed_document_ids,
                doc_name=doc_name,
                top_k=top_k,
                only_table_stubs=only_table_stubs,
                boost_table_stubs=boost_table_stubs,
            )

            for point in points:
                point_id = self._point_id(point)
                if not point_id:
                    continue

                existing = best_by_id.get(point_id)
                if existing is None:
                    best_by_id[point_id] = point
                    continue

                if self._point_score(point) > self._point_score(existing):
                    best_by_id[point_id] = point

        merged = self._deduplicate(list(best_by_id.values()))

        if self.reranker and merged:
            pairs = [[query, self._payload(point).get("text", "")] for point in merged]
            scores = await asyncio.to_thread(self.reranker.predict, pairs)

            reranked = list(zip(merged, scores))
            reranked.sort(
                key=lambda item: float(item[1] if item[1] is not None else 0.001)
                * (1 + self._metadata_boost(self._payload(item[0]), query, doc_name)),
                reverse=True,
            )

            final_results = [point for point, _score in reranked[:top_k]]
        else:
            final_results = sorted(
                merged,
                key=lambda point: self._point_score(point)
                * (1 + self._metadata_boost(self._payload(point), query, doc_name)),
                reverse=True,
            )[:top_k]

        chunks: list[dict[str, Any]] = []

        for rank, point in enumerate(final_results, 1):
            payload = self._payload(point)
            chunks.append(
                {
                    "rank": rank,
                    "score": self._point_score(point),
                    **payload,
                }
            )

        return chunks

    @staticmethod
    def _get_neighbor_index(payload: dict[str, Any]) -> int | None:
        for key in ("local_chunk_index", "text_chunk_index", "chunk_index"):
            value = payload.get(key)
            if value is None:
                continue
            try:
                index = int(value)
            except Exception:
                continue
            if index >= 0:
                return index
        return None

    async def _fetch_text_neighbors(
        self,
        *,
        document_id: str,
        text_chunk_index: int,
        window: int = 1,
    ) -> list[dict[str, Any]]:
        if text_chunk_index is None or text_chunk_index < 0 or window <= 0:
            return []

        try:
            points = await self.vector_repo.scroll_text_neighbors(
                document_id=str(document_id),
                center_index=int(text_chunk_index),
                window=int(window),
                limit=(2 * int(window) + 1),
            )
        except Exception as exc:
            logger.warning(
                "Neighbor expansion failed for document_id=%s index=%s window=%s: %s",
                document_id,
                text_chunk_index,
                window,
                exc,
            )
            return []

        neighbors: list[dict[str, Any]] = []

        for point in points:
            payload = self._payload(point)
            index = self._get_neighbor_index(payload)
            if index is None:
                continue

            payload["rank"] = None
            payload["score"] = None
            payload["context_role"] = (
                "hit"
                if index == text_chunk_index
                else "previous_neighbor"
                if index < text_chunk_index
                else "next_neighbor"
            )
            payload["neighbor_of_text_chunk_index"] = text_chunk_index
            neighbors.append(payload)

        neighbors.sort(key=lambda item: self._get_neighbor_index(item) or 10**12)
        return neighbors

    def _result_key(self, item: dict[str, Any]) -> tuple[str, int | str]:
        document_id = str(item.get("document_id") or item.get("doc_name") or "")
        index = self._get_neighbor_index(item)

        if index is not None:
            return document_id, index

        return (
            document_id,
            item.get("table_name")
            or item.get("chunk_id")
            or str(item.get("text") or "")[:120],
        )

    async def _expand_with_neighbor_chunks(
        self,
        results: list[dict[str, Any]],
        *,
        neighbor_window: int = 1,
        max_expanded: int | None = None,
    ) -> list[dict[str, Any]]:
        if not results or neighbor_window <= 0:
            return results

        expanded: list[dict[str, Any]] = []
        seen: set[tuple[str, int | str]] = set()

        for result in results:
            key = self._result_key(result)

            if key not in seen:
                item = dict(result)
                item.setdefault("context_role", "hit")
                expanded.append(item)
                seen.add(key)

            if result.get("is_table_stub"):
                continue

            document_id = result.get("document_id")
            index = self._get_neighbor_index(result)

            if not document_id or index is None:
                continue

            neighbors = await self._fetch_text_neighbors(
                document_id=str(document_id),
                text_chunk_index=index,
                window=neighbor_window,
            )

            for neighbor in neighbors:
                neighbor_key = self._result_key(neighbor)
                if neighbor_key in seen:
                    continue

                neighbor["origin_rank"] = result.get("rank")
                neighbor["origin_score"] = result.get("score")
                expanded.append(neighbor)
                seen.add(neighbor_key)

            if max_expanded is not None and len(expanded) >= max_expanded:
                return expanded[:max_expanded]

        return expanded

    @staticmethod
    def _format_results_for_tool(
        results: list[dict[str, Any]],
        *,
        include_table_metadata: bool = True,
    ) -> str:
        """
        Format retrieved chunks for tool-calling agent.
        Keeps [C#]/[T#] markers because citation validation expects those.
        """
        if not results:
            return "No relevant results found."

        blocks: list[str] = []

        for idx, result in enumerate(results, 1):
            is_table = bool(result.get("is_table_stub"))
            if not include_table_metadata and is_table:
                continue

            source_type = result.get("source_type") or ("table_stub" if is_table else "text")
            label_prefix = "T" if is_table else "C"

            citation_label = result.get("citation_label") or (
                f"{result.get('doc_name')} p.{result.get('page_number')}"
                if result.get("page_number") is not None
                else str(result.get("doc_name"))
            )
            evidence_id = result.get("evidence_id") or "unknown"

            header = (
                f"[{label_prefix}{idx} | rank={result.get('rank')} | score={result.get('score')} | "
                f"source={citation_label} | evidence_id={evidence_id} | "
                f"doc={result.get('doc_name')} | document_id={result.get('document_id')} | "
                f"page={result.get('page_number')} | source_type={source_type} | table_stub={is_table}"
            )

            if result.get("context_role"):
                header += f" | context={result.get('context_role')}"

            if result.get("origin_rank"):
                header += f" | neighbor_of_rank={result.get('origin_rank')}"

            if result.get("neighbor_of_text_chunk_index") is not None:
                header += f" | neighbor_of_text_chunk_index={result.get('neighbor_of_text_chunk_index')}"

            if result.get("local_chunk_index") is not None:
                header += f" | local_chunk_index={result.get('local_chunk_index')}"

            if result.get("chunk_index") is not None:
                header += f" | chunk_index={result.get('chunk_index')}"

            if result.get("section_path"):
                header += f" | section_path={result.get('section_path')}"

            if result.get("table_name"):
                header += f" | table={result.get('table_name')}"

            if result.get("statement_type"):
                header += f" | type={result.get('statement_type')}"

            if result.get("unit_scale"):
                header += f" | unit_scale={result.get('unit_scale')}"

            if result.get("is_primary_statement") is not None:
                header += f" | primary_statement={result.get('is_primary_statement')}"

            header += "]"

            blocks.append(f"{header}\n{result.get('text', '')}")

        return "\n\n".join(blocks) if blocks else "No relevant non-table text results found."

    async def search_text(
        self,
        *,
        query: str,
        allowed_document_ids: list[str],
        doc_name: str | None = None,
        top_k: int = 8,
        expand_neighbors: bool = False,
        neighbor_window: int = 1,
        extra_queries: list[str] | None = None,
    ) -> str:
        results = await self.search(
            query=query,
            allowed_document_ids=allowed_document_ids,
            doc_name=doc_name,
            top_k=top_k,
            only_table_stubs=False,
            extra_queries=extra_queries,
        )

        if expand_neighbors:
            results = await self._expand_with_neighbor_chunks(
                results,
                neighbor_window=neighbor_window,
                max_expanded=max(top_k * (2 * neighbor_window + 1), top_k),
            )

        return self._format_results_for_tool(results, include_table_metadata=True)

    async def search_tables(
        self,
        *,
        concept: str,
        allowed_document_ids: list[str],
        doc_name: str | None = None,
        top_k: int = 8,
        extra_queries: list[str] | None = None,
    ) -> str:
        results = await self.search(
            query=concept,
            allowed_document_ids=allowed_document_ids,
            doc_name=doc_name,
            top_k=top_k,
            only_table_stubs=True,
            extra_queries=extra_queries,
        )

        return self._format_results_for_tool(results, include_table_metadata=True)