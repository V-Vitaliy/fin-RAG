from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Protocol

logger = logging.getLogger(__name__)


class Tokenizer(Protocol):
    def count(self, text: str) -> int: ...

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...


class HFTokenizer:
    def __init__(self, model_name: str):
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

    def count(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=False)

    def decode(self, ids: list[int]) -> str:
        return self._tokenizer.decode(ids, skip_special_tokens=True)


class TiktokenTokenizer:
    def __init__(self, encoding_name: str = "cl100k_base"):
        import tiktoken

        self._encoding = tiktoken.get_encoding(encoding_name)

    def count(self, text: str) -> int:
        return len(self._encoding.encode(text))

    def encode(self, text: str) -> list[int]:
        return self._encoding.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self._encoding.decode(ids)


@dataclass(frozen=True)
class ChunkerConfig:
    max_tokens: int = 512
    min_tokens: int = 25
    overlap_tokens: int = 64
    merge_ratio: float = 0.80

    tokenizer_type: str = "huggingface"
    tokenizer_name: str = "BAAI/bge-large-en-v1.5"

    bm25_strip_context: bool = True
    preserve_short_financial_facts: bool = True

    micro_preserve_keywords: tuple[str, ...] = field(
        default_factory=lambda: (
            "increase",
            "increased",
            "decrease",
            "decreased",
            "decline",
            "declined",
            "growth",
            "grew",
            "driven",
            "due to",
            "primarily",
            "because",
            "revenue",
            "sales",
            "margin",
            "gross profit",
            "operating income",
            "net income",
            "cash",
            "liquidity",
            "debt",
            "assets",
            "liabilities",
            "equity",
            "acquisition",
            "acquired",
            "divestiture",
            "separation",
            "restructuring",
            "customer",
            "concentration",
            "segment",
            "stores",
            "products",
            "services",
            "capital expenditures",
            "capex",
            "depreciation",
            "amortization",
        )
    )

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        if self.min_tokens < 0:
            raise ValueError("min_tokens must be non-negative")

        if self.overlap_tokens < 0:
            raise ValueError("overlap_tokens must be non-negative")

        if self.overlap_tokens >= self.max_tokens:
            raise ValueError("overlap_tokens must be smaller than max_tokens")

        if not 0 < self.merge_ratio <= 1:
            raise ValueError("merge_ratio must be in the range (0, 1]")

        if self.tokenizer_type not in {"huggingface", "tiktoken"}:
            raise ValueError("tokenizer_type must be 'huggingface' or 'tiktoken'")


_DOC_TYPE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"10[\-_]?K", "10-K"),
    (r"10[\-_]?Q", "10-Q"),
    (r"8[\-_]?K", "8-K"),
    (r"DEF[\-_]?14A", "Proxy Statement"),
    (r"EARNINGS", "Earnings Release"),
    (r"PRESS[\-_]?RELEASE", "Press Release"),
)


class HierarchicalChunker:
    def __init__(self, config: ChunkerConfig):
        self.cfg = config
        self._tokenizer = self._build_tokenizer()

    def _build_tokenizer(self) -> Tokenizer:
        if self.cfg.tokenizer_type == "tiktoken":
            return TiktokenTokenizer(self.cfg.tokenizer_name)

        return HFTokenizer(self.cfg.tokenizer_name)

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return self._tokenizer.count(text)

    def truncate_text(self, text: str, max_tokens: int) -> str:
        if not text:
            return ""

        if max_tokens <= 0:
            return ""

        if self._tokenizer.count(text) <= max_tokens:
            return text

        return self._tokenizer.decode(self._tokenizer.encode(text)[:max_tokens])

    def parse_metadata(self, doc_name: str) -> tuple[str, int | None, str]:
        normalized = str(doc_name or "").strip()
        upper = normalized.upper()

        year_match = re.search(r"(20\d{2})", normalized)
        year = int(year_match.group(1)) if year_match else None

        doc_type = "UNKNOWN"
        doc_type_pos: int | None = None

        for pattern, label in _DOC_TYPE_PATTERNS:
            match = re.search(pattern, upper)
            if match:
                doc_type = label
                doc_type_pos = match.start()
                break

        cutoff = len(normalized)

        if year_match:
            cutoff = min(cutoff, year_match.start())

        if doc_type_pos is not None:
            cutoff = min(cutoff, doc_type_pos)

        ticker = normalized[:cutoff].strip("_- ") if normalized else "UNKNOWN"

        if not ticker:
            parts = re.split(r"[_\-\s]+", normalized)
            ticker = parts[0] if parts and parts[0] else "UNKNOWN"

        ticker = re.sub(r"[_\-\s]+", "_", ticker).strip("_") or "UNKNOWN"

        return ticker, year, doc_type

    @staticmethod
    def _is_table(paragraph: str) -> bool:
        text = str(paragraph or "").strip()

        if not re.search(r"[A-Za-z0-9]", text):
            return False

        lines = text.splitlines()
        pipe_lines = sum(1 for line in lines if "|" in line)
        has_separator = any(re.search(r"\|[\s\-:]+\|", line) for line in lines)

        return has_separator and pipe_lines >= 2

    @staticmethod
    def _is_list_item(paragraph: str) -> bool:
        text = str(paragraph or "")
        return bool(re.match(r"^\s*[-*•]\s", text)) or bool(re.match(r"^\s*\d+\.\s", text))

    @staticmethod
    def _clean_heading_title(line: str) -> str:
        return re.sub(r"^#{1,6}\s+", "", str(line or "")).strip() or "General"

    def _split_sections_with_paths(self, markdown: str) -> list[tuple[str, str, str, int]]:
        lines = str(markdown or "").splitlines()

        sections: list[tuple[str, str, str, int]] = []
        heading_stack: list[tuple[int, str]] = []

        current_lines: list[str] = []
        current_title = "General"
        current_path = "General"
        current_level = 0

        def flush_current() -> None:
            nonlocal current_lines
            text = "\n".join(current_lines).strip()
            if text:
                sections.append((text, current_title, current_path, current_level))
            current_lines = []

        for line in lines:
            match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)

            if not match:
                current_lines.append(line)
                continue

            flush_current()

            level = len(match.group(1))
            title = match.group(2).strip() or "General"

            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()

            heading_stack.append((level, title))

            current_title = title
            current_path = " > ".join(item[1] for item in heading_stack) or title
            current_level = level
            current_lines = [line]

        flush_current()

        if not sections and str(markdown or "").strip():
            return [(str(markdown).strip(), "General", "General", 0)]

        return sections

    def _is_financial_micro_paragraph(self, paragraph: str) -> bool:
        if not self.cfg.preserve_short_financial_facts:
            return False

        text = re.sub(r"\s+", " ", str(paragraph or "")).strip()

        if len(text) < 8:
            return False

        lower = text.lower()

        if any(keyword in lower for keyword in self.cfg.micro_preserve_keywords):
            return True

        if re.search(r"(?:\$|€|£)\s*\(?\d", text):
            return True

        if re.search(r"\b\d+(?:\.\d+)?\s*%", text):
            return True

        if re.search(r"\bFY\s?20\d{2}\b|\bfiscal\s+20\d{2}\b", text, re.IGNORECASE):
            return True

        return False

    def _split_with_overlap(self, text: str, context: str) -> list[str]:
        context_tokens = self._tokenizer.count(context)
        budget = self.cfg.max_tokens - context_tokens

        if budget <= 0:
            return [self.truncate_text(context, self.cfg.max_tokens)]

        if self._tokenizer.count(text) <= budget:
            return [context + text]

        overlap = min(self.cfg.overlap_tokens, max(0, budget // 4))
        stride = max(1, budget - overlap)

        sentences = re.split(r"(?<=[.!?])\s+", text)
        windows: list[str] = []
        current_sentences: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            sentence = sentence.strip()

            if not sentence:
                continue

            sentence_tokens = self._tokenizer.count(sentence)

            if sentence_tokens > budget:
                if current_sentences:
                    windows.append(context + " ".join(current_sentences))
                    current_sentences = []
                    current_tokens = 0

                ids = self._tokenizer.encode(sentence)
                start = 0

                while start < len(ids):
                    end = min(start + budget, len(ids))
                    windows.append(context + self._tokenizer.decode(ids[start:end]))

                    if end >= len(ids):
                        break

                    start = max(0, end - overlap)

                continue

            joiner_tokens = 1 if current_sentences else 0

            if current_tokens + sentence_tokens + joiner_tokens > budget:
                windows.append(context + " ".join(current_sentences))

                overlap_sentences: list[str] = []
                overlap_tokens = 0

                for old_sentence in reversed(current_sentences):
                    old_tokens = self._tokenizer.count(old_sentence)
                    old_joiner = 1 if overlap_sentences else 0

                    if overlap_tokens + old_tokens + old_joiner > overlap:
                        break

                    overlap_sentences.insert(0, old_sentence)
                    overlap_tokens += old_tokens + old_joiner

                current_sentences = overlap_sentences + [sentence]
                current_tokens = overlap_tokens + sentence_tokens + (1 if overlap_sentences else 0)
            else:
                current_sentences.append(sentence)
                current_tokens += sentence_tokens + joiner_tokens

        if current_sentences:
            windows.append(context + " ".join(current_sentences))

        if windows:
            return windows

        ids = self._tokenizer.encode(text)
        start = 0

        while start < len(ids):
            end = min(start + budget, len(ids))
            windows.append(context + self._tokenizer.decode(ids[start:end]))

            if end >= len(ids):
                break

            start += stride

        return windows

    def _merge_paragraphs(self, paragraphs: list[str], context: str) -> list[str]:
        if not paragraphs:
            return []

        context_tokens = self._tokenizer.count(context)
        budget = self.cfg.max_tokens - context_tokens

        if budget <= 0:
            return paragraphs

        soft_limit = max(1, int(budget * self.cfg.merge_ratio))
        hard_limit = max(1, budget)

        merged: list[str] = []
        buffer = paragraphs[0]
        buffer_tokens = self._tokenizer.count(buffer)
        buffer_is_list = self._is_list_item(buffer)

        for paragraph in paragraphs[1:]:
            paragraph_tokens = self._tokenizer.count(paragraph)
            paragraph_is_list = self._is_list_item(paragraph)

            if "[TABLE ON" in buffer or "[TABLE ON" in paragraph:
                merged.append(buffer)
                buffer = paragraph
                buffer_tokens = paragraph_tokens
                buffer_is_list = paragraph_is_list
                continue

            if buffer_is_list and paragraph_is_list:
                if buffer_tokens + paragraph_tokens + 1 <= hard_limit:
                    buffer = f"{buffer}\n\n{paragraph}"
                    buffer_tokens += paragraph_tokens + 1
                    continue

                merged.append(buffer)
                buffer = paragraph
                buffer_tokens = paragraph_tokens
                buffer_is_list = True
                continue

            if buffer_tokens + paragraph_tokens + 1 <= soft_limit:
                buffer = f"{buffer}\n\n{paragraph}"
                buffer_tokens += paragraph_tokens + 1
                buffer_is_list = paragraph_is_list
                continue

            merged.append(buffer)
            buffer = paragraph
            buffer_tokens = paragraph_tokens
            buffer_is_list = paragraph_is_list

        merged.append(buffer)

        return merged

    def chunk_document(
        self,
        markdown: str,
        doc_name: str,
        page_number: int | None = None,
    ) -> list[dict]:
        ticker, year, doc_type = self.parse_metadata(doc_name)
        sections = self._split_sections_with_paths(markdown)

        final_chunks: list[dict] = []
        local_chunk_index = 0

        logger.info(
            "Chunking doc_name=%s ticker=%s year=%s doc_type=%s page=%s sections=%s",
            doc_name,
            ticker,
            year,
            doc_type,
            page_number,
            len(sections),
        )

        for section_text, section_title, section_path, section_level in sections:
            context = (
                f"[Company: {ticker}, Report: {doc_type}, Year: {year}, "
                f"Section: {section_title}, Section Path: {section_path}]\n"
            )

            raw_paragraphs = [
                paragraph.strip()
                for paragraph in section_text.split("\n\n")
                if paragraph.strip()
            ]

            if len(raw_paragraphs) >= 2 and raw_paragraphs[0].startswith("#"):
                raw_paragraphs.pop(0)

            processed_paragraphs = [
                self._table_placeholder(paragraph, page_number)
                if self._is_table(paragraph)
                else paragraph
                for paragraph in raw_paragraphs
            ]

            filtered = [
                paragraph
                for index, paragraph in enumerate(processed_paragraphs)
                if self._should_keep_paragraph(paragraph, index)
            ]

            if not filtered:
                filtered = processed_paragraphs

            merged = self._merge_paragraphs(filtered, context)

            for paragraph in merged:
                is_markdown_table_reference = "[TABLE ON" in paragraph

                for window in self._split_with_overlap(paragraph, context):
                    bm25_text = (
                        re.sub(re.escape(context), "", window, count=1)
                        if self.cfg.bm25_strip_context
                        else window
                    )

                    final_chunks.append(
                        {
                            "text": window,
                            "text_bm25": bm25_text,
                            "is_table_stub": is_markdown_table_reference,
                            "local_chunk_index": local_chunk_index,
                            "doc_name": doc_name,
                            "section": section_title,
                            "section_path": section_path,
                            "section_level": section_level,
                            "ticker": ticker,
                            "year": year,
                            "doc_type": doc_type,
                            "page_number": page_number,
                        }
                    )
                    local_chunk_index += 1

        logger.info("Emitted %s chunks for doc_name=%s page=%s", len(final_chunks), doc_name, page_number)

        return final_chunks

    def _should_keep_paragraph(self, paragraph: str, index: int) -> bool:
        token_count = self._tokenizer.count(paragraph)

        if token_count >= self.cfg.min_tokens:
            return True

        if index == 0 and paragraph.startswith("#"):
            return True

        if "[TABLE ON" in paragraph:
            return True

        return self._is_financial_micro_paragraph(paragraph)

    @staticmethod
    def _table_placeholder(paragraph: str, page_number: int | None) -> str:
        header = paragraph.strip().split("\n")[0]
        page_label = f"PAGE {page_number}" if page_number is not None else "THIS DOCUMENT"
        return f"[TABLE ON {page_label}: {header} ... See extracted table metadata for SQL access]"