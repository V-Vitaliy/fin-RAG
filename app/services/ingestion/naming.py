import re
import uuid
from hashlib import sha1
from uuid import UUID



def _strip_doc_name_noise(value: str) -> str:
    """
    Removes non-semantic suffixes from SEC-style document names.

    Example:
        footlocker_2022_8k_dated_2022_08_19
        -> footlocker_2022_8k
    """
    noise_patterns = [
        r"_(?:dated|date|filed|filing|filing_date)_(?:19|20)\d{2}_\d{1,2}_\d{1,2}(?:_.*)?$",
        r"_(?:dated|date|filed|filing|filing_date)_(?:19|20)\d{6}(?:_.*)?$",
    ]

    result = value

    for pattern in noise_patterns:
        result = re.sub(pattern, "", result)

    return re.sub(r"_+", "_", result).strip("_")

def make_safe_doc_slug(doc_name: str, max_len: int = 48) -> str:
    value = str(doc_name or "document").strip().lower()

    value = re.sub(r"\.pdf$", "", value, flags=re.IGNORECASE)
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")

    value = _strip_doc_name_noise(value)

    if not value:
        value = "document"

    return value[:max_len].strip("_") or "document"


def short_document_id(document_id: UUID | str, length: int = 8) -> str:
    raw = str(document_id).replace("-", "")
    if len(raw) >= length and re.fullmatch(r"[0-9a-fA-F]+", raw):
        return raw[:length].lower()

    digest = sha1(str(document_id).encode("utf-8")).hexdigest()
    return digest[:length]


def make_table_name(
    *,
    document_id: UUID | str,
    doc_name: str,
    table_index: int,
    page_number: int | None = None,
) -> str:
    """
    Collision-safe DuckDB table name.

    Keeps doc slug for debugging, but uniqueness comes from document_id.
    """
    doc_short = short_document_id(document_id)
    slug = make_safe_doc_slug(doc_name)
    page = f"p{int(page_number):03d}" if page_number is not None else "p000"
    table = f"t{int(table_index):03d}"

    return f"tbl_{doc_short}_{slug}_{page}_{table}"


def make_qdrant_point_id(
    *,
    document_id: UUID | str,
    chunk_index: int,
    source_type: str,
) -> str:
    """
    Stable UUID point ID for Qdrant.
    """
    raw = f"{document_id}:{source_type}:{chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))