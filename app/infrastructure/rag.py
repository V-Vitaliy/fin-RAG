from __future__ import annotations

from openai import OpenAI
import logging

from app.core.config import settings
from app.services.ingestion.chunker import ChunkerConfig, HierarchicalChunker
from app.services.retrieval.embeddings import BGEDenseEmbedder, OpenAIDenseEmbedder, SparseEmbedder
from app.services.retrieval.reranker import CrossEncoderReranker, select_reranker

logger = logging.getLogger(__name__)


def build_dense_embedder():
    if settings.RAG_USE_OPENAI_EMBEDDINGS:
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required when RAG_USE_OPENAI_EMBEDDINGS=true")

        return OpenAIDenseEmbedder(
            openai_client=OpenAI(api_key=settings.OPENAI_API_KEY),
            model_name=settings.RAG_EMBEDDING_MODEL,
            dimensions=settings.RAG_EMBEDDING_DIMENSIONS,
            batch_size=settings.RAG_EMBEDDING_BATCH_SIZE,
            max_input_tokens=settings.RAG_EMBEDDING_MAX_INPUT_TOKENS,
        )

    return BGEDenseEmbedder(
        model_name=settings.RAG_LOCAL_EMBEDDING_MODEL,
        batch_size=settings.RAG_LOCAL_EMBEDDING_BATCH_SIZE,
    )


def build_sparse_embedder() -> SparseEmbedder:
    return SparseEmbedder("Qdrant/bm25")


def build_chunker() -> HierarchicalChunker:
    if settings.RAG_USE_OPENAI_EMBEDDINGS:
        tokenizer_type = "tiktoken"
        tokenizer_name = "cl100k_base"
    else:
        tokenizer_type = "huggingface"
        tokenizer_name = settings.RAG_LOCAL_EMBEDDING_MODEL

    return HierarchicalChunker(
        config=ChunkerConfig(
            max_tokens=settings.RAG_CHUNK_MAX_TOKENS,
            min_tokens=settings.RAG_CHUNK_MIN_TOKENS,
            overlap_tokens=settings.RAG_CHUNK_OVERLAP_TOKENS,
            merge_ratio=settings.RAG_CHUNK_MERGE_RATIO,
            tokenizer_type=tokenizer_type,
            tokenizer_name=tokenizer_name,
            bm25_strip_context=True,
        )
    )

def build_reranker():
    selection = select_reranker(
        use_reranker=settings.RAG_USE_RERANKER,
        mode=settings.RAG_RERANKER_MODE,
        cpu_model=settings.RAG_CPU_RERANKER_MODEL,
        gpu_model=settings.RAG_GPU_RERANKER_MODEL,
    )

    logger.info(
        "Reranker selection: enabled=%s mode=%s model=%s device=%s reason=%s",
        selection.enabled,
        selection.mode,
        selection.model_name,
        selection.device,
        selection.reason,
    )

    if not selection.enabled:
        return None

    return CrossEncoderReranker(
        model_name=selection.model_name,
        device=selection.device,
        max_length=settings.RAG_RERANKER_MAX_LENGTH,
        batch_size=settings.RAG_RERANKER_BATCH_SIZE,
    )