import logging
import torch
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
from openai import OpenAI

logger = logging.getLogger(__name__)

class BGEDenseEmbedder:
    """Local BGE model for dense embeddings (PoC/Free tier)."""
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5", batch_size: int = 32):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        self.vector_size = self.model.get_embedding_dimension()
        self.batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        prefixed = f"Represent this sentence for searching relevant passages: {text}"
        return self.model.encode([prefixed], normalize_embeddings=True)[0].tolist()

class OpenAIDenseEmbedder:
    """Production-grade OpenAI embedder with dynamic dimensions and batching.

    text-embedding-3-large supports shortening via the `dimensions` parameter.
    We keep `max_input_tokens` as a conservative ingestion-side guard; the
    chunker still controls normal text chunk sizes.
    """
    def __init__(
        self,
        openai_client: OpenAI,
        model_name: str = "text-embedding-3-large",
        dimensions: int = 3072,
        batch_size: int = 64,
        max_input_tokens: int = 7500,
    ):
        self.client = openai_client
        self.model_name = model_name
        self.dimensions = dimensions
        self.vector_size = dimensions
        self.batch_size = batch_size
        self.max_input_tokens = max_input_tokens

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(
            input=texts,
            model=self.model_name,
            dimensions=self.dimensions,
        )
        # The API returns embeddings in input order.
        return [data.embedding for data in response.data]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            if not batch:
                continue
            vectors.extend(self._embed_batch(batch))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed_batch([text])[0]

class SparseEmbedder:
    """Wrapper for BM25 sparse embeddings via fastembed."""
    def __init__(self, model_name: str = "Qdrant/bm25"):
        self.model = SparseTextEmbedding(model_name=model_name)

    def embed(self, texts: list[str]):
        return self.model.embed(texts)