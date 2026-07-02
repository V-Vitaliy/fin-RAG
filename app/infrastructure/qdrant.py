from qdrant_client import AsyncQdrantClient
from app.core.config import settings

def build_qdrant_client() -> AsyncQdrantClient:
    """
    Initializes and returns the async Qdrant client.
    """
    if not settings.QDRANT_URL:
        raise ValueError("QDRANT_URL is not configured")
    
    return AsyncQdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY
    )
