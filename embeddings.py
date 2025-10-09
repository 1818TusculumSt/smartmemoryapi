import numpy as np
import httpx
import logging
from typing import Optional
from sentence_transformers import SentenceTransformer
from config import settings

logger = logging.getLogger(__name__)

class EmbeddingProvider:
    """
    Async embedding provider with connection pooling.
    
    Supports:
    - Local: sentence-transformers
    - API: OpenAI-compatible
    - Pinecone: Pinecone inference API
    
    Optimizations:
    - HTTP/2 enabled
    - Connection pooling
    - Async operations
    """
    
    def __init__(self):
        self.provider_type = settings.EMBEDDING_PROVIDER
        self._local_model: Optional[SentenceTransformer] = None
        self._client: Optional[httpx.AsyncClient] = None
        
        if self.provider_type == "local":
            self._init_local_model()
        
        logger.info(f"🧠 Embedding provider: {self.provider_type}")
    
    def _init_local_model(self):
        """Initialize local sentence-transformer model"""
        try:
            logger.info(f"📥 Loading local model: {settings.EMBEDDING_MODEL}")
            self._local_model = SentenceTransformer(settings.EMBEDDING_MODEL)
            dim = self._local_model.get_sentence_embedding_dimension()
            logger.info(f"✅ Local model loaded (dim: {dim})")
        except Exception as e:
            logger.error(f"❌ Failed to load local model: {e}")
            raise RuntimeError(f"Could not initialize local model: {e}")
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client"""
        if not self._client or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(settings.EMBEDDING_TIMEOUT, connect=10.0),
                limits=httpx.Limits(
                    max_keepalive_connections=settings.KEEPALIVE_CONNECTIONS,
                    max_connections=settings.CONNECTION_POOL_SIZE
                ),
                http2=True
            )
        return self._client
    
    async def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Get embedding for text using configured provider.
        
        Returns:
            Normalized embedding vector or None on error
        """
        if not text or not text.strip():
            logger.warning("Empty text provided")
            return None
        
        try:
            if self.provider_type == "local":
                return await self._get_local_embedding(text)
            elif self.provider_type == "api":
                return await self._get_api_embedding(text)
            elif self.provider_type == "pinecone":
                return await self._get_pinecone_embedding(text)
            else:
                logger.error(f"Invalid provider: {self.provider_type}")
                return None
        except Exception as e:
            logger.error(f"💥 Embedding error: {e}", exc_info=True)
            return None
    
    async def _get_local_embedding(self, text: str) -> np.ndarray:
        """Get embedding from local model"""
        if not self._local_model:
            raise RuntimeError("Local model not initialized")
        
        max_len = 512
        truncated = text[:max_len * 4]
        
        try:
            embedding = self._local_model.encode(
                truncated,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            embedding = embedding.astype(np.float32)
            logger.debug(f"✅ Local embedding (dim: {len(embedding)})")
            return embedding
        except Exception as e:
            logger.error(f"❌ Local embedding failed: {e}")
            raise
    
    async def _get_pinecone_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from Pinecone inference API"""
        client = await self._get_client()
        
        url = "https://api.pinecone.io/inference/embed"
        headers = {
            "Api-Key": settings.PINECONE_API_KEY,
            "Content-Type": "application/json",
            "X-Pinecone-API-Version": "2024-10"
        }
        data = {
            "model": settings.EMBEDDING_MODEL,
            "parameters": {"input_type": "passage"},
            "inputs": [{"text": text}]
        }
        
        try:
            response = await client.post(url, json=data, headers=headers)
            
            if response.status_code != 200:
                error_text = await response.aread()
                logger.error(f"❌ Pinecone API error {response.status_code}: {error_text.decode()[:200]}")
                return None
            
            result = response.json()
            
            if not result.get("data") or len(result["data"]) == 0:
                logger.error("Invalid Pinecone response")
                return None
            
            embedding_list = result["data"][0].get("values")
            
            if not embedding_list:
                logger.error("Missing values in Pinecone response")
                return None
            
            embedding = np.array(embedding_list, dtype=np.float32)
            
            norm = np.linalg.norm(embedding)
            if norm > 1e-6:
                embedding = embedding / norm
            
            logger.debug(f"✅ Pinecone embedding (dim: {len(embedding)})")
            return embedding
        
        except httpx.TimeoutException:
            logger.error("⏱️ Pinecone API timeout")
            return None
        except Exception as e:
            logger.error(f"💥 Pinecone error: {e}", exc_info=True)
            return None
    
    async def _get_api_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from OpenAI-compatible API"""
        client = await self._get_client()
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.EMBEDDING_API_KEY}"
        }
        data = {
            "input": text,
            "model": settings.EMBEDDING_MODEL
        }
        
        try:
            response = await client.post(
                settings.EMBEDDING_API_URL,
                json=data,
                headers=headers
            )
            
            if response.status_code != 200:
                error_text = await response.aread()
                logger.error(f"❌ API error {response.status_code}: {error_text.decode()[:200]}")
                return None
            
            result = response.json()
            
            if not result.get("data") or len(result["data"]) == 0:
                logger.error("Invalid API response")
                return None
            
            embedding_list = result["data"][0].get("embedding")
            
            if not embedding_list:
                logger.error("Missing embedding in API response")
                return None
            
            embedding = np.array(embedding_list, dtype=np.float32)
            
            norm = np.linalg.norm(embedding)
            if norm > 1e-6:
                embedding = embedding / norm
            
            logger.debug(f"✅ API embedding (dim: {len(embedding)})")
            return embedding
        
        except httpx.TimeoutException:
            logger.error("⏱️ API timeout")
            return None
        except Exception as e:
            logger.error(f"💥 API error: {e}", exc_info=True)
            return None
    
    async def close(self):
        """Close HTTP client"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            logger.debug("🔌 Closed embedding client")
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
