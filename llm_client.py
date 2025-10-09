import httpx
import asyncio
import logging
from typing import Optional
from config import settings

logger = logging.getLogger(__name__)

class LLMClient:
    """
    Async LLM client with connection pooling and HTTP/2.
    
    Optimizations:
    - Connection pooling (20 keepalive, 100 max)
    - HTTP/2 support
    - Exponential backoff retry logic
    - Timeout management
    """
    
    def __init__(self):
        self.api_url = settings.LLM_API_URL
        self.api_key = settings.LLM_API_KEY
        self.model = settings.LLM_MODEL
        self.max_retries = settings.MAX_RETRIES
        self.retry_delay = settings.RETRY_DELAY
        self.timeout = settings.LLM_TIMEOUT
        self._client: Optional[httpx.AsyncClient] = None
        
        logger.info(f"⚡ LLM client initialized: {self.model}")
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client with pooling"""
        if not self._client or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout, connect=10.0),
                limits=httpx.Limits(
                    max_keepalive_connections=settings.KEEPALIVE_CONNECTIONS,
                    max_connections=settings.CONNECTION_POOL_SIZE
                ),
                http2=True  # Enable HTTP/2 for performance
            )
        return self._client
    
    async def query(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2048
    ) -> Optional[str]:
        """
        Query LLM with retry logic and connection pooling.
        
        Returns:
            LLM response text or None on failure
        """
        if not system_prompt or not user_prompt:
            logger.error("Empty prompt provided")
            return None
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        client = await self._get_client()
        
        for attempt in range(self.max_retries):
            try:
                response = await client.post(
                    self.api_url,
                    json=data,
                    headers=headers
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if result.get("choices") and len(result["choices"]) > 0:
                        message = result["choices"][0].get("message", {})
                        content = message.get("content")
                        
                        if content:
                            logger.debug(f"✅ LLM response: {len(content)} chars")
                            return content
                        else:
                            logger.error("LLM response missing content")
                            return None
                    else:
                        logger.error("LLM response missing choices")
                        return None
                
                elif response.status_code == 429:
                    logger.warning(f"⏳ Rate limited (429), attempt {attempt + 1}/{self.max_retries}")
                    if attempt < self.max_retries - 1:
                        wait_time = self.retry_delay * (2 ** attempt)
                        await asyncio.sleep(wait_time)
                        continue
                    return None
                
                elif response.status_code in [500, 502, 503, 504]:
                    logger.warning(f"🔥 Server error ({response.status_code}), attempt {attempt + 1}/{self.max_retries}")
                    if attempt < self.max_retries - 1:
                        wait_time = self.retry_delay * (2 ** attempt)
                        await asyncio.sleep(wait_time)
                        continue
                    return None
                
                else:
                    error_text = await response.aread()
                    logger.error(f"❌ LLM error ({response.status_code}): {error_text.decode()[:200]}")
                    return None
            
            except httpx.TimeoutException:
                logger.warning(f"⏱️ Timeout, attempt {attempt + 1}/{self.max_retries}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                    continue
                return None
            
            except Exception as e:
                logger.error(f"💥 Unexpected error: {e}", exc_info=True)
                return None
        
        logger.error("❌ LLM query failed after all retries")
        return None
    
    async def close(self):
        """Close HTTP client"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            logger.debug("🔌 Closed LLM client")
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
