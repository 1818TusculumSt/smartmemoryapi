from pydantic_settings import BaseSettings
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    """SmartMemory 2.0 Configuration"""
    
    # Pinecone Configuration
    PINECONE_API_KEY: str
    PINECONE_ENVIRONMENT: str = "us-east-1-aws"
    PINECONE_INDEX_NAME: str = "smartmemory-v2"
    
    # LLM Provider Configuration
    LLM_API_URL: str
    LLM_API_KEY: str
    LLM_MODEL: str = "gpt-4o-mini"
    
    # Embedding Provider Configuration
    EMBEDDING_PROVIDER: str = "pinecone"  # "local", "api", "pinecone"
    EMBEDDING_MODEL: str = "llama-text-embed-v2"
    EMBEDDING_API_URL: Optional[str] = None
    EMBEDDING_API_KEY: Optional[str] = None
    
    # Memory Management Settings
    MAX_MEMORIES: int = 1000
    DEDUP_THRESHOLD: float = 0.95
    MIN_CONFIDENCE: float = 0.6
    RELEVANCE_THRESHOLD: float = 0.65
    
    # Memory Categories (Mem0-style)
    MEMORY_CATEGORIES: List[str] = [
        "personal_information",
        "food_preferences", 
        "goals",
        "relationships",
        "behavior",
        "preferences",
        "hobbies",
        "work",
        "health",
        "likes",
        "dislikes",
        "skills"
    ]
    
    # Performance Settings
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 1.0
    LLM_TIMEOUT: int = 60
    EMBEDDING_TIMEOUT: int = 30
    CONNECTION_POOL_SIZE: int = 100
    KEEPALIVE_CONNECTIONS: int = 20
    
    # Pagination
    DEFAULT_PAGE_SIZE: int = 50
    MAX_PAGE_SIZE: int = 100
    
    # Batch Operations
    MAX_BATCH_SIZE: int = 1000
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    # Versioning
    ENABLE_MEMORY_HISTORY: bool = True
    MAX_HISTORY_VERSIONS: int = 10
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._validate_settings()
    
    def _validate_settings(self):
        """Validate settings and log warnings"""
        
        # Validate embedding provider
        if self.EMBEDDING_PROVIDER not in ["local", "api", "pinecone"]:
            logger.warning(
                f"Invalid EMBEDDING_PROVIDER '{self.EMBEDDING_PROVIDER}'. "
                f"Defaulting to 'pinecone'"
            )
            self.EMBEDDING_PROVIDER = "pinecone"
        
        # Validate thresholds
        if not 0.0 <= self.DEDUP_THRESHOLD <= 1.0:
            self.DEDUP_THRESHOLD = 0.95
        if not 0.0 <= self.MIN_CONFIDENCE <= 1.0:
            self.MIN_CONFIDENCE = 0.6
        if not 0.0 <= self.RELEVANCE_THRESHOLD <= 1.0:
            self.RELEVANCE_THRESHOLD = 0.65
        
        # Validate MAX_MEMORIES
        if self.MAX_MEMORIES < 1:
            self.MAX_MEMORIES = 1000
        
        # Log configuration
        logger.info("🚀 SmartMemory 2.0 Configuration Loaded")
        logger.info(f"  Embedding Provider: {self.EMBEDDING_PROVIDER}")
        logger.info(f"  Embedding Model: {self.EMBEDDING_MODEL}")
        logger.info(f"  LLM Model: {self.LLM_MODEL}")
        logger.info(f"  Pinecone Index: {self.PINECONE_INDEX_NAME}")
        logger.info(f"  Max Memories: {self.MAX_MEMORIES}")
        logger.info(f"  History Enabled: {self.ENABLE_MEMORY_HISTORY}")

try:
    settings = Settings()
except Exception as e:
    logger.error(f"Failed to load settings: {e}")
    raise
