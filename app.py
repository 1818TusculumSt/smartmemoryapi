from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import logging

from memory_engine import MemoryEngine
from config import settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SmartMemory API",
    description="Auto-extracting memory system with semantic search powered by Pinecone",
    version="1.0.0",
    servers=[{"url": "http://localhost:8099"}]
)

# Initialize memory engine
try:
    memory_engine = MemoryEngine()
    logger.info("Memory engine initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize memory engine: {e}")
    memory_engine = None

# Request/Response Models
class ExtractRequest(BaseModel):
    user_message: str = Field(..., description="User message to extract memories from")
    recent_history: Optional[List[str]] = Field(
        default=None, 
        description="Recent conversation history for context (optional)"
    )

class ExtractResponse(BaseModel):
    memories_stored: int
    memories: List[dict]

class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    limit: int = Field(default=5, ge=1, le=50, description="Max results to return")

class SearchResponse(BaseModel):
    memories: List[dict]
    count: int

class RelevantMemoriesRequest(BaseModel):
    current_message: str = Field(..., description="Current message to find relevant memories for")
    limit: int = Field(default=5, ge=1, le=20, description="Max memories to return")

class DeleteRequest(BaseModel):
    memory_id: str = Field(..., description="ID of memory to delete")

class DeleteResponse(BaseModel):
    status: str
    memory_id: str
    message: str

class StatusResponse(BaseModel):
    status: str
    total_memories: int
    config: dict

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str

# Endpoints
@app.get("/", include_in_schema=True)
async def root():
    """Root endpoint with service information"""
    return {
        "service": "SmartMemory API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "docs": "/docs",
            "openapi": "/openapi.json",
            "health": "/health",
            "extract": "/extract",
            "search": "/search",
            "relevant": "/relevant",
            "delete": "/memory/{memory_id}",
            "status": "/status"
        }
    }

@app.post("/extract", response_model=ExtractResponse, tags=["Memory Operations"])
async def extract_memories(request: ExtractRequest):
    """
    Extract and store memories from user message using LLM analysis.
    
    This endpoint:
    1. Analyzes the user message with an LLM to identify persistent facts
    2. Filters memories by confidence threshold
    3. Deduplicates against existing memories
    4. Stores unique memories in Pinecone
    5. Automatically prunes if memory limit exceeded
    """
    if not memory_engine:
        raise HTTPException(status_code=503, detail="Memory engine not initialized")
    
    try:
        logger.info(f"Extracting memories from message: {request.user_message[:100]}...")
        
        result = await memory_engine.extract_and_store(
            user_message=request.user_message,
            recent_history=request.recent_history or []
        )
        
        return ExtractResponse(
            memories_stored=len(result),
            memories=result,
        )
    except Exception as e:
        logger.error(f"Extraction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

@app.post("/search", response_model=SearchResponse, tags=["Memory Retrieval"])
async def search_memories(request: SearchRequest):
    """
    Search memories using semantic similarity.
    
    Returns memories ranked by cosine similarity to the query.
    """
    if not memory_engine:
        raise HTTPException(status_code=503, detail="Memory engine not initialized")
    
    try:
        logger.info(f"Searching memories for: {request.query[:100]}...")
        
        results = await memory_engine.search(
            query=request.query,
            limit=request.limit
        )
        
        return SearchResponse(
            memories=results,
            count=len(results)
        )
    except Exception as e:
        logger.error(f"Search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/relevant", response_model=SearchResponse, tags=["Memory Retrieval"])
async def get_relevant_memories(request: RelevantMemoriesRequest):
    """
    Get memories relevant to current context.
    
    Returns only memories above the relevance threshold, sorted by relevance score.
    """
    if not memory_engine:
        raise HTTPException(status_code=503, detail="Memory engine not initialized")
    
    try:
        logger.info(f"Getting relevant memories for: {request.current_message[:100]}...")
        
        results = await memory_engine.get_relevant(
            current_message=request.current_message,
            limit=request.limit
        )
        
        return SearchResponse(
            memories=results,
            count=len(results)
        )
    except Exception as e:
        logger.error(f"Relevance error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Relevance check failed: {str(e)}")

@app.delete("/memory/{memory_id}", response_model=DeleteResponse, tags=["Memory Operations"])
async def delete_memory(memory_id: str):
    """
    Delete a specific memory by ID.
    """
    if not memory_engine:
        raise HTTPException(status_code=503, detail="Memory engine not initialized")
    
    try:
        logger.info(f"Deleting memory: {memory_id}")
        
        success = await memory_engine.delete(memory_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Memory not found: {memory_id}")
        
        return DeleteResponse(
            status="deleted",
            memory_id=memory_id,
            message=f"Memory {memory_id} successfully deleted"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")

@app.get("/status", response_model=StatusResponse, tags=["System"])
async def get_status():
    """
    Get system status and configuration.
    """
    if not memory_engine:
        raise HTTPException(status_code=503, detail="Memory engine not initialized")
    
    try:
        stats = await memory_engine.get_stats()
        
        return StatusResponse(
            status="operational",
            total_memories=stats["count"],
            config={
                "embedding_provider": settings.EMBEDDING_PROVIDER,
                "embedding_model": settings.EMBEDDING_MODEL,
                "llm_model": settings.LLM_MODEL,
                "max_memories": settings.MAX_MEMORIES,
                "dedup_threshold": settings.DEDUP_THRESHOLD,
                "min_confidence": settings.MIN_CONFIDENCE,
                "relevance_threshold": settings.RELEVANCE_THRESHOLD
            }
        )
    except Exception as e:
        logger.error(f"Status error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint for monitoring.
    """
    return HealthResponse(
        status="healthy" if memory_engine else "degraded",
        service="smartmemory",
        version="1.0.0"
    )

@app.get("/openapi.json", include_in_schema=False)
async def get_openapi_spec():
    """
    Return OpenAPI specification for Open WebUI integration.
    """
    return app.openapi()

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("SmartMemory API starting up...")
    logger.info(f"Embedding provider: {settings.EMBEDDING_PROVIDER}")
    logger.info(f"LLM model: {settings.LLM_MODEL}")
    logger.info(f"Pinecone index: {settings.PINECONE_INDEX_NAME}")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("SmartMemory API shutting down...")
