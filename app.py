from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging
from contextlib import asynccontextmanager

from memory_engine import MemoryEngine
from config import settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global memory engine
memory_engine: Optional[MemoryEngine] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage memory engine lifecycle"""
    global memory_engine
    try:
        memory_engine = MemoryEngine()
        logger.info("✅ SmartMemory 2.0 engine initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize engine: {e}")
        memory_engine = None
    
    yield
    
    # Cleanup
    if memory_engine:
        if hasattr(memory_engine, 'embedder'):
            await memory_engine.embedder.close()
        if hasattr(memory_engine, 'llm'):
            await memory_engine.llm.close()
    logger.info("🔌 SmartMemory 2.0 shutting down")

app = FastAPI(
    title="SmartMemory 2.0 API",
    description="Next-gen memory system with Mem0 features + enhancements",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# REQUEST MODELS
class AddMemoryRequest(BaseModel):
    """Add memory request - saves user information from conversation"""
    messages: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Message history array with role/content pairs (Mem0 format). Example: [{'role': 'user', 'content': 'I love pizza'}]"
    )
    user_message: Optional[str] = Field(
        default=None,
        description="Direct user message to extract memories from. Use this for single messages. Example: 'I love pizza'"
    )
    user_id: Optional[str] = Field(
        default=None,
        description="Unique identifier for the user. Use this to isolate memories per user in multi-user systems."
    )
    agent_id: Optional[str] = Field(
        default=None,
        description="Agent/assistant identifier. Use to track which agent interacted with user."
    )
    run_id: Optional[str] = Field(
        default=None,
        description="Session/conversation identifier. Use to group memories by conversation session."
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional custom metadata to attach to extracted memories."
    )
    recent_history: Optional[List[str]] = Field(
        default=None,
        description="Recent conversation messages for context. Helps improve extraction quality."
    )

class SearchRequest(BaseModel):
    """Search memories with advanced filtering"""
    query: str = Field(..., description="Search query text. Example: 'food preferences' or 'programming skills'")
    user_id: Optional[str] = Field(default=None, description="Filter memories for specific user")
    agent_id: Optional[str] = Field(default=None, description="Filter memories by agent")
    run_id: Optional[str] = Field(default=None, description="Filter memories by conversation session")
    categories: Optional[List[str]] = Field(default=None, description="Filter by categories. Example: ['food_preferences', 'work']")
    limit: int = Field(default=5, ge=1, le=100, description="Maximum number of results to return (1-100)")
    page: int = Field(default=1, ge=1, description="Page number for pagination")
    page_size: Optional[int] = Field(default=None, ge=1, le=100, description="Results per page")

class GetRelevantRequest(BaseModel):
    """Get context-relevant memories for personalized responses"""
    current_message: str = Field(..., description="Current user message or conversation context to find relevant memories for")
    user_id: Optional[str] = Field(default=None, description="User identifier to retrieve their specific memories")
    agent_id: Optional[str] = Field(default=None, description="Filter by agent identifier")
    run_id: Optional[str] = Field(default=None, description="Filter by conversation session")
    limit: int = Field(default=5, ge=1, le=20, description="Maximum relevant memories to return (1-20)")

class GetRecentRequest(BaseModel):
    """Get recent memories request"""
    limit: int = Field(default=10, ge=1, le=50)
    user_id: Optional[str] = None

class AutoRecallRequest(BaseModel):
    """Auto-recall memories request"""
    conversation_context: str = Field(..., description="Brief summary of current conversation context")
    limit: int = Field(default=5, ge=1, le=20)
    user_id: Optional[str] = None

class ConsolidateRequest(BaseModel):
    """Consolidate memories request"""
    user_id: Optional[str] = None
    tag: Optional[str] = None

class BatchDeleteRequest(BaseModel):
    """Batch delete request"""
    memory_ids: List[str] = Field(..., description="List of memory IDs to delete")

class StatusResponse(BaseModel):
    """System status response"""
    status: str
    version: str
    total_memories: int
    config: Dict[str, Any]

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    service: str
    version: str

# ENDPOINTS
@app.get("/", include_in_schema=True)
async def root():
    """Root endpoint"""
    return {
        "service": "SmartMemory 2.0 API",
        "version": "2.0.0",
        "status": "operational",
        "features": [
            "Silent operations",
            "Smart memory updates (auto-evolving memories)",
            "Hybrid search (semantic + keyword boost)",
            "Temporal awareness (date context)",
            "Memory consolidation (merge fragments)",
            "Proactive memory recall (auto-recall)",
            "Recent memories retrieval",
            "Memory categories/tags",
            "Session support (run_id)",
            "Advanced filtering",
            "Pagination",
            "Batch operations",
            "Async/HTTP2 optimized"
        ],
        "endpoints": {
            "docs": "/docs",
            "openapi": "/openapi.json",
            "health": "/health",
            "status": "/status",
            "add": "/add",
            "search": "/search",
            "relevant": "/relevant",
            "recent": "/recent",
            "auto_recall": "/auto-recall",
            "consolidate": "/consolidate",
            "delete": "/memory/{memory_id}",
            "batch_delete": "/batch/delete"
        }
    }

@app.post("/add", tags=["Memory Operations"], summary="Persist user context")
async def add_memory(request: AddMemoryRequest):
    """
    Extracts and stores user context. Use extracted information immediately in responses
    for personalization and continuity without announcing the storage operation.

    Trigger: User shares preferences, identity details, goals, relationships, opinions,
    habits, skills, work context, or life situation.

    Scope: Personal facts, persistent preferences, identity, goals, relationships.
    Excluded: Temporary queries, general knowledge, greetings.

    Processing: LLM extraction, confidence filtering, deduplication, automatic updates.

    Usage: Call when context is shared. Incorporate extracted facts naturally into
    ongoing conversation responses. Do not announce or confirm the storage operation.

    Returns: {ok: bool, stored: int}
    """
    if not memory_engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    # Extract user message
    user_message = None
    if request.user_message:
        user_message = request.user_message
    elif request.messages:
        # Extract from messages array (Mem0 format)
        user_messages = [
            msg.get("content", "") 
            for msg in request.messages 
            if msg.get("role") == "user"
        ]
        if user_messages:
            user_message = " ".join(user_messages)
    
    if not user_message:
        raise HTTPException(status_code=400, detail="No message content provided")
    
    try:
        result = await memory_engine.extract_and_store(
            user_message=user_message,
            user_id=request.user_id,
            agent_id=request.agent_id,
            run_id=request.run_id,
            metadata=request.metadata,
            recent_history=request.recent_history or []
        )

        # Silent response with count (LLMs typically ignore metadata-like fields)
        return {"ok": True, "stored": result.get("count", 0)}
    
    except Exception as e:
        logger.error(f"💥 Add memory error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", tags=["Memory Retrieval"], summary="Search memories with filters")
async def search_memories(request: SearchRequest):
    """
    🔎 Advanced memory search with filtering, categories, and pagination.

    **When to call this tool:**
    - User explicitly asks to search their memories: "Search my memories for..."
    - User asks about specific topics: "What do you know about my work?"
    - User wants to filter by category: "Show my food preferences"
    - User asks "What do you remember about X?"
    - Need to find specific memories with more control than /relevant

    **Features:**
    - Hybrid search: semantic similarity + keyword matching
    - Filter by user_id, agent_id, run_id, categories
    - Pagination support for large result sets
    - Returns memories sorted by relevance

    **Difference from /relevant:**
    - /relevant: Auto-filtered by threshold, best for proactive context injection
    - /search: Returns all matches, best for explicit user queries

    **Examples when to call:**
    - ✅ User: "What do you know about my food preferences?"
    - ✅ User: "Search for memories about Python"
    - ✅ User: "Show me what you remember about my job"
    - ✅ User: "List all my goals"
    - ❌ Just personalizing a response → Use /relevant instead
    """
    if not memory_engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    try:
        result = await memory_engine.search(
            query=request.query,
            user_id=request.user_id,
            agent_id=request.agent_id,
            run_id=request.run_id,
            categories=request.categories,
            limit=request.limit,
            page=request.page,
            page_size=request.page_size
        )
        
        # Return just memories, no count or pagination metadata
        return {"memories": result["memories"]}
    
    except Exception as e:
        logger.error(f"💥 Search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/relevant", tags=["Memory Retrieval"], summary="Retrieve context-relevant memories")
async def get_relevant_memories(request: GetRelevantRequest):
    """
    Retrieves user context relevant to current message. Incorporate retrieved memories
    naturally into response without announcing the retrieval operation.

    Trigger: Complex conversations requiring user context, questions about preferences,
    personalized recommendations, continued conversations, user profile queries.

    Algorithm: Semantic similarity search with hybrid keyword matching, relevance
    threshold filtering, sorted by relevance score.

    Usage: Call before responding to inject context. Use retrieved information seamlessly
    in response formulation. Do not announce retrieval operation or phrase discoveries
    ("I found", "I recall"). Apply context directly to personalize answer.

    Returns: {memories: []}
    """
    if not memory_engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    try:
        memories = await memory_engine.get_relevant(
            current_message=request.current_message,
            user_id=request.user_id,
            agent_id=request.agent_id,
            run_id=request.run_id,
            limit=request.limit
        )

        # Return just memories list, no count
        return {"memories": memories}

    except Exception as e:
        logger.error(f"💥 Relevant error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recent", tags=["Memory Retrieval"])
async def get_recent_memories(request: GetRecentRequest):
    """
    Get most recent memories sorted by timestamp.

    Useful for seeing latest stored context
    """
    if not memory_engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    try:
        memories = await memory_engine.get_recent(
            limit=request.limit,
            user_id=request.user_id
        )

        return {"memories": memories}

    except Exception as e:
        logger.error(f"💥 Recent error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/auto-recall", tags=["Memory Retrieval"])
async def auto_recall_memories(request: AutoRecallRequest):
    """
    Automatically recall memories relevant to conversation context.

    This is the proactive memory feature - call at the start of responses
    to inject relevant context automatically
    """
    if not memory_engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    try:
        memories = await memory_engine.get_relevant(
            current_message=request.conversation_context,
            user_id=request.user_id,
            limit=request.limit
        )

        return {"memories": memories}

    except Exception as e:
        logger.error(f"💥 Auto-recall error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/consolidate", tags=["Memory Operations"])
async def consolidate_memories(request: ConsolidateRequest):
    """
    Consolidate fragmented memories into coherent summaries.

    Groups similar memories and uses LLM to merge them.
    Reduces redundancy and improves memory quality.
    """
    if not memory_engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    try:
        result = await memory_engine.consolidate_memories(
            user_id=request.user_id,
            tag=request.tag
        )

        return result

    except Exception as e:
        logger.error(f"💥 Consolidate error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/memory/{memory_id}", tags=["Memory Operations"])
async def delete_memory(memory_id: str):
    """Delete a specific memory by ID"""
    if not memory_engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    try:
        success = await memory_engine.delete(memory_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Memory not found: {memory_id}")
        
        # Silent response - just ok
        return {"ok": True}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"💥 Delete error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch/delete", tags=["Batch Operations"])
async def batch_delete_memories(request: BatchDeleteRequest):
    """
    Batch delete memories (up to 1000 at once).
    """
    if not memory_engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    try:
        result = await memory_engine.batch_delete(request.memory_ids)
        
        # Silent response - just ok based on success
        return {"ok": result["success"]}
    
    except Exception as e:
        logger.error(f"💥 Batch delete error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status", response_model=StatusResponse, tags=["System"])
async def get_status():
    """Get system status and configuration"""
    if not memory_engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    try:
        stats = await memory_engine.get_stats()
        
        return StatusResponse(
            status="operational",
            version="2.0.0",
            total_memories=stats["total_count"],
            config={
                "embedding_provider": settings.EMBEDDING_PROVIDER,
                "embedding_model": settings.EMBEDDING_MODEL,
                "llm_model": settings.LLM_MODEL,
                "max_memories": settings.MAX_MEMORIES,
                "categories": settings.MEMORY_CATEGORIES,
                "dedup_threshold": settings.DEDUP_THRESHOLD,
                "min_confidence": settings.MIN_CONFIDENCE,
                "relevance_threshold": settings.RELEVANCE_THRESHOLD,
                "history_enabled": settings.ENABLE_MEMORY_HISTORY
            }
        )
    
    except Exception as e:
        logger.error(f"💥 Status error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if memory_engine else "degraded",
        service="smartmemory-v2",
        version="2.0.0"
    )

@app.get("/openapi.json", include_in_schema=False)
async def get_openapi_spec():
    """Return OpenAPI spec for Open WebUI integration"""
    return app.openapi()
