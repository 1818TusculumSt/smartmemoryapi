import json
import re
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np
from pinecone import Pinecone, ServerlessSpec
import time

from embeddings import EmbeddingProvider
from llm_client import LLMClient
from config import settings

logger = logging.getLogger(__name__)

class MemoryEngine:
    """
    SmartMemory 2.0 Core Engine
    
    Features:
    - Async operations throughout
    - Memory categories (Mem0-style)
    - Session support (run_id)
    - Advanced filtering (AND/OR/NOT)
    - Pagination
    - Batch operations
    - Memory history/versioning
    - Silent operations
    """
    
    def __init__(self):
        self.embedder = EmbeddingProvider()
        self.llm = LLMClient()
        self.embedding_dim = None
        
        logger.info("🚀 Initializing SmartMemory 2.0 Engine")
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        
        index_name = settings.PINECONE_INDEX_NAME
        self._determine_embedding_dimension()
        
        if index_name not in self.pc.list_indexes().names():
            logger.info(f"📦 Creating Pinecone index: {index_name}")
            self.pc.create_index(
                name=index_name,
                dimension=self.embedding_dim,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region=settings.PINECONE_ENVIRONMENT
                )
            )
            logger.info(f"✅ Index created (dim: {self.embedding_dim})")
        
        self.index = self.pc.Index(index_name)
        logger.info(f"✅ Connected to index: {index_name}")
    
    def _determine_embedding_dimension(self):
        """Determine embedding dimension based on provider"""
        model_dims = {
            # Local models
            "all-MiniLM-L6-v2": 384,
            "all-mpnet-base-v2": 768,
            "all-MiniLM-L12-v2": 384,
            # Pinecone models (llama-text-embed-v2 supports 384, 512, 768, 1024, 2048)
            "llama-text-embed-v2": 384,  # Using 384 to match index
            "multilingual-e5-large": 1024,
        }

        if settings.EMBEDDING_PROVIDER == "local":
            self.embedding_dim = model_dims.get(settings.EMBEDDING_MODEL, 384)
        elif settings.EMBEDDING_PROVIDER == "pinecone":
            self.embedding_dim = model_dims.get(settings.EMBEDDING_MODEL, 384)
        else:
            self.embedding_dim = 1536  # OpenAI default

        logger.info(f"📏 Embedding dimension: {self.embedding_dim}")
    
    async def extract_and_store(
        self,
        user_message: str,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        recent_history: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Extract and store memories.
        
        Args:
            user_message: Message to analyze
            user_id: User identifier (for user memories)
            agent_id: Agent identifier (for agent memories)
            run_id: Session identifier (for session memories)
            metadata: Additional metadata to store
            recent_history: Recent conversation context
        
        Returns:
            Silent response: {"ok": True, "count": N}
        """
        logger.info(f"🔍 Extracting memories: {user_message[:80]}...")
        
        # Extract memories using LLM
        memories = await self._extract_memories(user_message, recent_history or [])
        
        if not memories:
            logger.info("💭 No memories extracted")
            return {"ok": True, "count": 0}
        
        logger.info(f"📝 Extracted {len(memories)} potential memories")
        
        # Filter by confidence
        filtered = [
            m for m in memories 
            if m.get("confidence", 0) >= settings.MIN_CONFIDENCE
        ]
        
        if not filtered:
            logger.info("🚫 No memories passed confidence filter")
            return {"ok": True, "count": 0}
        
        # Deduplicate
        deduplicated = await self._deduplicate(filtered, user_id, agent_id, run_id)
        
        if not deduplicated:
            logger.info("🔄 All memories were duplicates")
            return {"ok": True, "count": 0}
        
        # Store memories
        stored_count = 0
        for memory in deduplicated:
            result = await self._store_memory(
                memory,
                user_id=user_id,
                agent_id=agent_id,
                run_id=run_id,
                metadata=metadata
            )
            if result:
                stored_count += 1
        
        logger.info(f"✅ Stored {stored_count} new memories")
        
        # Prune if needed
        await self._prune_if_needed()
        
        # SILENT RESPONSE (Mem0-style)
        return {"ok": True, "count": stored_count}
    
    async def _extract_memories(
        self,
        user_message: str,
        recent_history: List[str]
    ) -> List[Dict[str, Any]]:
        """Extract memories using LLM with category support"""
        
        categories_str = ", ".join(settings.MEMORY_CATEGORIES)
        
        system_prompt = f"""You are a memory extraction system. Extract user-specific facts, preferences, goals, relationships, and persistent information from natural conversation.

CRITICAL OUTPUT REQUIREMENTS:
1. Your ENTIRE response MUST be ONLY a valid JSON array: [...]
2. NO text before or after the JSON array
3. NO markdown code blocks (no ```json)
4. NO explanations or notes
5. Return [] if no user-specific memories found

Each memory object MUST have:
- "content": the extracted user-specific fact (string)
- "categories": array of relevant categories from: {categories_str}
- "confidence": score from 0.0 to 1.0 indicating certainty

EXTRACT (be generous - catch subtle patterns):
- Direct statements: "I love X", "My favorite is Y", "I hate Z"
- Implied preferences: "tried X, it was good", "X didn't work out", "might check out Y"
- Identity/background: name, location, profession, age, family structure
- Goals and plans: "working on X", "planning to Y", "want to learn Z"
- Relationships: mentions of people (names, roles, context)
- Possessions and environment: "my X", "just got Y", "have Z at home"
- Behavioral patterns: habits, routines, tendencies
- Skills and experience: "used to do X", "know how to Y", "worked with Z"
- Reactions and opinions: "X was great", "Y didn't help", "Z is overrated"
- Context about life: work situation, living situation, challenges, interests

CONFIDENCE SCORING:
- 0.9-1.0: Explicit, clear, unambiguous ("I am a software engineer")
- 0.7-0.9: Strong implication, clear context ("been coding for 10 years")
- 0.5-0.7: Implied preference or pattern ("tried that approach, worked well")
- 0.3-0.5: Weak signal, ambiguous (use sparingly)

DO NOT EXTRACT:
- Pure questions without context
- General knowledge or facts about the world
- Temporary states ("I'm tired right now")
- Information solely about the AI

EXAMPLE OUTPUT:
[
  {{
    "content": "User tried pineapple pizza and found it acceptable",
    "categories": ["food_preferences"],
    "confidence": 0.65
  }},
  {{
    "content": "User is a software engineer",
    "categories": ["work", "personal_information"],
    "confidence": 0.95
  }},
  {{
    "content": "User prefers working from home",
    "categories": ["work", "preferences"],
    "confidence": 0.8
  }}
]

If no user-specific memories found, return: []"""

        # Build context
        context = ""
        if recent_history:
            context = "Recent conversation:\n"
            for msg in recent_history[-3:]:
                context += f"- {msg}\n"
            context += "\n"
        
        user_prompt = f"""{context}Analyze this user message and extract memories:

"{user_message}"

Return ONLY the JSON array. No other text."""

        logger.debug("🤖 Calling LLM for extraction")
        response = await self.llm.query(system_prompt, user_prompt, temperature=0.1)
        
        if not response:
            logger.error("❌ LLM returned no response")
            return []
        
        # Parse JSON
        try:
            cleaned = re.sub(r'```(?:json)?\s*', '', response)
            cleaned = re.sub(r'\s*```', '', cleaned).strip()
            
            memories = json.loads(cleaned)
            
            # Handle wrapped responses
            if isinstance(memories, dict):
                if "memories" in memories:
                    memories = memories["memories"]
                elif len(memories) == 1:
                    memories = list(memories.values())[0]
            
            if not isinstance(memories, list):
                logger.error(f"Invalid format: {type(memories)}")
                return []
            
            # Validate memories
            valid_memories = []
            for mem in memories:
                if not isinstance(mem, dict) or "content" not in mem:
                    continue
                
                # Ensure categories
                if "categories" not in mem:
                    mem["categories"] = ["behavior"]
                elif not isinstance(mem["categories"], list):
                    mem["categories"] = [str(mem["categories"])]
                
                # Ensure confidence
                if "confidence" not in mem:
                    mem["confidence"] = 0.6
                
                valid_memories.append(mem)
            
            logger.info(f"✅ Validated {len(valid_memories)}/{len(memories)} memories")
            return valid_memories
        
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON parse error: {e}")
            return []
        except Exception as e:
            logger.error(f"💥 Unexpected error: {e}", exc_info=True)
            return []    
    async def _deduplicate(
        self,
        memories: List[Dict[str, Any]],
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Deduplicate against existing memories"""
        if not memories:
            return []
        
        logger.debug(f"🔄 Deduplicating {len(memories)} memories")
        
        # Get existing memories for this context
        existing = await self._get_all_memories(
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id
        )
        
        logger.debug(f"📊 Checking against {len(existing)} existing memories")
        
        unique = []
        
        for new_mem in memories:
            content = new_mem.get("content", "").strip()
            if not content:
                continue
            
            # Generate embedding
            new_emb = await self.embedder.get_embedding(content)
            if new_emb is None:
                logger.warning(f"⚠️ Failed embedding: {content[:50]}...")
                continue
            
            # Check similarity
            is_duplicate = False
            max_similarity = 0.0
            
            for exist_mem in existing:
                exist_emb = np.array(exist_mem["values"], dtype=np.float32)
                similarity = float(np.dot(new_emb, exist_emb))
                max_similarity = max(max_similarity, similarity)
                
                if similarity >= settings.DEDUP_THRESHOLD:
                    logger.debug(f"🔄 Duplicate (sim={similarity:.3f}): {content[:50]}...")
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                logger.debug(f"✨ Unique (max_sim={max_similarity:.3f}): {content[:50]}...")
                new_mem["embedding"] = new_emb
                unique.append(new_mem)
        
        logger.info(f"✅ Deduplication: {len(unique)}/{len(memories)} unique")
        return unique
    
    async def _store_memory(
        self,
        memory: Dict[str, Any],
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Store memory in Pinecone with full metadata"""
        content = memory.get("content", "").strip()
        embedding = memory.get("embedding")
        
        if not content:
            return None
        
        # Generate embedding if not provided
        if embedding is None:
            embedding = await self.embedder.get_embedding(content)
        
        if embedding is None:
            logger.error(f"❌ Failed embedding for: {content[:50]}...")
            return None
        
        # Generate ID
        timestamp = int(time.time() * 1000)
        content_hash = abs(hash(content)) % 10000
        memory_id = f"mem_{timestamp}_{content_hash}"
        
        # Build metadata (Mem0-style)
        mem_metadata = {
            "content": content,
            "categories": ",".join(memory.get("categories", [])),
            "confidence": float(memory.get("confidence", 0.6)),
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "version": 1
        }
        
        # Add identifiers
        if user_id:
            mem_metadata["user_id"] = user_id
        if agent_id:
            mem_metadata["agent_id"] = agent_id
        if run_id:
            mem_metadata["run_id"] = run_id
        
        # Add custom metadata
        if metadata:
            mem_metadata.update(metadata)
        
        try:
            self.index.upsert(
                vectors=[(memory_id, embedding.tolist(), mem_metadata)]
            )
            
            logger.info(f"💾 Stored [{memory_id}]: {content[:60]}...")
            return memory_id
        
        except Exception as e:
            logger.error(f"💥 Storage failed: {e}", exc_info=True)
            return None
    
    async def search(
        self,
        query: str,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        categories: Optional[List[str]] = None,
        limit: int = 5,
        page: int = 1,
        page_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """Search memories with pagination and filtering"""
        logger.debug(f"🔍 Searching: {query[:80]}...")
        
        # Generate query embedding
        query_emb = await self.embedder.get_embedding(query)
        
        if query_emb is None:
            logger.error("❌ Failed query embedding")
            return {
                "memories": [],
                "count": 0,
                "page": page,
                "page_size": page_size or settings.DEFAULT_PAGE_SIZE,
                "total_count": 0,
                "has_more": False
            }
        
        # Build filter
        filter_dict = {}
        if user_id:
            filter_dict["user_id"] = user_id
        if agent_id:
            filter_dict["agent_id"] = agent_id
        if run_id:
            filter_dict["run_id"] = run_id
        
        try:
            # Query Pinecone
            results = self.index.query(
                vector=query_emb.tolist(),
                top_k=min(limit * 2, 100),
                include_metadata=True,
                filter=filter_dict if filter_dict else None
            )
            
            memories = []
            for match in results.matches:
                memories.append({
                    "id": match.id,
                    "content": match.metadata.get("content", ""),
                    "relevance": float(match.score),
                    "categories": match.metadata.get("categories", "").split(",") if match.metadata.get("categories") else [],
                    "confidence": match.metadata.get("confidence", 0.6),
                    "created_at": match.metadata.get("created_at", ""),
                    "user_id": match.metadata.get("user_id"),
                    "agent_id": match.metadata.get("agent_id"),
                    "run_id": match.metadata.get("run_id")
                })
            
            # Apply pagination
            page_size = page_size or settings.DEFAULT_PAGE_SIZE
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            
            paginated = memories[start_idx:end_idx]
            
            logger.info(f"✅ Search returned {len(paginated)} results")
            
            return {
                "memories": paginated,
                "count": len(paginated),
                "page": page,
                "page_size": page_size,
                "total_count": len(memories),
                "has_more": end_idx < len(memories)
            }
        
        except Exception as e:
            logger.error(f"💥 Search failed: {e}", exc_info=True)
            return {
                "memories": [],
                "count": 0,
                "page": page,
                "page_size": page_size or settings.DEFAULT_PAGE_SIZE,
                "total_count": 0,
                "has_more": False
            }
    
    async def get_relevant(
        self,
        current_message: str,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get relevant memories above threshold"""
        results = await self.search(
            query=current_message,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            limit=limit * 2
        )
        
        # Filter by relevance threshold
        filtered = [
            m for m in results["memories"]
            if m["relevance"] >= settings.RELEVANCE_THRESHOLD
        ]
        
        relevant = filtered[:limit]
        
        logger.info(
            f"🎯 Relevant: {len(relevant)}/{len(results['memories'])} "
            f"(threshold: {settings.RELEVANCE_THRESHOLD})"
        )
        
        return relevant
    
    async def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID"""
        try:
            self.index.delete(ids=[memory_id])
            logger.info(f"🗑️ Deleted: {memory_id}")
            return True
        except Exception as e:
            logger.error(f"💥 Delete failed: {e}")
            return False
    
    async def batch_delete(self, memory_ids: List[str]) -> Dict[str, Any]:
        """Batch delete memories"""
        if len(memory_ids) > settings.MAX_BATCH_SIZE:
            return {
                "success": False,
                "error": f"Batch size exceeds limit of {settings.MAX_BATCH_SIZE}"
            }
        
        try:
            self.index.delete(ids=memory_ids)
            logger.info(f"🗑️ Batch deleted {len(memory_ids)} memories")
            return {"success": True, "deleted_count": len(memory_ids)}
        except Exception as e:
            logger.error(f"💥 Batch delete failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        try:
            stats = self.index.describe_index_stats()
            return {
                "total_count": stats.total_vector_count,
                "dimension": stats.dimension
            }
        except Exception as e:
            logger.error(f"💥 Stats failed: {e}")
            return {"total_count": 0, "dimension": 0}
    
    async def _get_all_memories(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get all memories for deduplication"""
        try:
            dummy = np.zeros(self.embedding_dim, dtype=np.float32)
            
            filter_dict = {}
            if user_id:
                filter_dict["user_id"] = user_id
            if agent_id:
                filter_dict["agent_id"] = agent_id
            if run_id:
                filter_dict["run_id"] = run_id
            
            results = self.index.query(
                vector=dummy.tolist(),
                top_k=settings.MAX_MEMORIES,
                include_values=True,
                include_metadata=True,
                filter=filter_dict if filter_dict else None
            )
            
            memories = []
            for match in results.matches:
                memories.append({
                    "id": match.id,
                    "values": match.values,
                    "metadata": match.metadata
                })
            
            return memories
        
        except Exception as e:
            logger.error(f"💥 Failed to get all memories: {e}")
            return []
    
    async def _prune_if_needed(self):
        """Prune oldest memories if over limit"""
        stats = await self.get_stats()
        count = stats["total_count"]
        
        if count <= settings.MAX_MEMORIES:
            return
        
        to_delete = count - settings.MAX_MEMORIES
        logger.info(f"✂️ Pruning {to_delete} old memories (limit: {settings.MAX_MEMORIES})")
        
        # Get all and sort by timestamp
        all_mems = await self._get_all_memories()
        all_mems.sort(key=lambda x: x["metadata"].get("created_at", ""))
        
        # Delete oldest
        for mem in all_mems[:to_delete]:
            await self.delete(mem["id"])
        
        logger.info(f"✅ Pruned {to_delete} memories")
