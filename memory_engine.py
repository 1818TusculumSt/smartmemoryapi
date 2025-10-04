import json
import re
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np
from pinecone import Pinecone, ServerlessSpec
import asyncio
import time

from embeddings import EmbeddingProvider
from llm_client import LLMClient
from config import settings

logger = logging.getLogger(__name__)

class MemoryEngine:
    """
    Core memory management engine.
    
    Handles:
    - Memory extraction from text using LLM
    - Deduplication via embedding similarity
    - Storage in Pinecone vector database
    - Semantic search and retrieval
    - Automatic pruning when memory limit exceeded
    """
    
    def __init__(self):
        self.embedder = EmbeddingProvider()
        self.llm = LLMClient()
        self.embedding_dim = None
        
        # Initialize Pinecone
        logger.info("Initializing Pinecone connection")
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        
        # Get or create index
        index_name = settings.PINECONE_INDEX_NAME
        
        # Determine embedding dimension
        self._determine_embedding_dimension()
        
        if index_name not in self.pc.list_indexes().names():
            logger.info(f"Creating new Pinecone index: {index_name}")
            self.pc.create_index(
                name=index_name,
                dimension=self.embedding_dim,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region=settings.PINECONE_ENVIRONMENT
                )
            )
            logger.info(f"Pinecone index created with dimension {self.embedding_dim}")
        
        self.index = self.pc.Index(index_name)
        logger.info(f"Connected to Pinecone index: {index_name}")
    
    def _determine_embedding_dimension(self):
        """Determine embedding dimension based on provider"""
        if settings.EMBEDDING_PROVIDER == "local":
            # Common dimensions for sentence-transformers models
            model_dims = {
                "all-MiniLM-L6-v2": 384,
                "all-mpnet-base-v2": 768,
                "all-MiniLM-L12-v2": 384,
            }
            self.embedding_dim = model_dims.get(settings.EMBEDDING_MODEL, 384)
            logger.info(f"Using embedding dimension {self.embedding_dim} for {settings.EMBEDDING_MODEL}")
        elif settings.EMBEDDING_PROVIDER == "pinecone":
            # Pinecone inference models
            pinecone_dims = {
                "llama-text-embed-v2": 1024,
                "multilingual-e5-large": 1024,
            }
            self.embedding_dim = pinecone_dims.get(settings.EMBEDDING_MODEL, 1024)
            logger.info(f"Using Pinecone inference dimension {self.embedding_dim} for {settings.EMBEDDING_MODEL}")
        else:
            # API embeddings (OpenAI)
            self.embedding_dim = 1536
            logger.info(f"Using embedding dimension {self.embedding_dim} for API provider")
    
    async def extract_and_store(
        self,
        user_message: str,
        recent_history: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract memories from message and store them.
        
        Args:
            user_message: User's message to analyze
            recent_history: Recent conversation context (optional)
            
        Returns:
            List of successfully stored memories
        """
        logger.info(f"Starting memory extraction for message: {user_message[:100]}...")
        
        # Step 1: Extract memories using LLM
        memories = await self._extract_memories(user_message, recent_history or [])
        
        if not memories:
            logger.info("No memories extracted from message")
            return []
        
        logger.info(f"Extracted {len(memories)} potential memories")
        
        # Step 2: Filter by confidence threshold
        filtered = [
            m for m in memories 
            if m.get("confidence", 0) >= settings.MIN_CONFIDENCE
        ]
        
        discarded = len(memories) - len(filtered)
        if discarded > 0:
            logger.info(f"Filtered out {discarded} low-confidence memories (threshold: {settings.MIN_CONFIDENCE})")
        
        if not filtered:
            logger.info("No memories passed confidence filter")
            return []
        
        # Step 3: Deduplicate against existing memories
        deduplicated = await self._deduplicate(filtered)
        
        duplicates = len(filtered) - len(deduplicated)
        if duplicates > 0:
            logger.info(f"Removed {duplicates} duplicate memories")
        
        if not deduplicated:
            logger.info("All memories were duplicates")
            return []
        
        # Step 4: Store in Pinecone
        stored = []
        for memory in deduplicated:
            result = await self._store_memory(memory)
            if result:
                stored.append(result)
        
        logger.info(f"Successfully stored {len(stored)} new memories")
        
        # Step 5: Prune if needed
        await self._prune_if_needed()
        
        return stored
    
    async def _extract_memories(
        self,
        user_message: str,
        recent_history: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Use LLM to extract memories from text.
        
        Args:
            user_message: Message to analyze
            recent_history: Recent conversation for context
            
        Returns:
            List of extracted memory dictionaries
        """
        system_prompt = """You are a memory extraction system. Extract ONLY user-specific facts, preferences, goals, relationships, and persistent information.

CRITICAL OUTPUT REQUIREMENTS:
1. Your ENTIRE response MUST be ONLY a valid JSON array: [...]
2. NO text before or after the JSON array
3. NO markdown code blocks (no ```json)
4. NO explanations or notes
5. Return [] if no user-specific memories found

Each memory object MUST have:
- "content": the extracted user-specific fact (string)
- "tags": array of relevant tags from: identity, preference, goal, relationship, possession, behavior
- "confidence": score from 0.0 to 1.0 indicating certainty

EXTRACT:
- Explicit user preferences ("I love X", "My favorite is Y")
- Identity details (name, location, profession, age)
- Goals and aspirations
- Relationships (family, friends, colleagues)
- Possessions (things owned or desired)
- Behavioral patterns and interests

DO NOT EXTRACT:
- General knowledge or trivia
- Temporary thoughts or questions
- Information about the AI
- Meta-commentary about remembering

EXAMPLE OUTPUT:
[
  {
    "content": "User has a cat named Whiskers",
    "tags": ["relationship", "possession"],
    "confidence": 0.9
  },
  {
    "content": "User prefers working remotely",
    "tags": ["preference", "behavior"],
    "confidence": 0.75
  }
]

If no user-specific memories are found, return: []"""

        # Build context from recent history
        context = ""
        if recent_history:
            context = "Recent conversation:\n"
            for msg in recent_history[-3:]:
                context += f"- {msg}\n"
            context += "\n"
        
        user_prompt = f"""{context}Analyze this user message and extract memories:

"{user_message}"

Return ONLY the JSON array. No other text."""

        logger.debug("Calling LLM for memory extraction")
        response = await self.llm.query(system_prompt, user_prompt, temperature=0.1)
        
        if not response:
            logger.error("LLM extraction returned no response")
            return []
        
        logger.debug(f"LLM response length: {len(response)} chars")
        
        # Parse JSON response
        try:
            # Clean response - remove markdown code blocks
            cleaned = re.sub(r'```(?:json)?\s*', '', response)
            cleaned = re.sub(r'\s*```', '', cleaned)
            cleaned = cleaned.strip()
            
            logger.debug(f"Cleaned response: {cleaned[:200]}...")
            
            # Parse JSON
            memories = json.loads(cleaned)
            
            # Handle case where LLM wraps in object
            if isinstance(memories, dict):
                if "memories" in memories:
                    memories = memories["memories"]
                elif len(memories) == 1:
                    # Single-key dict, unwrap to list
                    memories = list(memories.values())[0]
            
            if not isinstance(memories, list):
                logger.error(f"LLM returned invalid format: {type(memories)}")
                return []
            
            # Validate each memory
            valid_memories = []
            for mem in memories:
                if not isinstance(mem, dict):
                    logger.warning(f"Skipping non-dict memory: {mem}")
                    continue
                
                # Validate required fields
                if "content" not in mem or not mem["content"]:
                    logger.warning(f"Skipping memory without content: {mem}")
                    continue
                
                # Ensure tags is a list
                if "tags" not in mem:
                    mem["tags"] = ["behavior"]
                elif not isinstance(mem["tags"], list):
                    mem["tags"] = [str(mem["tags"])]
                
                # Ensure confidence is present
                if "confidence" not in mem:
                    mem["confidence"] = 0.5
                
                valid_memories.append(mem)
            
            logger.info(f"Validated {len(valid_memories)}/{len(memories)} extracted memories")
            return valid_memories
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            logger.debug(f"Failed to parse: {response[:500]}...")
            return []
        except Exception as e:
            logger.error(f"Unexpected error parsing memories: {e}", exc_info=True)
            return []
    
    async def _deduplicate(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicates using embedding similarity.
        
        Args:
            memories: List of new memories to check
            
        Returns:
            List of unique memories (with embeddings attached)
        """
        if not memories:
            return []
        
        logger.debug(f"Deduplicating {len(memories)} memories")
        
        # Get existing memories from Pinecone
        existing = await self._get_all_memories()
        logger.debug(f"Checking against {len(existing)} existing memories")
        
        unique = []
        
        for new_mem in memories:
            content = new_mem.get("content", "").strip()
            if not content:
                continue
            
            # Generate embedding for new memory
            new_emb = await self.embedder.get_embedding(content)
            if new_emb is None:
                logger.warning(f"Failed to generate embedding for: {content[:50]}...")
                continue
            
            # Check similarity against existing memories
            is_duplicate = False
            max_similarity = 0.0
            
            for exist_mem in existing:
                exist_emb = np.array(exist_mem["values"], dtype=np.float32)
                
                # Cosine similarity (vectors are already normalized)
                similarity = float(np.dot(new_emb, exist_emb))
                max_similarity = max(max_similarity, similarity)
                
                if similarity >= settings.DEDUP_THRESHOLD:
                    logger.debug(
                        f"Duplicate detected (sim={similarity:.3f}): "
                        f"{content[:50]}... == {exist_mem['metadata'].get('content', '')[:50]}..."
                    )
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                logger.debug(f"Unique memory (max_sim={max_similarity:.3f}): {content[:50]}...")
                new_mem["embedding"] = new_emb
                unique.append(new_mem)
        
        logger.info(f"Deduplication complete: {len(unique)}/{len(memories)} unique")
        return unique
    
    async def _store_memory(self, memory: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Store memory in Pinecone.
        
        Args:
            memory: Memory dict with content, tags, confidence, and embedding
            
        Returns:
            Stored memory dict with ID, or None on failure
        """
        content = memory.get("content", "").strip()
        embedding = memory.get("embedding")
        
        if not content:
            logger.error("Cannot store memory without content")
            return None
        
        # Generate embedding if not provided
        if embedding is None:
            embedding = await self.embedder.get_embedding(content)
        
        if embedding is None:
            logger.error(f"Failed to generate embedding for storage: {content[:50]}...")
            return None
        
        # Generate unique ID
        timestamp = int(time.time() * 1000)
        content_hash = abs(hash(content)) % 10000
        memory_id = f"mem_{timestamp}_{content_hash}"
        
        # Prepare metadata
        metadata = {
            "content": content,
            "tags": ",".join(memory.get("tags", [])),
            "confidence": float(memory.get("confidence", 0.5)),
            "timestamp": datetime.now().isoformat(),
            "created_at": datetime.now().isoformat()
        }
        
        try:
            # Upsert to Pinecone
            self.index.upsert(
                vectors=[(memory_id, embedding.tolist(), metadata)]
            )
            
            logger.info(f"Stored memory [{memory_id}]: {content[:80]}...")
            
            return {
                "id": memory_id,
                "content": content,
                "tags": memory.get("tags", []),
                "confidence": memory.get("confidence", 0.5),
                "timestamp": metadata["timestamp"]
            }
            
        except Exception as e:
            logger.error(f"Failed to store memory in Pinecone: {e}", exc_info=True)
            return None
    
    async def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search memories by semantic similarity.
        
        Args:
            query: Search query
            limit: Maximum results to return
            
        Returns:
            List of memories with relevance scores
        """
        logger.debug(f"Searching for: {query[:100]}...")
        
        # Generate query embedding
        query_emb = await self.embedder.get_embedding(query)
        
        if query_emb is None:
            logger.error("Failed to generate query embedding")
            return []
        
        try:
            # Query Pinecone
            results = self.index.query(
                vector=query_emb.tolist(),
                top_k=limit,
                include_metadata=True
            )
            
            memories = []
            for match in results.matches:
                memories.append({
                    "id": match.id,
                    "content": match.metadata.get("content", ""),
                    "relevance": float(match.score),
                    "tags": match.metadata.get("tags", "").split(",") if match.metadata.get("tags") else [],
                    "confidence": match.metadata.get("confidence", 0.5),
                    "timestamp": match.metadata.get("timestamp", "")
                })
            
            logger.info(f"Search returned {len(memories)} results")
            return memories
            
        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            return []
    
    async def get_relevant(self, current_message: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get memories relevant to current context with threshold filtering.
        
        Args:
            current_message: Current message to find relevant memories for
            limit: Maximum memories to return
            
        Returns:
            List of relevant memories above threshold
        """
        # Search with higher limit to allow filtering
        results = await self.search(current_message, limit=limit * 2)
        
        # Filter by relevance threshold
        filtered = [
            r for r in results 
            if r["relevance"] >= settings.RELEVANCE_THRESHOLD
        ]
        
        # Return top N after filtering
        relevant = filtered[:limit]
        
        logger.info(
            f"Relevant memories: {len(relevant)}/{len(results)} "
            f"(threshold: {settings.RELEVANCE_THRESHOLD})"
        )
        
        return relevant
    
    async def delete(self, memory_id: str) -> bool:
        """
        Delete a memory by ID.
        
        Args:
            memory_id: ID of memory to delete
            
        Returns:
            True if deleted, False otherwise
        """
        try:
            self.index.delete(ids=[memory_id])
            logger.info(f"Deleted memory: {memory_id}")
            return True
        except Exception as e:
            logger.error(f"Delete failed for {memory_id}: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get memory statistics.
        
        Returns:
            Dict with count and dimension info
        """
        try:
            stats = self.index.describe_index_stats()
            return {
                "count": stats.total_vector_count,
                "dimension": stats.dimension
            }
        except Exception as e:
            logger.error(f"Stats retrieval failed: {e}")
            return {"count": 0, "dimension": 0}
    
    async def _get_all_memories(self) -> List[Dict[str, Any]]:
        """
        Get all memories for deduplication.
        
        Returns:
            List of memory dicts with values and metadata
        """
        try:
            # Use dummy vector to retrieve all
            dummy = np.zeros(self.embedding_dim, dtype=np.float32)
            
            results = self.index.query(
                vector=dummy.tolist(),
                top_k=settings.MAX_MEMORIES,
                include_values=True,
                include_metadata=True
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
            logger.error(f"Failed to retrieve all memories: {e}")
            return []
    
    async def _prune_if_needed(self):
        """Prune oldest memories if over limit"""
        stats = await self.get_stats()
        count = stats["count"]
        
        if count <= settings.MAX_MEMORIES:
            return
        
        to_delete = count - settings.MAX_MEMORIES
        logger.info(f"Memory limit exceeded ({count} > {settings.MAX_MEMORIES}), pruning {to_delete} memories")
        
        # Get all memories sorted by timestamp
        all_mems = await self._get_all_memories()
        all_mems.sort(key=lambda x: x["metadata"].get("timestamp", ""))
        
        # Delete oldest
        for mem in all_mems[:to_delete]:
            await self.delete(mem["id"])
        
        logger.info(f"Pruned {to_delete} old memories")
