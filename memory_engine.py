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
            # API models
            "voyage-3.5": 1024,
            "voyage-3": 1024,
            "voyage-large-2": 1536,
            "voyage-code-2": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }

        if settings.EMBEDDING_PROVIDER == "local":
            self.embedding_dim = model_dims.get(settings.EMBEDDING_MODEL, 384)
        elif settings.EMBEDDING_PROVIDER == "pinecone":
            self.embedding_dim = model_dims.get(settings.EMBEDDING_MODEL, 384)
        else:
            # API provider - look up model or default to 1536
            self.embedding_dim = model_dims.get(settings.EMBEDDING_MODEL, 1536)

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
        """Extract memories using LLM with category support and temporal awareness"""

        from datetime import datetime
        current_date = datetime.now().strftime("%Y-%m-%d")

        system_prompt = f"""You are a memory extraction system. Extract ONLY user-specific facts, preferences, goals, relationships, and persistent information from natural conversation.

CURRENT DATE: {current_date}

Include temporal context when relevant (e.g., "As of {current_date}, user prefers X").

CRITICAL OUTPUT REQUIREMENTS:
1. Your ENTIRE response MUST be ONLY a valid JSON array: [...]
2. NO text before or after the JSON array
3. NO markdown code blocks (no ```json)
4. NO explanations or notes
5. Return [] if no user-specific memories found

Each memory object MUST have:
- "content": the extracted user-specific fact (string)
- "tags": array of relevant tags (e.g., ["homelab", "technical", "family"])
- "confidence": score from 0.0 to 1.0 indicating certainty
- "category": ONE category from the list below (REQUIRED)
- "importance": integer from 1-10 indicating significance (REQUIRED)
- "sentiment": ONE of: positive, negative, neutral, mixed (REQUIRED)

CATEGORIZATION (choose ONE):
- "achievement": Completed tasks, successes, victories, solved problems
- "frustration": Problems, failures, issues, roadblocks
- "idea": Thoughts, plans, possibilities, future considerations
- "fact": Factual information about user (name, location, possessions)
- "event": Things that happened, meetings, activities
- "conversation": Discussion topics, things mentioned
- "relationship": Information about people, connections, family
- "technical": Code, systems, infrastructure, tools, technology
- "personal": Family, hobbies, interests, daily life
- "misc": Everything else that doesn't fit above

IMPORTANCE SCORING (1-10):
- 1-3: Low - Minor details, trivial facts, passing mentions
- 4-6: Medium - Useful information, regular preferences, typical activities
- 7-8: High - Important facts, strong preferences, key relationships, significant events
- 9-10: Critical - Core identity, mission-critical information, deeply held values

SENTIMENT ANALYSIS:
- "positive": Achievements, good news, preferences, successes, joy
- "negative": Frustrations, problems, dislikes, failures, anger
- "neutral": Facts, observations, routine information
- "mixed": Complex feelings, both good and bad aspects

EXTRACT:
- Explicit user preferences ("I love X", "My favorite is Y")
- Identity details (name, location, profession, age)
- Goals and aspirations
- Relationships (family, friends, colleagues)
- Possessions (things owned or desired)
- Behavioral patterns and interests
- Achievements and victories
- Frustrations and problems

DO NOT EXTRACT:
- General knowledge or trivia
- Temporary thoughts or questions
- Information about the AI
- Meta-commentary about remembering

EXAMPLE OUTPUT:
[
  {{
    "content": "User fixed BADBUNNY PSU issue after 2 hours of troubleshooting",
    "tags": ["homelab", "technical", "hardware", "BADBUNNY"],
    "category": "achievement",
    "importance": 8,
    "sentiment": "positive",
    "confidence": 0.95
  }},
  {{
    "content": "User has a cat named Whiskers",
    "tags": ["relationship", "possession", "pets"],
    "category": "fact",
    "importance": 6,
    "sentiment": "neutral",
    "confidence": 0.9
  }},
  {{
    "content": "User frustrated with network connectivity issues",
    "tags": ["technical", "network", "problem"],
    "category": "frustration",
    "importance": 7,
    "sentiment": "negative",
    "confidence": 0.85
  }}
]

If no user-specific memories are found, return: []"""

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

                # V2.0: Ensure tags (array)
                if "tags" not in mem:
                    mem["tags"] = []
                elif not isinstance(mem["tags"], list):
                    mem["tags"] = [str(mem["tags"])]

                # V2.0: Ensure category (single value with default)
                if "category" not in mem or mem["category"] not in [
                    "achievement", "frustration", "idea", "fact", "event",
                    "conversation", "relationship", "technical", "personal", "misc"
                ]:
                    mem["category"] = "misc"

                # V2.0: Ensure importance (1-10 with default 5)
                if "importance" not in mem:
                    mem["importance"] = 5
                else:
                    try:
                        mem["importance"] = int(mem["importance"])
                        if not 1 <= mem["importance"] <= 10:
                            mem["importance"] = 5
                    except (ValueError, TypeError):
                        mem["importance"] = 5

                # V2.0: Ensure sentiment (with default neutral)
                if "sentiment" not in mem or mem["sentiment"] not in ["positive", "negative", "neutral", "mixed"]:
                    mem["sentiment"] = "neutral"

                # Legacy: Ensure categories (keep for backward compatibility)
                if "categories" not in mem:
                    mem["categories"] = mem["tags"] if mem.get("tags") else []
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
        """Smart deduplication with memory updating"""
        if not memories:
            return []

        logger.debug(f"🔄 Processing {len(memories)} memories for deduplication/update")

        # Get existing memories for this context
        existing = await self._get_all_memories(
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id
        )

        logger.debug(f"📊 Checking against {len(existing)} existing memories")

        unique = []
        update_threshold = 0.85  # Lower than dedup threshold - allows for updates

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
            should_update = False
            similar_memory = None
            max_similarity = 0.0

            for exist_mem in existing:
                exist_emb = np.array(exist_mem["values"], dtype=np.float32)
                similarity = float(np.dot(new_emb, exist_emb))

                if similarity > max_similarity:
                    max_similarity = similarity
                    similar_memory = exist_mem

                # Exact duplicate - skip completely
                if similarity >= settings.DEDUP_THRESHOLD:
                    logger.debug(
                        f"🔄 Duplicate detected (sim={similarity:.3f}): "
                        f"{content[:50]}... == {exist_mem['metadata'].get('content', '')[:50]}..."
                    )
                    is_duplicate = True
                    break

                # Similar but different - might be an update
                elif similarity >= update_threshold:
                    logger.info(
                        f"♻️ Similar memory found (sim={similarity:.3f}), treating as update: "
                        f"NEW: {content[:50]}... | OLD: {exist_mem['metadata'].get('content', '')[:50]}..."
                    )
                    should_update = True
                    similar_memory = exist_mem
                    break

            if is_duplicate:
                continue  # Skip exact duplicates
            elif should_update and similar_memory:
                # Delete old version and store new one
                logger.info(f"♻️ Updating memory {similar_memory['id']}")
                await self.delete(similar_memory['id'])
                new_mem["embedding"] = new_emb
                new_mem["updated_from"] = similar_memory['id']
                unique.append(new_mem)
            else:
                # Completely new memory
                logger.debug(f"✨ New memory (max_sim={max_similarity:.3f}): {content[:50]}...")
                new_mem["embedding"] = new_emb
                unique.append(new_mem)

        logger.info(f"✅ Deduplication complete: {len(unique)}/{len(memories)} to store (includes updates)")
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
        
        # Build metadata (v2.0 schema - MCP parity)
        mem_metadata = {
            # Core fields
            "content": content,
            "confidence": float(memory.get("confidence", 0.6)),
            "timestamp": datetime.now().isoformat(),  # Legacy field (keep for compatibility)
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "version": 1,

            # Legacy categories field (keep for backward compatibility)
            "categories": ",".join(memory.get("categories", [])) if memory.get("categories") else "",

            # V2.0 fields - MCP parity
            "tags": memory.get("tags", []),  # Array (replaces CSV categories)
            "category": memory.get("category", "misc"),  # Single category
            "importance": memory.get("importance", 5),  # 1-10 scale
            "pinned": memory.get("pinned", False),
            "archived": memory.get("archived", False),
            "sentiment": memory.get("sentiment", "neutral"),  # positive/negative/neutral/mixed
            "word_count": len(content.split()),
            "event_date": memory.get("event_date")  # Optional: when event occurred
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
        page_size: Optional[int] = None,
        use_hybrid: bool = True
    ) -> Dict[str, Any]:
        """Search memories with hybrid semantic + keyword matching"""
        logger.debug(f"🔍 Searching: {query[:80]}... (hybrid={use_hybrid})")

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
            # Get more results for hybrid re-ranking
            top_k = (limit * 3) if use_hybrid else (limit * 2)
            top_k = min(top_k, 100)

            # Query Pinecone
            results = self.index.query(
                vector=query_emb.tolist(),
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict if filter_dict else None
            )

            memories = []
            query_terms = query.lower().split()

            for match in results.matches:
                # Filter by categories if specified
                cats_raw = match.metadata.get("categories", "")
                mem_categories = cats_raw if isinstance(cats_raw, list) else (cats_raw.split(",") if cats_raw else [])
                if categories:
                    if not any(cat in mem_categories for cat in categories):
                        continue

                content = match.metadata.get("content", "")
                semantic_score = float(match.score)

                # Hybrid scoring: boost results with keyword matches
                if use_hybrid:
                    content_lower = content.lower()
                    keyword_matches = sum(1 for term in query_terms if term in content_lower)
                    keyword_boost = 1.0 + (0.15 * keyword_matches)  # 15% boost per keyword match
                    final_score = semantic_score * keyword_boost
                else:
                    final_score = semantic_score

                memories.append({
                    "id": match.id,
                    "content": content,
                    "relevance": final_score,
                    "semantic_score": semantic_score,
                    "categories": mem_categories,
                    "confidence": match.metadata.get("confidence", 0.6),
                    "created_at": match.metadata.get("created_at", ""),
                    "user_id": match.metadata.get("user_id"),
                    "agent_id": match.metadata.get("agent_id"),
                    "run_id": match.metadata.get("run_id")
                })

            # Re-sort by final score if using hybrid
            if use_hybrid:
                memories.sort(key=lambda x: x["relevance"], reverse=True)

            # Apply pagination
            page_size = page_size or settings.DEFAULT_PAGE_SIZE
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size

            paginated = memories[start_idx:end_idx]

            logger.info(f"✅ Search returned {len(paginated)} results (hybrid={use_hybrid})")

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

    async def get_recent(self, limit: int = 10, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get most recent memories sorted by timestamp"""
        try:
            all_mems = await self._get_all_memories(user_id=user_id)
            # Sort by timestamp descending (newest first)
            all_mems.sort(key=lambda x: x["metadata"].get("created_at", ""), reverse=True)

            recent = []
            for mem in all_mems[:limit]:
                cats_raw = mem["metadata"].get("categories", "")
                categories = cats_raw if isinstance(cats_raw, list) else (cats_raw.split(",") if cats_raw else [])
                recent.append({
                    "id": mem["id"],
                    "content": mem["metadata"].get("content", ""),
                    "categories": categories,
                    "confidence": mem["metadata"].get("confidence", 0.5),
                    "created_at": mem["metadata"].get("created_at", ""),
                    "user_id": mem["metadata"].get("user_id")
                })

            logger.info(f"📋 Retrieved {len(recent)} recent memories")
            return recent
        except Exception as e:
            logger.error(f"💥 Failed to get recent memories: {e}", exc_info=True)
            return []

    async def consolidate_memories(
        self,
        user_id: Optional[str] = None,
        tag: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Consolidate related memories into coherent summaries using LLM.
        Useful for merging fragmented information into unified facts.
        """
        logger.info(f"🔄 Starting memory consolidation for user_id={user_id}, tag={tag}")

        # Get all memories or filter by tag
        all_mems = await self._get_all_memories(user_id=user_id)

        if tag:
            all_mems = [
                m for m in all_mems
                if tag in (m["metadata"].get("categories", []) if isinstance(m["metadata"].get("categories"), list) else m["metadata"].get("categories", "").split(","))
            ]

        if len(all_mems) < 3:
            logger.info("⚠️ Not enough memories to consolidate (need at least 3)")
            return {"consolidated": 0, "message": "Not enough memories to consolidate"}

        # Group memories by semantic similarity
        groups = await self._group_similar_memories(all_mems)

        consolidated_count = 0
        for group in groups:
            if len(group) < 2:
                continue  # Skip single memories

            # Use LLM to consolidate group
            consolidated = await self._consolidate_group(group, user_id=user_id)
            if consolidated:
                consolidated_count += 1

        logger.info(f"✅ Consolidated {consolidated_count} memory groups")
        return {
            "consolidated": consolidated_count,
            "message": f"Successfully consolidated {consolidated_count} memory groups"
        }

    async def _group_similar_memories(
        self,
        memories: List[Dict[str, Any]],
        similarity_threshold: float = 0.75
    ) -> List[List[Dict[str, Any]]]:
        """Group memories by semantic similarity"""
        if not memories:
            return []

        groups = []
        used = set()

        for i, mem in enumerate(memories):
            if i in used:
                continue

            group = [mem]
            mem_emb = np.array(mem["values"], dtype=np.float32)

            # Find similar memories
            for j, other_mem in enumerate(memories):
                if j <= i or j in used:
                    continue

                other_emb = np.array(other_mem["values"], dtype=np.float32)
                similarity = float(np.dot(mem_emb, other_emb))

                if similarity >= similarity_threshold:
                    group.append(other_mem)
                    used.add(j)

            if len(group) >= 2:  # Only keep groups with multiple memories
                groups.append(group)
                used.add(i)

        logger.debug(f"📊 Grouped {len(memories)} memories into {len(groups)} groups")
        return groups

    async def _consolidate_group(
        self,
        group: List[Dict[str, Any]],
        user_id: Optional[str] = None
    ) -> Optional[str]:
        """Use LLM to consolidate a group of related memories"""
        memory_texts = [m["metadata"].get("content", "") for m in group]

        system_prompt = """You are a memory consolidation system. Given multiple related memory fragments, create a single coherent, comprehensive memory that captures all the information.

RULES:
1. Combine all relevant information into one clear statement
2. Resolve any contradictions (prefer more recent information)
3. Keep the consolidated memory concise but complete
4. Maintain factual accuracy
5. Return ONLY the consolidated memory text, nothing else"""

        user_prompt = f"""Consolidate these related memories into one comprehensive memory:

{chr(10).join(f"{i+1}. {text}" for i, text in enumerate(memory_texts))}

Return only the consolidated memory:"""

        try:
            consolidated_text = await self.llm.query(system_prompt, user_prompt, temperature=0.2)

            if not consolidated_text:
                logger.warning("⚠️ LLM returned no consolidation")
                return None

            # Delete old memories
            for mem in group:
                await self.delete(mem["id"])

            # Store consolidated memory
            consolidated_mem = {
                "content": consolidated_text.strip(),
                "categories": list(set(
                    cat for mem in group
                    for cat in (mem["metadata"].get("categories", []) if isinstance(mem["metadata"].get("categories"), list) else mem["metadata"].get("categories", "").split(","))
                    if cat
                )),
                "confidence": max(
                    mem["metadata"].get("confidence", 0.5)
                    for mem in group
                )
            }

            result = await self._store_memory(
                consolidated_mem,
                user_id=user_id
            )

            if result:
                logger.info(f"✅ Consolidated {len(group)} memories into: {consolidated_text[:80]}...")
                return result

            return None

        except Exception as e:
            logger.error(f"💥 Failed to consolidate group: {e}", exc_info=True)
            return None
