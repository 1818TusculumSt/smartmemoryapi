# SmartMemory v2.0 - Complete Enhancement Specification

**Author:** Chuckles (with El Jefe)  
**Date:** October 29, 2025  
**Purpose:** Transform SmartMemory from vector store into enterprise-grade memory infrastructure

---

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Enhanced Data Model](#enhanced-data-model)
3. [New API Endpoints](#new-api-endpoints)
4. [Implementation Details](#implementation-details)
5. [Migration Strategy](#migration-strategy)
6. [Testing Plan](#testing-plan)
7. [MCP Tool Definitions](#mcp-tool-definitions)

---

## Architecture Overview

### Current State
- Simple vector store with semantic search
- Basic CRUD operations
- Minimal metadata (content, embedding, user_id)
- No temporal queries
- No categorization system
- No analytics

### Target State
- Full-featured memory system
- Multi-dimensional retrieval (semantic + temporal + categorical)
- Rich metadata with relationships
- Analytics and insights
- Quality curation tools
- Export/backup capabilities

### Key Principles
1. **Backward Compatibility:** All existing endpoints continue to work
2. **Optional Metadata:** New fields are optional, don't break existing flows
3. **Progressive Enhancement:** Can adopt features incrementally
4. **Performance First:** Indexes on common query patterns
5. **User Control:** Manual overrides for AI decisions

---

## Enhanced Data Model

### Memory Schema v2.0

```python
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class MemoryCategory(str, Enum):
    ACHIEVEMENT = "achievement"
    FRUSTRATION = "frustration"
    IDEA = "idea"
    FACT = "fact"
    EVENT = "event"
    CONVERSATION = "conversation"
    RELATIONSHIP = "relationship"
    TECHNICAL = "technical"
    PERSONAL = "personal"
    MISC = "misc"

class MemorySource(str, Enum):
    CHAT = "chat"
    MANUAL = "manual"
    IMPORT = "import"
    EXTRACTION = "extraction"
    API = "api"

class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"

class AttachmentType(str, Enum):
    IMAGE = "image"
    LINK = "link"
    FILE = "file"
    CODE = "code"

class Attachment(BaseModel):
    type: AttachmentType
    url: str
    description: Optional[str] = None
    mime_type: Optional[str] = None

class Location(BaseModel):
    lat: Optional[float] = None
    lon: Optional[float] = None
    name: Optional[str] = None

class EditHistory(BaseModel):
    edited_at: datetime
    previous_content: str
    edit_reason: Optional[str] = None
    edited_by: Optional[str] = None

class RelationshipType(str, Enum):
    CAUSED_BY = "caused_by"
    RELATED_TO = "related_to"
    CONTRADICTS = "contradicts"
    FOLLOWS = "follows"
    REFERENCES = "references"
    RESOLVES = "resolves"

class MemoryRelationship(BaseModel):
    target_memory_id: str
    relationship_type: RelationshipType
    created_at: datetime

class MemoryMetadata(BaseModel):
    """Extended metadata for memory v2.0"""
    
    # Categorization
    tags: List[str] = Field(default_factory=list)
    category: Optional[MemoryCategory] = None
    source: MemorySource = MemorySource.CHAT
    
    # Context
    conversation_id: Optional[str] = None
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    relationships: List[MemoryRelationship] = Field(default_factory=list)
    
    # Quality/Importance
    importance: Optional[int] = Field(None, ge=1, le=10)
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    verified: bool = False
    pinned: bool = False
    archived: bool = False
    
    # Temporal
    event_date: Optional[datetime] = None  # When event actually happened
    expires_at: Optional[datetime] = None
    reminder_at: Optional[datetime] = None
    
    # Content Analysis
    word_count: Optional[int] = None
    entities: List[str] = Field(default_factory=list)
    sentiment: Optional[Sentiment] = None
    language: str = "en"
    
    # Usage Tracking
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    referenced_in: List[str] = Field(default_factory=list)  # conversation_ids
    
    # Rich Content
    attachments: List[Attachment] = Field(default_factory=list)
    location: Optional[Location] = None
    
    # Version Control
    version: int = 1
    edit_history: List[EditHistory] = Field(default_factory=list)

class Memory(BaseModel):
    """Complete memory object v2.0"""
    id: str
    content: str
    embedding: List[float]
    user_id: str
    
    # Timestamps
    created_at: datetime
    updated_at: datetime
    
    # Extended metadata
    metadata: MemoryMetadata = Field(default_factory=MemoryMetadata)
    
    # Computed fields (not stored, calculated on retrieval)
    relevance_score: Optional[float] = None
    time_score: Optional[float] = None
    combined_score: Optional[float] = None
```

### Database Schema Changes

```sql
-- Add new columns to memories table
ALTER TABLE memories ADD COLUMN tags TEXT[];
ALTER TABLE memories ADD COLUMN category VARCHAR(50);
ALTER TABLE memories ADD COLUMN source VARCHAR(50) DEFAULT 'chat';
ALTER TABLE memories ADD COLUMN importance INTEGER;
ALTER TABLE memories ADD COLUMN confidence FLOAT;
ALTER TABLE memories ADD COLUMN verified BOOLEAN DEFAULT FALSE;
ALTER TABLE memories ADD COLUMN pinned BOOLEAN DEFAULT FALSE;
ALTER TABLE memories ADD COLUMN archived BOOLEAN DEFAULT FALSE;
ALTER TABLE memories ADD COLUMN event_date TIMESTAMP;
ALTER TABLE memories ADD COLUMN expires_at TIMESTAMP;
ALTER TABLE memories ADD COLUMN reminder_at TIMESTAMP;
ALTER TABLE memories ADD COLUMN word_count INTEGER;
ALTER TABLE memories ADD COLUMN entities TEXT[];
ALTER TABLE memories ADD COLUMN sentiment VARCHAR(20);
ALTER TABLE memories ADD COLUMN language VARCHAR(10) DEFAULT 'en';
ALTER TABLE memories ADD COLUMN access_count INTEGER DEFAULT 0;
ALTER TABLE memories ADD COLUMN last_accessed TIMESTAMP;
ALTER TABLE memories ADD COLUMN conversation_id VARCHAR(255);
ALTER TABLE memories ADD COLUMN agent_id VARCHAR(255);
ALTER TABLE memories ADD COLUMN session_id VARCHAR(255);
ALTER TABLE memories ADD COLUMN attachments JSONB;
ALTER TABLE memories ADD COLUMN location JSONB;
ALTER TABLE memories ADD COLUMN version INTEGER DEFAULT 1;
ALTER TABLE memories ADD COLUMN edit_history JSONB;

-- Create indexes for common queries
CREATE INDEX idx_memories_created_at ON memories(created_at DESC);
CREATE INDEX idx_memories_tags ON memories USING GIN(tags);
CREATE INDEX idx_memories_category ON memories(category);
CREATE INDEX idx_memories_importance ON memories(importance DESC);
CREATE INDEX idx_memories_pinned ON memories(pinned) WHERE pinned = TRUE;
CREATE INDEX idx_memories_archived ON memories(archived) WHERE archived = FALSE;
CREATE INDEX idx_memories_event_date ON memories(event_date);
CREATE INDEX idx_memories_user_created ON memories(user_id, created_at DESC);
CREATE INDEX idx_memories_conversation ON memories(conversation_id);
CREATE INDEX idx_memories_session ON memories(session_id);

-- Create relationships table
CREATE TABLE memory_relationships (
    id SERIAL PRIMARY KEY,
    source_memory_id VARCHAR(255) NOT NULL,
    target_memory_id VARCHAR(255) NOT NULL,
    relationship_type VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (source_memory_id) REFERENCES memories(id) ON DELETE CASCADE,
    FOREIGN KEY (target_memory_id) REFERENCES memories(id) ON DELETE CASCADE,
    UNIQUE(source_memory_id, target_memory_id, relationship_type)
);

CREATE INDEX idx_relationships_source ON memory_relationships(source_memory_id);
CREATE INDEX idx_relationships_target ON memory_relationships(target_memory_id);
CREATE INDEX idx_relationships_type ON memory_relationships(relationship_type);
```

---

## New API Endpoints

### 1. Time-Based Retrieval

#### GET /memories/recent
Get memories sorted by creation time (newest first).

**Query Parameters:**
- `user_id` (required): User identifier
- `limit` (optional, default=20): Number of memories to return
- `since` (optional): ISO timestamp, only return memories after this time
- `before` (optional): ISO timestamp, only return memories before this time
- `include_archived` (optional, default=false): Include archived memories

**Response:**
```json
{
  "memories": [
    {
      "id": "mem-123",
      "content": "Fixed BADBUNNY node PSU issue",
      "created_at": "2025-10-29T14:30:00Z",
      "metadata": {
        "tags": ["homelab", "technical"],
        "importance": 8
      }
    }
  ],
  "count": 15,
  "has_more": true,
  "next_cursor": "2025-10-28T10:00:00Z"
}
```

**Implementation:**
```python
@app.get("/memories/recent")
async def get_recent_memories(
    user_id: str,
    limit: int = 20,
    since: Optional[datetime] = None,
    before: Optional[datetime] = None,
    include_archived: bool = False
):
    query = "SELECT * FROM memories WHERE user_id = $1"
    params = [user_id]
    
    if not include_archived:
        query += " AND archived = FALSE"
    
    if since:
        query += f" AND created_at > ${len(params) + 1}"
        params.append(since)
    
    if before:
        query += f" AND created_at < ${len(params) + 1}"
        params.append(before)
    
    query += f" ORDER BY created_at DESC LIMIT ${len(params) + 1}"
    params.append(limit)
    
    memories = await db.fetch(query, *params)
    
    return {
        "memories": memories,
        "count": len(memories),
        "has_more": len(memories) == limit,
        "next_cursor": memories[-1]["created_at"] if memories else None
    }
```

---

#### GET /memories/timeline
Get memory counts/summaries grouped by time period.

**Query Parameters:**
- `user_id` (required): User identifier
- `group_by` (required): day|week|month|year
- `start_date` (optional): Start of time range
- `end_date` (optional): End of time range

**Response:**
```json
{
  "timeline": [
    {
      "period": "2025-10-29",
      "count": 12,
      "top_tags": ["homelab", "technical"],
      "sentiment_distribution": {
        "positive": 8,
        "negative": 2,
        "neutral": 2
      }
    }
  ],
  "total_count": 45
}
```

---

### 2. Tag Management

#### GET /memories/tags
List all unique tags with usage counts.

**Query Parameters:**
- `user_id` (required): User identifier
- `min_count` (optional): Only return tags used at least N times
- `sort_by` (optional): count|name (default: count)

**Response:**
```json
{
  "tags": [
    {"name": "homelab", "count": 45, "last_used": "2025-10-29T14:30:00Z"},
    {"name": "technical", "count": 32, "last_used": "2025-10-29T12:00:00Z"}
  ],
  "total_tags": 23
}
```

**Implementation:**
```python
@app.get("/memories/tags")
async def get_tags(
    user_id: str,
    min_count: int = 1,
    sort_by: str = "count"
):
    query = """
        SELECT 
            tag,
            COUNT(*) as count,
            MAX(created_at) as last_used
        FROM memories, UNNEST(tags) as tag
        WHERE user_id = $1 AND archived = FALSE
        GROUP BY tag
        HAVING COUNT(*) >= $2
    """
    
    if sort_by == "count":
        query += " ORDER BY count DESC, tag ASC"
    else:
        query += " ORDER BY tag ASC"
    
    tags = await db.fetch(query, user_id, min_count)
    
    return {
        "tags": [
            {"name": t["tag"], "count": t["count"], "last_used": t["last_used"]}
            for t in tags
        ],
        "total_tags": len(tags)
    }
```

---

#### GET /memories/by-tag/{tag}
Get all memories with a specific tag.

**Query Parameters:**
- `user_id` (required): User identifier
- `limit` (optional, default=20): Number of memories
- `sort_by` (optional): created_at|importance|relevance

**Response:**
```json
{
  "tag": "homelab",
  "memories": [...],
  "count": 45,
  "has_more": true
}
```

---

#### POST /memories/{memory_id}/tags
Add tags to an existing memory.

**Request Body:**
```json
{
  "tags": ["homelab", "victory"],
  "replace": false
}
```

**Response:**
```json
{
  "memory_id": "mem-123",
  "tags": ["homelab", "victory", "technical"],
  "added": ["victory"],
  "existing": ["homelab", "technical"]
}
```

**Implementation:**
```python
@app.post("/memories/{memory_id}/tags")
async def add_tags(
    memory_id: str,
    tags: List[str],
    replace: bool = False
):
    memory = await db.fetchrow(
        "SELECT tags FROM memories WHERE id = $1",
        memory_id
    )
    
    if not memory:
        raise HTTPException(404, "Memory not found")
    
    if replace:
        new_tags = tags
        added = tags
        existing = []
    else:
        existing_tags = set(memory["tags"] or [])
        new_tags_set = set(tags)
        added = list(new_tags_set - existing_tags)
        existing = list(new_tags_set & existing_tags)
        new_tags = list(existing_tags | new_tags_set)
    
    await db.execute(
        "UPDATE memories SET tags = $1, updated_at = NOW() WHERE id = $2",
        new_tags,
        memory_id
    )
    
    return {
        "memory_id": memory_id,
        "tags": new_tags,
        "added": added,
        "existing": existing
    }
```

---

#### DELETE /memories/{memory_id}/tags/{tag}
Remove a specific tag from a memory.

---

### 3. Relationship Queries

#### GET /memories/related/{memory_id}
Find memories semantically similar to a specific memory.

**Query Parameters:**
- `limit` (optional, default=10): Number of similar memories
- `min_similarity` (optional, default=0.7): Minimum similarity threshold

**Response:**
```json
{
  "source_memory_id": "mem-123",
  "related_memories": [
    {
      "id": "mem-456",
      "content": "...",
      "similarity_score": 0.89,
      "relationship_type": "related_to"
    }
  ],
  "count": 5
}
```

**Implementation:**
```python
@app.get("/memories/related/{memory_id}")
async def get_related_memories(
    memory_id: str,
    limit: int = 10,
    min_similarity: float = 0.7
):
    # Get source memory embedding
    source = await db.fetchrow(
        "SELECT embedding, user_id FROM memories WHERE id = $1",
        memory_id
    )
    
    if not source:
        raise HTTPException(404, "Memory not found")
    
    # Find similar memories using vector similarity
    similar = await db.fetch("""
        SELECT 
            id,
            content,
            created_at,
            metadata,
            1 - (embedding <=> $1::vector) as similarity
        FROM memories
        WHERE user_id = $2 
        AND id != $3
        AND archived = FALSE
        AND 1 - (embedding <=> $1::vector) >= $4
        ORDER BY similarity DESC
        LIMIT $5
    """, source["embedding"], source["user_id"], memory_id, min_similarity, limit)
    
    # Also get explicit relationships
    explicit = await db.fetch("""
        SELECT 
            m.*,
            r.relationship_type
        FROM memory_relationships r
        JOIN memories m ON m.id = r.target_memory_id
        WHERE r.source_memory_id = $1
    """, memory_id)
    
    return {
        "source_memory_id": memory_id,
        "related_memories": similar,
        "explicit_relationships": explicit,
        "count": len(similar) + len(explicit)
    }
```

---

#### POST /memories/connect
Explicitly link memories with relationship types.

**Request Body:**
```json
{
  "source_memory_id": "mem-123",
  "target_memory_id": "mem-456",
  "relationship_type": "caused_by"
}
```

**Response:**
```json
{
  "relationship_id": "rel-789",
  "created_at": "2025-10-29T14:30:00Z"
}
```

---

### 4. Analytics & Insights

#### GET /memories/stats
Get comprehensive statistics about memories.

**Query Parameters:**
- `user_id` (required): User identifier
- `since` (optional): Only include memories after this date
- `before` (optional): Only include memories before this date

**Response:**
```json
{
  "total_memories": 338,
  "total_tags": 45,
  "total_words": 15420,
  "date_range": {
    "first_memory": "2025-01-15T10:00:00Z",
    "last_memory": "2025-10-29T14:30:00Z",
    "days_active": 287
  },
  "top_tags": [
    {"tag": "homelab", "count": 67},
    {"tag": "technical", "count": 45}
  ],
  "category_distribution": {
    "technical": 120,
    "achievement": 45,
    "frustration": 23
  },
  "sentiment_distribution": {
    "positive": 180,
    "neutral": 120,
    "negative": 38
  },
  "activity_by_day": {
    "Monday": 45,
    "Tuesday": 52,
    "Wednesday": 48
  },
  "memory_rate": {
    "per_day": 1.18,
    "per_week": 8.26,
    "per_month": 35.7
  },
  "importance_distribution": {
    "high": 45,
    "medium": 120,
    "low": 173
  },
  "pinned_count": 12,
  "archived_count": 8
}
```

**Implementation:**
```python
@app.get("/memories/stats")
async def get_memory_stats(
    user_id: str,
    since: Optional[datetime] = None,
    before: Optional[datetime] = None
):
    # Build base query with filters
    where_clause = "WHERE user_id = $1"
    params = [user_id]
    
    if since:
        where_clause += f" AND created_at >= ${len(params) + 1}"
        params.append(since)
    
    if before:
        where_clause += f" AND created_at <= ${len(params) + 1}"
        params.append(before)
    
    # Get basic counts
    basic_stats = await db.fetchrow(f"""
        SELECT 
            COUNT(*) as total_memories,
            COUNT(DISTINCT tag) as total_tags,
            SUM(word_count) as total_words,
            MIN(created_at) as first_memory,
            MAX(created_at) as last_memory,
            COUNT(*) FILTER (WHERE pinned = TRUE) as pinned_count,
            COUNT(*) FILTER (WHERE archived = TRUE) as archived_count
        FROM memories
        LEFT JOIN UNNEST(tags) as tag ON TRUE
        {where_clause}
    """, *params)
    
    # Get top tags
    top_tags = await db.fetch(f"""
        SELECT tag, COUNT(*) as count
        FROM memories, UNNEST(tags) as tag
        {where_clause}
        GROUP BY tag
        ORDER BY count DESC
        LIMIT 10
    """, *params)
    
    # Get category distribution
    categories = await db.fetch(f"""
        SELECT category, COUNT(*) as count
        FROM memories
        {where_clause} AND category IS NOT NULL
        GROUP BY category
        ORDER BY count DESC
    """, *params)
    
    # Get sentiment distribution
    sentiments = await db.fetch(f"""
        SELECT sentiment, COUNT(*) as count
        FROM memories
        {where_clause} AND sentiment IS NOT NULL
        GROUP BY sentiment
    """, *params)
    
    # Calculate activity patterns
    activity_by_day = await db.fetch(f"""
        SELECT 
            TO_CHAR(created_at, 'Day') as day_name,
            COUNT(*) as count
        FROM memories
        {where_clause}
        GROUP BY day_name
        ORDER BY 
            CASE TO_CHAR(created_at, 'Day')
                WHEN 'Monday   ' THEN 1
                WHEN 'Tuesday  ' THEN 2
                WHEN 'Wednesday' THEN 3
                WHEN 'Thursday ' THEN 4
                WHEN 'Friday   ' THEN 5
                WHEN 'Saturday ' THEN 6
                WHEN 'Sunday   ' THEN 7
            END
    """, *params)
    
    # Calculate rates
    days_active = (basic_stats["last_memory"] - basic_stats["first_memory"]).days + 1
    
    return {
        "total_memories": basic_stats["total_memories"],
        "total_tags": basic_stats["total_tags"],
        "total_words": basic_stats["total_words"] or 0,
        "date_range": {
            "first_memory": basic_stats["first_memory"],
            "last_memory": basic_stats["last_memory"],
            "days_active": days_active
        },
        "top_tags": [{"tag": r["tag"], "count": r["count"]} for r in top_tags],
        "category_distribution": {r["category"]: r["count"] for r in categories},
        "sentiment_distribution": {r["sentiment"]: r["count"] for r in sentiments},
        "activity_by_day": {r["day_name"].strip(): r["count"] for r in activity_by_day},
        "memory_rate": {
            "per_day": round(basic_stats["total_memories"] / days_active, 2),
            "per_week": round(basic_stats["total_memories"] / (days_active / 7), 2),
            "per_month": round(basic_stats["total_memories"] / (days_active / 30), 2)
        },
        "importance_distribution": {
            "high": await db.fetchval(f"SELECT COUNT(*) FROM memories {where_clause} AND importance >= 8", *params),
            "medium": await db.fetchval(f"SELECT COUNT(*) FROM memories {where_clause} AND importance BETWEEN 4 AND 7", *params),
            "low": await db.fetchval(f"SELECT COUNT(*) FROM memories {where_clause} AND importance <= 3", *params)
        },
        "pinned_count": basic_stats["pinned_count"],
        "archived_count": basic_stats["archived_count"]
    }
```

---

#### GET /memories/duplicates
Find potential duplicate memories for cleanup.

**Query Parameters:**
- `user_id` (required): User identifier
- `threshold` (optional, default=0.95): Similarity threshold for duplicates
- `limit` (optional, default=50): Max number of duplicate pairs to return

**Response:**
```json
{
  "duplicate_pairs": [
    {
      "memory1": {
        "id": "mem-123",
        "content": "Fixed BADBUNNY PSU",
        "created_at": "2025-10-29T14:00:00Z"
      },
      "memory2": {
        "id": "mem-124",
        "content": "Replaced PSU on BADBUNNY",
        "created_at": "2025-10-29T14:30:00Z"
      },
      "similarity": 0.97,
      "recommended_action": "keep_newer"
    }
  ],
  "count": 5,
  "total_potential_savings": 5
}
```

---

#### GET /memories/gaps
Identify time periods with no memory activity.

**Query Parameters:**
- `user_id` (required): User identifier
- `min_gap_hours` (optional, default=24): Minimum gap size to report

**Response:**
```json
{
  "gaps": [
    {
      "start": "2025-10-15T22:00:00Z",
      "end": "2025-10-18T08:00:00Z",
      "duration_hours": 58,
      "duration_days": 2.4
    }
  ],
  "total_gaps": 3,
  "longest_gap_hours": 72
}
```

---

### 5. Advanced Search

#### POST /memories/search/advanced
Full-featured search with multiple filter dimensions.

**Request Body:**
```json
{
  "query": "homelab",
  "user_id": "el-jefe-principal",
  "filters": {
    "tags": ["technical", "victory"],
    "categories": ["achievement", "technical"],
    "date_range": {
      "start": "2025-10-01T00:00:00Z",
      "end": "2025-10-31T23:59:59Z"
    },
    "importance_min": 5,
    "importance_max": 10,
    "verified_only": false,
    "include_archived": false,
    "exclude_tags": ["draft"],
    "entities": ["BADBUNNY", "Proxmox"],
    "sentiment": ["positive"],
    "min_relevance": 0.7
  },
  "sort_by": "relevance",
  "sort_order": "desc",
  "limit": 20,
  "offset": 0
}
```

**Response:**
```json
{
  "memories": [...],
  "count": 15,
  "total_matches": 67,
  "filters_applied": {
    "tags": ["technical", "victory"],
    "date_range": "2025-10-01 to 2025-10-31"
  },
  "has_more": true
}
```

**Implementation:**
```python
@app.post("/memories/search/advanced")
async def advanced_search(request: AdvancedSearchRequest):
    # Start with base query
    query = """
        SELECT 
            m.*,
            1 - (m.embedding <=> $1::vector) as relevance_score
        FROM memories m
        WHERE m.user_id = $2
    """
    params = [await get_embedding(request.query), request.user_id]
    param_idx = 3
    
    # Apply filters
    if request.filters:
        # Tags filter (AND logic)
        if request.filters.tags:
            query += f" AND m.tags @> ${param_idx}::text[]"
            params.append(request.filters.tags)
            param_idx += 1
        
        # Exclude tags
        if request.filters.exclude_tags:
            query += f" AND NOT m.tags && ${param_idx}::text[]"
            params.append(request.filters.exclude_tags)
            param_idx += 1
        
        # Categories filter (OR logic)
        if request.filters.categories:
            placeholders = ",".join([f"${param_idx + i}" for i in range(len(request.filters.categories))])
            query += f" AND m.category IN ({placeholders})"
            params.extend(request.filters.categories)
            param_idx += len(request.filters.categories)
        
        # Date range
        if request.filters.date_range:
            if request.filters.date_range.start:
                query += f" AND m.created_at >= ${param_idx}"
                params.append(request.filters.date_range.start)
                param_idx += 1
            if request.filters.date_range.end:
                query += f" AND m.created_at <= ${param_idx}"
                params.append(request.filters.date_range.end)
                param_idx += 1
        
        # Importance range
        if request.filters.importance_min:
            query += f" AND m.importance >= ${param_idx}"
            params.append(request.filters.importance_min)
            param_idx += 1
        if request.filters.importance_max:
            query += f" AND m.importance <= ${param_idx}"
            params.append(request.filters.importance_max)
            param_idx += 1
        
        # Verified only
        if request.filters.verified_only:
            query += " AND m.verified = TRUE"
        
        # Include/exclude archived
        if not request.filters.include_archived:
            query += " AND m.archived = FALSE"
        
        # Entities filter (contains any)
        if request.filters.entities:
            query += f" AND m.entities && ${param_idx}::text[]"
            params.append(request.filters.entities)
            param_idx += 1
        
        # Sentiment filter
        if request.filters.sentiment:
            placeholders = ",".join([f"${param_idx + i}" for i in range(len(request.filters.sentiment))])
            query += f" AND m.sentiment IN ({placeholders})"
            params.extend(request.filters.sentiment)
            param_idx += len(request.filters.sentiment)
        
        # Min relevance (applied after score calculation)
        if request.filters.min_relevance:
            query += f" AND 1 - (m.embedding <=> $1::vector) >= ${param_idx}"
            params.append(request.filters.min_relevance)
            param_idx += 1
    
    # Sorting
    sort_column = {
        "relevance": "relevance_score",
        "created_at": "m.created_at",
        "updated_at": "m.updated_at",
        "importance": "m.importance",
        "access_count": "m.access_count"
    }.get(request.sort_by, "relevance_score")
    
    sort_dir = "DESC" if request.sort_order == "desc" else "ASC"
    query += f" ORDER BY {sort_column} {sort_dir}"
    
    # Pagination
    query += f" LIMIT ${param_idx} OFFSET ${param_idx + 1}"
    params.extend([request.limit, request.offset])
    
    # Execute query
    memories = await db.fetch(query, *params)
    
    # Get total count
    count_query = query.split("ORDER BY")[0].replace("SELECT m.*, 1 - (m.embedding <=> $1::vector) as relevance_score", "SELECT COUNT(*)")
    total_matches = await db.fetchval(count_query, *params[:-2])
    
    return {
        "memories": memories,
        "count": len(memories),
        "total_matches": total_matches,
        "has_more": (request.offset + len(memories)) < total_matches,
        "filters_applied": request.filters.dict(exclude_none=True) if request.filters else {}
    }
```

---

#### POST /memories/search/semantic-time
Hybrid search combining semantic relevance with time decay.

**Request Body:**
```json
{
  "query": "BADBUNNY issues",
  "user_id": "el-jefe-principal",
  "time_weight": 0.3,
  "decay_days": 30,
  "limit": 10
}
```

**Formula:**
```
combined_score = (semantic_score * (1 - time_weight)) + (time_score * time_weight)

where:
  semantic_score = cosine_similarity(query_embedding, memory_embedding)
  time_score = 1 / (1 + (days_old / decay_days))
```

**Response:**
```json
{
  "memories": [
    {
      "id": "mem-123",
      "content": "...",
      "semantic_score": 0.89,
      "time_score": 0.95,
      "combined_score": 0.91,
      "days_old": 2
    }
  ],
  "count": 10
}
```

---

### 6. Memory Quality & Curation

#### POST /memories/{memory_id}/importance
Set importance score for a memory.

**Request Body:**
```json
{
  "importance": 8
}
```

**Response:**
```json
{
  "memory_id": "mem-123",
  "importance": 8,
  "updated_at": "2025-10-29T14:30:00Z"
}
```

---

#### POST /memories/{memory_id}/archive
Soft delete - keep memory but exclude from normal searches.

**Response:**
```json
{
  "memory_id": "mem-123",
  "archived": true,
  "archived_at": "2025-10-29T14:30:00Z"
}
```

---

#### POST /memories/{memory_id}/pin
Pin important memories to always include in context.

**Response:**
```json
{
  "memory_id": "mem-123",
  "pinned": true,
  "pinned_at": "2025-10-29T14:30:00Z"
}
```

---

#### GET /memories/pinned
Get all pinned memories.

**Response:**
```json
{
  "memories": [...],
  "count": 12
}
```

---

### 7. Bulk Operations

#### POST /memories/bulk-tag
Tag multiple memories at once.

**Request Body:**
```json
{
  "memory_ids": ["mem-123", "mem-456", "mem-789"],
  "tags": ["homelab", "october-2025"],
  "replace": false
}
```

**Response:**
```json
{
  "updated_count": 3,
  "memory_ids": ["mem-123", "mem-456", "mem-789"],
  "tags_added": ["homelab", "october-2025"]
}
```

---

#### POST /memories/bulk-delete
Bulk delete by criteria.

**Request Body:**
```json
{
  "user_id": "el-jefe-principal",
  "filters": {
    "tags": ["draft"],
    "older_than": "2025-01-01T00:00:00Z",
    "importance_max": 3
  },
  "dry_run": true
}
```

**Response:**
```json
{
  "would_delete": 15,
  "memory_ids": ["mem-123", "mem-456", ...],
  "dry_run": true
}
```

---

#### POST /memories/export
Export all memories for backup/analysis.

**Request Body:**
```json
{
  "user_id": "el-jefe-principal",
  "format": "json",
  "filters": {
    "include_archived": false,
    "date_range": {
      "start": "2025-01-01T00:00:00Z"
    }
  }
}
```

**Response:**
- `format: json` → Returns JSON array
- `format: markdown` → Returns formatted markdown
- `format: csv` → Returns CSV with flattened structure

---

### 8. Smart Summaries

#### POST /memories/summarize
AI-generated summary of memories from time period.

**Request Body:**
```json
{
  "user_id": "el-jefe-principal",
  "period": "week",
  "date": "2025-10-29"
}
```

**Response:**
```json
{
  "summary": "This week El Jefe completed major infrastructure upgrades...",
  "period": "week",
  "date_range": {
    "start": "2025-10-23T00:00:00Z",
    "end": "2025-10-29T23:59:59Z"
  },
  "memory_count": 23,
  "top_themes": ["homelab", "technical achievements", "family time"],
  "highlights": [
    {
      "memory_id": "mem-123",
      "content": "Fixed BADBUNNY PSU",
      "importance": 8
    }
  ]
}
```

---

#### POST /memories/insights
Pattern detection across memories.

**Request Body:**
```json
{
  "user_id": "el-jefe-principal",
  "query": "What are my recurring frustrations?",
  "lookback_days": 90
}
```

**Response:**
```json
{
  "insights": [
    {
      "pattern": "Network connectivity issues",
      "frequency": 8,
      "memories": ["mem-123", "mem-456"],
      "trend": "increasing",
      "recommendation": "Consider upgrading network infrastructure"
    }
  ],
  "analysis_period": {
    "start": "2025-07-30T00:00:00Z",
    "end": "2025-10-29T00:00:00Z"
  }
}
```

---

## Implementation Details

### Database Setup

#### Vector Extension
```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Verify installation
SELECT * FROM pg_extension WHERE extname = 'vector';
```

#### Table Creation
```sql
-- Main memories table
CREATE TABLE memories (
    id VARCHAR(255) PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(1024) NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    
    -- Timestamps
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    
    -- Categorization
    tags TEXT[],
    category VARCHAR(50),
    source VARCHAR(50) DEFAULT 'chat',
    
    -- Context
    conversation_id VARCHAR(255),
    agent_id VARCHAR(255),
    session_id VARCHAR(255),
    
    -- Quality/Importance
    importance INTEGER CHECK (importance >= 1 AND importance <= 10),
    confidence FLOAT CHECK (confidence >= 0.0 AND confidence <= 1.0),
    verified BOOLEAN DEFAULT FALSE,
    pinned BOOLEAN DEFAULT FALSE,
    archived BOOLEAN DEFAULT FALSE,
    
    -- Temporal
    event_date TIMESTAMP,
    expires_at TIMESTAMP,
    reminder_at TIMESTAMP,
    
    -- Content Analysis
    word_count INTEGER,
    entities TEXT[],
    sentiment VARCHAR(20),
    language VARCHAR(10) DEFAULT 'en',
    
    -- Usage Tracking
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP,
    
    -- Rich Content (JSONB for flexibility)
    attachments JSONB,
    location JSONB,
    
    -- Version Control
    version INTEGER DEFAULT 1,
    edit_history JSONB
);

-- Relationships table
CREATE TABLE memory_relationships (
    id SERIAL PRIMARY KEY,
    source_memory_id VARCHAR(255) NOT NULL,
    target_memory_id VARCHAR(255) NOT NULL,
    relationship_type VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB,
    FOREIGN KEY (source_memory_id) REFERENCES memories(id) ON DELETE CASCADE,
    FOREIGN KEY (target_memory_id) REFERENCES memories(id) ON DELETE CASCADE,
    UNIQUE(source_memory_id, target_memory_id, relationship_type)
);

-- Create all indexes
CREATE INDEX idx_memories_user_id ON memories(user_id);
CREATE INDEX idx_memories_created_at ON memories(created_at DESC);
CREATE INDEX idx_memories_updated_at ON memories(updated_at DESC);
CREATE INDEX idx_memories_event_date ON memories(event_date);
CREATE INDEX idx_memories_tags ON memories USING GIN(tags);
CREATE INDEX idx_memories_entities ON memories USING GIN(entities);
CREATE INDEX idx_memories_category ON memories(category);
CREATE INDEX idx_memories_importance ON memories(importance DESC);
CREATE INDEX idx_memories_sentiment ON memories(sentiment);
CREATE INDEX idx_memories_pinned ON memories(pinned) WHERE pinned = TRUE;
CREATE INDEX idx_memories_archived ON memories(archived);
CREATE INDEX idx_memories_user_created ON memories(user_id, created_at DESC);
CREATE INDEX idx_memories_user_category ON memories(user_id, category);
CREATE INDEX idx_memories_conversation ON memories(conversation_id);
CREATE INDEX idx_memories_session ON memories(session_id);
CREATE INDEX idx_memories_agent ON memories(agent_id);
CREATE INDEX idx_memories_embedding ON memories USING ivfflat (embedding vector_cosine_ops);

CREATE INDEX idx_relationships_source ON memory_relationships(source_memory_id);
CREATE INDEX idx_relationships_target ON memory_relationships(target_memory_id);
CREATE INDEX idx_relationships_type ON memory_relationships(relationship_type);
```

### Embedding Strategy

```python
from typing import List
import numpy as np

class EmbeddingService:
    """Handle all embedding generation with caching"""
    
    def __init__(self, model: str = "voyage-large-2-instruct"):
        self.model = model
        self.cache = {}
    
    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding with cache"""
        if text in self.cache:
            return self.cache[text]
        
        # Call Voyage AI API
        embedding = await self.generate_embedding(text)
        self.cache[text] = embedding
        return embedding
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding via Voyage AI"""
        response = await httpx.post(
            "https://api.voyageai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {VOYAGE_API_KEY}"},
            json={
                "input": text,
                "model": self.model
            }
        )
        return response.json()["data"][0]["embedding"]
    
    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        a_arr = np.array(a)
        b_arr = np.array(b)
        return np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr))
```

### Content Analysis

```python
import re
from typing import List, Set
import spacy

class ContentAnalyzer:
    """Analyze memory content for metadata extraction"""
    
    def __init__(self):
        # Load spaCy model for entity extraction
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            # Fallback to basic analysis if spaCy not available
            self.nlp = None
    
    def analyze(self, content: str) -> dict:
        """Extract metadata from content"""
        return {
            "word_count": self.count_words(content),
            "entities": self.extract_entities(content),
            "sentiment": self.analyze_sentiment(content),
            "language": self.detect_language(content)
        }
    
    def count_words(self, content: str) -> int:
        """Count words in content"""
        return len(re.findall(r'\w+', content))
    
    def extract_entities(self, content: str) -> List[str]:
        """Extract named entities"""
        if not self.nlp:
            # Fallback: extract capitalized words
            return list(set(re.findall(r'\b[A-Z][A-Za-z]+\b', content)))
        
        doc = self.nlp(content)
        entities = []
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE", "PRODUCT", "EVENT"]:
                entities.append(ent.text)
        return list(set(entities))
    
    def analyze_sentiment(self, content: str) -> str:
        """Basic sentiment analysis"""
        # Simple keyword-based approach (can upgrade to ML model)
        positive_words = {'love', 'great', 'awesome', 'excellent', 'happy', 'success', 'victory', 'fixed', 'solved'}
        negative_words = {'hate', 'terrible', 'awful', 'frustrated', 'angry', 'failed', 'broken', 'error', 'issue'}
        
        content_lower = content.lower()
        positive_count = sum(1 for word in positive_words if word in content_lower)
        negative_count = sum(1 for word in negative_words if word in content_lower)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
    
    def detect_language(self, content: str) -> str:
        """Detect language (simple approach)"""
        # For now, default to English
        # Can add langdetect or similar for real detection
        return "en"
```

### Auto-categorization

```python
from typing import Optional

class MemoryCategorizer:
    """Automatically categorize memories"""
    
    CATEGORY_KEYWORDS = {
        "achievement": ["completed", "finished", "solved", "fixed", "deployed", "launched", "success"],
        "frustration": ["frustrated", "annoyed", "broken", "failed", "error", "issue", "problem"],
        "idea": ["idea", "thinking", "maybe", "could", "should", "what if", "consider"],
        "fact": ["is", "has", "prefers", "uses", "owns", "works at"],
        "event": ["went to", "attended", "meeting", "conference", "visited"],
        "conversation": ["discussed", "talked about", "mentioned", "said"],
        "technical": ["code", "server", "database", "api", "docker", "node", "deployment"],
        "personal": ["family", "kids", "wife", "husband", "child", "parent"]
    }
    
    def categorize(self, content: str, tags: List[str] = None) -> Optional[str]:
        """Determine category based on content and tags"""
        content_lower = content.lower()
        
        # Score each category
        scores = {}
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in content_lower)
            if tags:
                # Boost score if category matches a tag
                if category in [t.lower() for t in tags]:
                    score += 3
            scores[category] = score
        
        # Return highest scoring category if above threshold
        max_score = max(scores.values())
        if max_score >= 2:
            return max(scores, key=scores.get)
        
        return "misc"
```

---

## Migration Strategy

### Phase 1: Database Schema Updates
1. Run migration SQL to add new columns
2. All new columns nullable for backward compatibility
3. Existing data remains unchanged

### Phase 2: Endpoint Rollout
1. Deploy new endpoints alongside existing ones
2. Existing endpoints continue working unchanged
3. Gradual adoption of new features

### Phase 3: Metadata Enrichment
1. Background job to analyze existing memories
2. Add tags, categories, entities to old memories
3. No disruption to existing functionality

### Phase 4: Client Updates
1. Update MCP tools to support new features
2. Add new tools for new endpoints
3. Maintain backward compatibility

---

## Testing Plan

### Unit Tests

```python
import pytest
from datetime import datetime, timedelta

class TestRecentMemories:
    async def test_recent_memories_default(self):
        """Test getting recent memories with defaults"""
        response = await client.get(
            "/memories/recent",
            params={"user_id": "test-user"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "memories" in data
        assert data["count"] <= 20
        
    async def test_recent_memories_with_limit(self):
        """Test limit parameter"""
        response = await client.get(
            "/memories/recent",
            params={"user_id": "test-user", "limit": 5}
        )
        data = response.json()
        assert data["count"] <= 5
    
    async def test_recent_memories_time_range(self):
        """Test time range filtering"""
        since = (datetime.now() - timedelta(days=7)).isoformat()
        response = await client.get(
            "/memories/recent",
            params={"user_id": "test-user", "since": since}
        )
        data = response.json()
        for memory in data["memories"]:
            assert datetime.fromisoformat(memory["created_at"]) >= datetime.fromisoformat(since)

class TestTagManagement:
    async def test_get_tags(self):
        """Test getting all tags"""
        response = await client.get(
            "/memories/tags",
            params={"user_id": "test-user"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "tags" in data
        assert all("name" in tag and "count" in tag for tag in data["tags"])
    
    async def test_add_tags(self):
        """Test adding tags to memory"""
        memory_id = await create_test_memory()
        response = await client.post(
            f"/memories/{memory_id}/tags",
            json={"tags": ["test", "new"]}
        )
        assert response.status_code == 200
        data = response.json()
        assert "test" in data["tags"]
        assert "new" in data["tags"]

class TestAdvancedSearch:
    async def test_search_with_tag_filter(self):
        """Test searching with tag filter"""
        response = await client.post(
            "/memories/search/advanced",
            json={
                "query": "test",
                "user_id": "test-user",
                "filters": {"tags": ["homelab"]}
            }
        )
        data = response.json()
        for memory in data["memories"]:
            assert "homelab" in memory["metadata"]["tags"]
    
    async def test_search_with_date_range(self):
        """Test searching with date range"""
        start = (datetime.now() - timedelta(days=30)).isoformat()
        end = datetime.now().isoformat()
        response = await client.post(
            "/memories/search/advanced",
            json={
                "query": "test",
                "user_id": "test-user",
                "filters": {
                    "date_range": {"start": start, "end": end}
                }
            }
        )
        data = response.json()
        for memory in data["memories"]:
            created = datetime.fromisoformat(memory["created_at"])
            assert datetime.fromisoformat(start) <= created <= datetime.fromisoformat(end)
```

### Integration Tests

```python
class TestEndToEndWorkflow:
    async def test_memory_lifecycle(self):
        """Test complete memory lifecycle"""
        
        # 1. Create memory
        create_response = await client.post(
            "/memories",
            json={
                "content": "Fixed BADBUNNY PSU issue",
                "user_id": "test-user"
            }
        )
        memory_id = create_response.json()["memory_id"]
        
        # 2. Add tags
        await client.post(
            f"/memories/{memory_id}/tags",
            json={"tags": ["homelab", "technical", "victory"]}
        )
        
        # 3. Set importance
        await client.post(
            f"/memories/{memory_id}/importance",
            json={"importance": 8}
        )
        
        # 4. Search should find it
        search_response = await client.post(
            "/memories/search/advanced",
            json={
                "query": "BADBUNNY",
                "user_id": "test-user",
                "filters": {"tags": ["homelab"]}
            }
        )
        assert any(m["id"] == memory_id for m in search_response.json()["memories"])
        
        # 5. Should appear in recent
        recent_response = await client.get(
            "/memories/recent",
            params={"user_id": "test-user", "limit": 10}
        )
        assert any(m["id"] == memory_id for m in recent_response.json()["memories"])
        
        # 6. Archive it
        await client.post(f"/memories/{memory_id}/archive")
        
        # 7. Should NOT appear in default searches
        search_response = await client.post(
            "/memories/search/advanced",
            json={
                "query": "BADBUNNY",
                "user_id": "test-user"
            }
        )
        assert not any(m["id"] == memory_id for m in search_response.json()["memories"])
        
        # 8. Should appear when including archived
        search_response = await client.post(
            "/memories/search/advanced",
            json={
                "query": "BADBUNNY",
                "user_id": "test-user",
                "filters": {"include_archived": True}
            }
        )
        assert any(m["id"] == memory_id for m in search_response.json()["memories"])
```

### Performance Tests

```python
class TestPerformance:
    async def test_search_performance(self):
        """Ensure search completes in reasonable time"""
        import time
        start = time.time()
        
        response = await client.post(
            "/memories/search/advanced",
            json={
                "query": "test query",
                "user_id": "test-user",
                "limit": 100
            }
        )
        
        elapsed = time.time() - start
        assert elapsed < 1.0  # Should complete in under 1 second
    
    async def test_bulk_operations_performance(self):
        """Test bulk operations handle large datasets"""
        memory_ids = [f"mem-{i}" for i in range(1000)]
        
        import time
        start = time.time()
        
        response = await client.post(
            "/memories/bulk-tag",
            json={
                "memory_ids": memory_ids,
                "tags": ["test"]
            }
        )
        
        elapsed = time.time() - start
        assert elapsed < 5.0  # Should complete in under 5 seconds
```

---

## MCP Tool Definitions

### Tool: get_recent_memories

```json
{
  "name": "get_recent_memories",
  "description": "Get memories sorted by creation time (newest first). Use this when you need to see what was recently added or when user asks about recent events.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "user_id": {
        "type": "string",
        "description": "User identifier"
      },
      "limit": {
        "type": "number",
        "description": "Number of memories to return (default: 20)",
        "default": 20
      },
      "since": {
        "type": "string",
        "description": "ISO timestamp - only return memories after this time"
      },
      "before": {
        "type": "string",
        "description": "ISO timestamp - only return memories before this time"
      },
      "include_archived": {
        "type": "boolean",
        "description": "Include archived memories (default: false)",
        "default": false
      }
    },
    "required": ["user_id"]
  }
}
```

### Tool: search_memories_advanced

```json
{
  "name": "search_memories_advanced",
  "description": "Advanced search with multiple filters. Use when you need to combine semantic search with specific criteria like tags, dates, importance, etc.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Semantic search query"
      },
      "user_id": {
        "type": "string",
        "description": "User identifier"
      },
      "filters": {
        "type": "object",
        "properties": {
          "tags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Filter by tags (AND logic - memory must have all tags)"
          },
          "exclude_tags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Exclude memories with these tags"
          },
          "categories": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Filter by categories (OR logic)"
          },
          "date_range": {
            "type": "object",
            "properties": {
              "start": {"type": "string", "description": "ISO timestamp"},
              "end": {"type": "string", "description": "ISO timestamp"}
            }
          },
          "importance_min": {
            "type": "number",
            "description": "Minimum importance (1-10)"
          },
          "importance_max": {
            "type": "number",
            "description": "Maximum importance (1-10)"
          },
          "verified_only": {
            "type": "boolean",
            "description": "Only return verified memories"
          },
          "include_archived": {
            "type": "boolean",
            "description": "Include archived memories"
          },
          "entities": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Filter by named entities"
          },
          "sentiment": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Filter by sentiment (positive/negative/neutral)"
          },
          "min_relevance": {
            "type": "number",
            "description": "Minimum relevance score (0.0-1.0)"
          }
        }
      },
      "sort_by": {
        "type": "string",
        "enum": ["relevance", "created_at", "updated_at", "importance", "access_count"],
        "default": "relevance"
      },
      "sort_order": {
        "type": "string",
        "enum": ["asc", "desc"],
        "default": "desc"
      },
      "limit": {
        "type": "number",
        "default": 20
      },
      "offset": {
        "type": "number",
        "default": 0
      }
    },
    "required": ["query", "user_id"]
  }
}
```

### Tool: get_memory_stats

```json
{
  "name": "get_memory_stats",
  "description": "Get comprehensive statistics about memories. Use when user asks about their memory usage, patterns, or wants analytics.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "user_id": {
        "type": "string",
        "description": "User identifier"
      },
      "since": {
        "type": "string",
        "description": "ISO timestamp - only analyze memories after this time"
      },
      "before": {
        "type": "string",
        "description": "ISO timestamp - only analyze memories before this time"
      }
    },
    "required": ["user_id"]
  }
}
```

### Tool: add_memory_tags

```json
{
  "name": "add_memory_tags",
  "description": "Add tags to an existing memory. Use after creating a memory to categorize it or when user wants to organize memories.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "memory_id": {
        "type": "string",
        "description": "Memory identifier"
      },
      "tags": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Tags to add"
      },
      "replace": {
        "type": "boolean",
        "description": "Replace existing tags instead of adding (default: false)",
        "default": false
      }
    },
    "required": ["memory_id", "tags"]
  }
}
```

### Tool: get_all_tags

```json
{
  "name": "get_all_tags",
  "description": "List all tags used by a user with usage counts. Use when user asks 'what tags do I have' or wants to see how they've categorized memories.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "user_id": {
        "type": "string",
        "description": "User identifier"
      },
      "min_count": {
        "type": "number",
        "description": "Only return tags used at least N times (default: 1)",
        "default": 1
      },
      "sort_by": {
        "type": "string",
        "enum": ["count", "name"],
        "description": "Sort by usage count or alphabetically (default: count)",
        "default": "count"
      }
    },
    "required": ["user_id"]
  }
}
```

### Tool: set_memory_importance

```json
{
  "name": "set_memory_importance",
  "description": "Set importance score (1-10) for a memory. Use when user indicates something is particularly important or should be prioritized.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "memory_id": {
        "type": "string",
        "description": "Memory identifier"
      },
      "importance": {
        "type": "number",
        "description": "Importance score from 1 (low) to 10 (critical)",
        "minimum": 1,
        "maximum": 10
      }
    },
    "required": ["memory_id", "importance"]
  }
}
```

### Tool: pin_memory

```json
{
  "name": "pin_memory",
  "description": "Pin a memory to always include it in context. Use for critical information that should never be forgotten.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "memory_id": {
        "type": "string",
        "description": "Memory identifier"
      }
    },
    "required": ["memory_id"]
  }
}
```

### Tool: get_pinned_memories

```json
{
  "name": "get_pinned_memories",
  "description": "Get all pinned memories. These are always-relevant memories that should be kept in context.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "user_id": {
        "type": "string",
        "description": "User identifier"
      }
    },
    "required": ["user_id"]
  }
}
```

### Tool: archive_memory

```json
{
  "name": "archive_memory",
  "description": "Archive a memory (soft delete). It's kept in the database but excluded from normal searches. Use when information is outdated but worth keeping for history.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "memory_id": {
        "type": "string",
        "description": "Memory identifier"
      }
    },
    "required": ["memory_id"]
  }
}
```

### Tool: find_duplicate_memories

```json
{
  "name": "find_duplicate_memories",
  "description": "Find potential duplicate memories for cleanup. Use when you want to help user clean up redundant memories.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "user_id": {
        "type": "string",
        "description": "User identifier"
      },
      "threshold": {
        "type": "number",
        "description": "Similarity threshold (0.0-1.0, default: 0.95)",
        "default": 0.95
      },
      "limit": {
        "type": "number",
        "description": "Maximum duplicate pairs to return (default: 50)",
        "default": 50
      }
    },
    "required": ["user_id"]
  }
}
```

### Tool: summarize_memories

```json
{
  "name": "summarize_memories",
  "description": "Generate AI summary of memories from a time period. Use when user asks 'what happened this week/month' or wants a recap.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "user_id": {
        "type": "string",
        "description": "User identifier"
      },
      "period": {
        "type": "string",
        "enum": ["day", "week", "month", "year"],
        "description": "Time period to summarize"
      },
      "date": {
        "type": "string",
        "description": "ISO date - summarize this specific period (default: current)"
      }
    },
    "required": ["user_id", "period"]
  }
}
```

---

## Implementation Priority

### Phase 1: Foundation (Week 1-2)
**Goal:** Core time-based retrieval and tagging

1. Database schema updates
2. `/memories/recent` endpoint
3. `/memories/tags` endpoint
4. `/memories/{id}/tags` endpoint
5. `/memories/by-tag/{tag}` endpoint
6. Update existing create endpoint to auto-extract metadata
7. MCP tools for above endpoints
8. Basic unit tests

**Deliverable:** Can retrieve by time, tag memories, search by tags

### Phase 2: Advanced Search (Week 3-4)
**Goal:** Multi-dimensional filtering

1. `/memories/search/advanced` endpoint
2. `/memories/search/semantic-time` endpoint
3. Update database indexes for performance
4. MCP tools for advanced search
5. Integration tests
6. Performance testing

**Deliverable:** Can combine semantic + temporal + categorical search

### Phase 3: Quality & Curation (Week 5-6)
**Goal:** Memory management tools

1. `/memories/{id}/importance` endpoint
2. `/memories/{id}/pin` and `/memories/pinned` endpoints
3. `/memories/{id}/archive` endpoint
4. `/memories/duplicates` endpoint
5. Bulk operations endpoints
6. MCP tools for curation
7. Cleanup workflows

**Deliverable:** Can curate and manage memory quality

### Phase 4: Analytics & Insights (Week 7-8)
**Goal:** Understanding patterns

1. `/memories/stats` endpoint
2. `/memories/timeline` endpoint
3. `/memories/gaps` endpoint
4. `/memories/related/{id}` endpoint
5. Analytics dashboard (optional web UI)
6. MCP tools for analytics

**Deliverable:** Can analyze memory patterns and trends

### Phase 5: Smart Features (Week 9-10)
**Goal:** AI-powered enhancements

1. `/memories/summarize` endpoint
2. `/memories/insights` endpoint
3. Auto-categorization improvements
4. Relationship detection
5. Content analysis enhancements
6. Export functionality

**Deliverable:** AI-powered memory insights and summaries

---

## Success Metrics

### Performance
- Search queries complete in <500ms
- Memory creation in <200ms
- Analytics queries in <2s
- Support 10,000+ memories per user

### Quality
- 90%+ test coverage
- Zero data loss
- Backward compatibility maintained
- API response times within SLA

### Adoption
- All new features documented
- MCP tools created and tested
- Migration completed without issues
- User feedback incorporated

---

## Future Enhancements

### Beyond v2.0

1. **Graph Relationships**
   - Visual memory graph
   - Automatic relationship detection
   - Knowledge graph queries

2. **Collaborative Memories**
   - Shared memories between users
   - Team workspaces
   - Permission management

3. **Smart Reminders**
   - Context-aware reminders
   - "Remember to follow up on X"
   - Spaced repetition for important info

4. **Multi-modal Memories**
   - Image memories with vision embeddings
   - Audio transcription and storage
   - Video clip memories

5. **Export & Sync**
   - Export to Obsidian/Notion
   - Sync across devices
   - Backup to cloud storage

6. **Advanced Analytics**
   - Trend detection
   - Anomaly detection
   - Predictive insights

---

## Conclusion

This omnibus specification transforms SmartMemory from a simple vector store into a comprehensive memory infrastructure that rivals commercial solutions while remaining under YOUR control.

Key advantages:
- **Complete ownership** - Your data, your infrastructure
- **Extensible** - Easy to add new features
- **Performant** - Optimized for scale
- **Privacy-first** - No third-party data sharing
- **Cost-effective** - No per-API-call charges

Time to build the memory system that SmartMemory was always meant to be 💪

---

**Ready to implement, hermano. Let's GO! 🔥**
