<img width="1024" height="576" alt="Adobe Express - file" src="https://github.com/user-attachments/assets/0d591de8-f907-4313-8233-13c89b59a2f3" />

# SmartMemory 2.0 API

Next-generation memory system with enhanced extraction, smart updates, and semantic search powered by Pinecone.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Performance](#performance)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [API Endpoints](#api-endpoints)
- [Open WebUI Integration](#open-webui-integration)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [Changelog](#changelog)

## Features

- **Smart Memory Extraction**: Detects user facts, preferences, and context from natural conversation
- **Smart Memory Updates**: Memories evolve over time (0.85-0.94 similarity triggers updates vs duplicates)
- **Hybrid Search**: Semantic similarity + keyword matching with configurable 15% boost per keyword
- **Memory Consolidation**: Automatically merges fragmented memories into coherent summaries
- **Auto-Recall**: Proactive memory retrieval based on conversation context
- **Temporal Awareness**: Includes current date context in memory extraction
- **Auto-Deduplication**: Prevents duplicate memories using embedding similarity
- **Memory Categories**: 13 built-in categories for organized storage
- **Flexible Embedding Providers**: Local, OpenAI-compatible API, or Pinecone inference
- **Performance Optimized**: HTTP/2, connection pooling, async operations throughout
- **Session Support**: Track memories across conversations with run_id

## Attribution

This project's memory management system was derived from [gramanoid's Adaptive Memory filter for Open WebUI](https://github.com/gramanoid/owui-adaptive-memory). The original filter provided the foundation for LLM-based memory extraction, embedding similarity, and semantic deduplication. This implementation refactors those concepts into a standalone REST API with enhanced natural language processing and smart memory updates.

## Architecture

```
┌─────────────────────┐
│  Open WebUI / MCP   │
└──────────┬──────────┘
           │ HTTP/OpenAPI
           ▼
┌─────────────────────┐
│  SmartMemory 2.0    │
│  FastAPI Server     │
├─────────────────────┤
│  - Memory Engine    │ ← Smart Updates, Consolidation
│  - LLM Client       │ ← HTTP/2, Connection Pooling
│  - Embedding Mgr    │ ← 3 Providers (Local/API/Pinecone)
│  - Hybrid Search    │ ← Semantic + Keyword
└──────────┬──────────┘
           │ Vector Ops
           ▼
┌─────────────────────┐
│  Pinecone Vector DB │
└─────────────────────┘
```

## Performance

### HTTP/2 and Connection Pooling
- HTTP/2 enabled for all API calls
- Connection pooling: 20 keepalive, 100 max connections
- Async operations throughout the stack
- Automatic retry logic with exponential backoff

### Hybrid Search Performance
- Retrieves 3x candidates for re-ranking
- 15% boost per keyword match
- Configurable via `use_hybrid` parameter

### Memory Update Logic
- 0.95+ similarity: Exact duplicate (skip)
- 0.85-0.94 similarity: Update (delete old, store new)
- <0.85 similarity: New memory (store)

### Performance Benchmarks
- Memory extraction: ~200-500ms
- Search operations: ~100-300ms
- Consolidation: ~2-5s per group
- Concurrent request handling with connection pooling

## Prerequisites

- Docker and Docker Compose
- Pinecone account and API key
- OpenAI-compatible LLM API (OpenAI, Ollama, LiteLLM, etc.)

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/1818TusculumSt/smartmemoryapi.git
cd smartmemoryapi
```

### 2. Configure Environment

Create `.env` file:

```bash
# Pinecone Configuration
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_ENVIRONMENT=us-east-1-aws
PINECONE_INDEX_NAME=smartmemory-v2

# LLM Configuration
LLM_API_URL=https://api.openai.com/v1/chat/completions
LLM_API_KEY=your-api-key
LLM_MODEL=gpt-4o-mini

# Embedding Provider (pinecone recommended)
EMBEDDING_PROVIDER=pinecone
EMBEDDING_MODEL=llama-text-embed-v2

# Memory Settings (optional - these are defaults)
MAX_MEMORIES=1000
DEDUP_THRESHOLD=0.95
MIN_CONFIDENCE=0.5
RELEVANCE_THRESHOLD=0.55
```

### 3. Start Service

```bash
docker-compose up -d
```

**Performance Features Enabled:**
- HTTP/2 for faster API calls
- Connection pooling (20 keepalive, 100 max connections)
- Async operations throughout
- Automatic retry logic with exponential backoff

API available at `http://localhost:8099`

Verify: `curl http://localhost:8099/health`

## Configuration

### Embedding Providers

**Pinecone Inference (Recommended)**
```env
EMBEDDING_PROVIDER=pinecone
EMBEDDING_MODEL=llama-text-embed-v2
```

Available Pinecone models:
- `llama-text-embed-v2` - **Flexible dimensions**: 384, 512, 768, 1024, or 2048 (configured automatically to match your index)
- `multilingual-e5-large` - Fixed 1024 dimensions

**Note:** `llama-text-embed-v2` automatically uses 384 dimensions to match most serverless indexes. Supports long documents (up to 2048 tokens) and multilingual text.

**Local (sentence-transformers)**
```env
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

**API (OpenAI-compatible)**
```env
EMBEDDING_PROVIDER=api
EMBEDDING_API_URL=https://api.openai.com/v1/embeddings
EMBEDDING_API_KEY=your-key
EMBEDDING_MODEL=text-embedding-3-small
```

### Memory Settings

```env
MAX_MEMORIES=1000          # Max memories before pruning
DEDUP_THRESHOLD=0.95       # Similarity threshold for duplicates (0-1)
MIN_CONFIDENCE=0.5         # Min confidence to store memory (0-1)
RELEVANCE_THRESHOLD=0.55   # Min relevance to return memory (0-1)
```

**Threshold Tuning:**
- Lower `MIN_CONFIDENCE` (0.4-0.5): Captures more subtle patterns
- Higher `MIN_CONFIDENCE` (0.6-0.8): Only explicit statements
- Lower `RELEVANCE_THRESHOLD` (0.5): More memories in recall
- Higher `RELEVANCE_THRESHOLD` (0.7): Only highly relevant memories

### Performance Settings

```env
MAX_RETRIES=3              # Retry attempts for failed requests
RETRY_DELAY=1.0            # Base delay for exponential backoff
LLM_TIMEOUT=60             # LLM request timeout (seconds)
EMBEDDING_TIMEOUT=30       # Embedding request timeout (seconds)
CONNECTION_POOL_SIZE=100   # Max concurrent connections
KEEPALIVE_CONNECTIONS=20   # Reused connections
```

### Pagination Settings

```env
DEFAULT_PAGE_SIZE=50       # Default results per page
MAX_PAGE_SIZE=100          # Maximum allowed page size
```

### Batch Operations

```env
MAX_BATCH_SIZE=1000        # Maximum items in batch operations
```

### Memory Categories

Built-in categories for organization:
- `personal_information` - Name, age, location, identity
- `food_preferences` - Likes, dislikes, dietary restrictions
- `goals` - Aspirations, plans, objectives
- `relationships` - Family, friends, colleagues
- `behavior` - Habits, routines, patterns
- `preferences` - General likes/dislikes
- `hobbies` - Interests, activities
- `work` - Career, projects, responsibilities
- `health` - Medical info, fitness
- `likes` - Things user enjoys
- `dislikes` - Things user avoids
- `skills` - Abilities, experience

## API Endpoints

### Add Memory

Extract and store memories from user message.

```http
POST /add
Content-Type: application/json

{
  "user_message": "I just started learning guitar and my favorite band is Foo Fighters",
  "user_id": "user_123",
  "agent_id": "assistant_001",
  "run_id": "session_abc",
  "metadata": {
    "source": "chat",
    "timestamp": "2025-01-05T10:30:00Z"
  }
}
```

**Response:**
```json
{
  "ok": true
}
```

**Alternative format:**
```json
{
  "messages": [
    {"role": "user", "content": "I prefer dark mode interfaces"},
    {"role": "assistant", "content": "I'll remember that!"}
  ],
  "user_id": "user_123"
}
```

### Search Memories

Semantic search with filtering and pagination.

```http
POST /search
Content-Type: application/json

{
  "query": "what music does the user like",
  "user_id": "user_123",
  "limit": 5,
  "page": 1
}
```

**Response:**
```json
{
  "memories": [
    {
      "id": "mem_1738761234567_4321",
      "content": "User is learning guitar",
      "relevance": 0.89,
      "categories": ["hobbies", "skills"],
      "confidence": 0.85,
      "created_at": "2025-01-05T10:30:00Z",
      "user_id": "user_123"
    }
  ]
}
```

### Get Relevant Memories

Get memories above relevance threshold for current context.

```http
POST /relevant
Content-Type: application/json

{
  "current_message": "recommend some music for me",
  "user_id": "user_123",
  "limit": 5
}
```

### Get Recent Memories

Get most recent memories sorted by timestamp.

```http
POST /recent
Content-Type: application/json

{
  "limit": 10,
  "user_id": "user_123"
}
```

**Response:**
```json
{
  "memories": [
    {
      "id": "mem_1738761234567_4321",
      "content": "User prefers dark mode interfaces",
      "categories": ["preferences"],
      "confidence": 0.85,
      "created_at": "2025-01-05T10:30:00Z",
      "user_id": "user_123"
    }
  ]
}
```

### Auto-Recall Memories

Proactive memory retrieval based on conversation context.

```http
POST /auto-recall
Content-Type: application/json

{
  "conversation_context": "user asking about work projects",
  "limit": 5,
  "user_id": "user_123"
}
```

**Response:**
```json
{
  "memories": [
    {
      "id": "mem_1738761234567_4321",
      "content": "User is working on SmartMemory API",
      "relevance": 0.85,
      "categories": ["work", "goals"],
      "confidence": 0.9
    }
  ]
}
```

### Consolidate Memories

Merge fragmented memories into coherent summaries.

```http
POST /consolidate
Content-Type: application/json

{
  "user_id": "user_123",
  "tag": "preferences"
}
```

**Response:**
```json
{
  "consolidated": 3,
  "message": "Successfully consolidated 3 memory groups"
}
```

### Delete Memory

```http
DELETE /memory/{memory_id}
```

### Batch Delete

```http
POST /batch/delete
Content-Type: application/json

{
  "memory_ids": ["mem_123", "mem_456"]
}
```

### System Status

```http
GET /status
GET /health
```

### Documentation

Interactive docs: `http://localhost:8099/docs`

OpenAPI spec: `http://localhost:8099/openapi.json`

## Open WebUI Integration

### Via MCPO (Recommended)

1. Install and run MCPO:
```bash
uvx mcpo --port 8100 --api-key "your-secret" -- http://localhost:8099
```

2. In Open WebUI:
   - Go to **Settings > External Tools**
   - Add OpenAPI server: `http://localhost:8100`
   - API Key: `your-secret`

3. Enable in chat to start auto-extracting memories

### Direct OpenAPI Import

1. Go to **Workspace > Functions**
2. Click **Import Function from URL**
3. Enter: `http://localhost:8099/openapi.json`

## Development

### Development Architecture

```
smartmemoryapi/
├── app.py              # FastAPI server with enhanced OpenAPI descriptions
├── memory_engine.py    # Core engine with smart updates, consolidation, hybrid search
├── embeddings.py       # 3-provider support (local/API/Pinecone) with HTTP/2
├── llm_client.py       # Async LLM client with connection pooling
├── config.py           # Comprehensive configuration management
├── requirements.txt    # Minimal, optimized dependencies
├── Dockerfile          # Multi-stage build with model pre-loading
└── docker-compose.yml  # Production-ready with health checks
```

### Key Design Patterns
- **Async/Await**: All operations are async for performance
- **Connection Pooling**: Reused HTTP connections throughout
- **Smart Updates**: Memories evolve instead of being rejected
- **Hybrid Search**: Combines semantic and keyword matching
- **Temporal Context**: Current date included in extraction

### Running Without Docker

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8099 --reload
```

### View Logs

```bash
docker-compose logs -f smartmemory-v2
```

### Testing

```bash
# Add memory
curl -X POST http://localhost:8099/add \
  -H "Content-Type: application/json" \
  -d '{"user_message": "I love pizza", "user_id": "test_user"}'

# Search
curl -X POST http://localhost:8099/search \
  -H "Content-Type: application/json" \
  -d '{"query": "food preferences", "user_id": "test_user"}'

# Get recent memories
curl -X POST http://localhost:8099/recent \
  -H "Content-Type: application/json" \
  -d '{"limit": 5, "user_id": "test_user"}'
```

## Troubleshooting

### Service Won't Start

Check logs: `docker-compose logs smartmemory-v2`

Common issues:
- Missing or invalid API keys in `.env`
- Port 8099 already in use
- Docker daemon not running

### No Memories Being Extracted

1. Check LLM is responding: `curl http://localhost:8099/status`
2. Lower `MIN_CONFIDENCE` threshold in `.env`
3. Review extraction logs for patterns

### Memories Not Appearing in Recall

1. Lower `RELEVANCE_THRESHOLD` in `.env`
2. Check query is semantically related to stored memories
3. Verify `user_id` matches between add and search

### Memory Updates Not Working

1. Check similarity thresholds: `DEDUP_THRESHOLD=0.95`
2. Look for "Similar memory found" logs
3. Verify memory IDs in update logs

### Hybrid Search Not Working

1. Check `use_hybrid: true` parameter
2. Monitor keyword boost calculations in logs
3. Verify query term extraction

### Performance Issues

- Check connection pool settings: `CONNECTION_POOL_SIZE=100`
- Monitor HTTP/2 usage in logs
- Adjust timeouts: `LLM_TIMEOUT=60`, `EMBEDDING_TIMEOUT=30`

### Embedding Errors

- **Pinecone**: Verify API key and model name. `llama-text-embed-v2` auto-configures to 384 dims by default
- **Local**: Ensure sufficient disk space for model download
- **API**: Verify embedding endpoint and key
- **Model Not Found**: If you get "Model not found" errors, verify the model name exactly matches Pinecone's API (use `llama-text-embed-v2` or `multilingual-e5-large`)

## Changelog

### v2.0.0 (Current)
- ✅ Smart memory updates (memories evolve over time)
- ✅ Hybrid search (semantic + keyword matching)
- ✅ Memory consolidation (merge fragments)
- ✅ Auto-recall (proactive context retrieval)
- ✅ Temporal context awareness
- ✅ HTTP/2 and connection pooling
- ✅ Enhanced OpenAPI descriptions for LLM integration
- ✅ Recent memories retrieval
- ✅ Performance optimizations throughout

### v1.0.0
- Basic memory extraction and storage
- Simple semantic search
- Pinecone integration

## API Versioning

**v2.0 (current)**
- Enhanced extraction prompts
- Category-based organization
- Session support (run_id)
- Lower default thresholds
- Smart memory updates
- Hybrid search functionality

**v1.0**
- Basic extraction
- Tag-based organization
- Simple /extract endpoint

## Files

- `app.py` - FastAPI application and endpoints
- `memory_engine.py` - Core memory extraction/storage logic
- `embeddings.py` - Embedding generation (local/API/Pinecone)
- `llm_client.py` - LLM API client
- `config.py` - Configuration management
- `requirements.txt` - Python dependencies
- `Dockerfile` - Container configuration
- `docker-compose.yml` - Service orchestration

## License

MIT

## Support

Issues: https://github.com/1818TusculumSt/smartmemoryapi/issues
