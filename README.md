<img width="1024" height="576" alt="Adobe Express - file" src="https://github.com/user-attachments/assets/0d591de8-f907-4313-8233-13c89b59a2f3" />

# SmartMemory 2.0 API

Next-generation memory system with Mem0 compatibility, enhanced extraction, and semantic search powered by Pinecone.

## Features

- **Smart Memory Extraction**: Detects user facts, preferences, and context from natural conversation
- **Mem0-Compatible API**: Drop-in replacement with user_id, agent_id, run_id support
- **Semantic Search**: Vector-based similarity search with configurable thresholds
- **Auto-Deduplication**: Prevents duplicate memories using embedding similarity
- **Memory Categories**: Organized storage (personal_info, preferences, goals, relationships, etc.)
- **Flexible Embedding Providers**: Local, OpenAI-compatible API, or Pinecone inference
- **Auto-Pruning**: Maintains memory limits automatically
- **Session Support**: Track memories across conversations with run_id

## Attribution

This project's memory management system was derived from [gramanoid's Adaptive Memory filter for Open WebUI](https://github.com/gramanoid/owui-adaptive-memory). The original filter provided the foundation for LLM-based memory extraction, embedding similarity, and semantic deduplication. This implementation refactors those concepts into a standalone REST API with Mem0 compatibility and enhanced natural language processing.

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
EMBEDDING_MODEL=multilingual-e5-small

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

API available at `http://localhost:8099`

Verify: `curl http://localhost:8099/health`

## API Endpoints

### Add Memory (Mem0-compatible)

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

**Alternative format (Mem0 messages array):**
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

## Configuration

### Embedding Providers

**Pinecone Inference (Recommended)**
```env
EMBEDDING_PROVIDER=pinecone
EMBEDDING_MODEL=multilingual-e5-small
```

Available Pinecone models:
- `multilingual-e5-small` (384 dimensions) - Recommended for serverless, faster and cheaper
- `multilingual-e5-large` (1024 dimensions) - Higher quality, larger vectors

**Important:** Ensure your Pinecone index dimension matches the model:
- 384-dimensional index → use `multilingual-e5-small`
- 1024-dimensional index → use `multilingual-e5-large`

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

### Embedding Errors

- **Pinecone**: Verify API key and model name. Ensure model dimension matches index dimension (384 for multilingual-e5-small, 1024 for multilingual-e5-large)
- **Local**: Ensure sufficient disk space for model download
- **API**: Verify embedding endpoint and key
- **Dimension Mismatch**: If you get errors about vector dimensions, check that your EMBEDDING_MODEL dimension matches your Pinecone index dimension

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
│  - Memory Engine    │
│  - LLM Client       │
│  - Embedding Mgr    │
└──────────┬──────────┘
           │ Vector Ops
           ▼
┌─────────────────────┐
│  Pinecone Vector DB │
└─────────────────────┘
```

## API Versioning

**v2.0 (current)**
- Mem0-compatible endpoints
- Enhanced extraction prompts
- Category-based organization
- Session support (run_id)
- Lower default thresholds

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
