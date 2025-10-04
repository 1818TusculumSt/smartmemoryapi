# SmartMemory API

Auto-extracting memory system with semantic search powered by Pinecone. FastAPI REST server with Docker deployment.

## Features

- **Automatic Memory Extraction**: Uses LLM to identify user-specific facts, preferences, and context
- **Semantic Search**: Vector-based similarity search using embeddings
- **Deduplication**: Prevents duplicate memories using embedding similarity
- **Flexible Embedding Providers**: Switch between local, API, or Pinecone inference
- **Auto-Pruning**: Maintains memory limit by removing oldest entries
- **OpenAPI Specification**: Ready for Open WebUI integration

## Prerequisites

- Docker and Docker Compose
- Pinecone account and API key
- OpenAI API key (or compatible LLM API)

## Installation

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
PINECONE_INDEX_NAME=adaptive-memory

# LLM Configuration
LLM_API_URL=https://api.openai.com/v1/chat/completions
LLM_API_KEY=your-openai-api-key
LLM_MODEL=gpt-4o-mini

# Embedding Provider
EMBEDDING_PROVIDER=pinecone
EMBEDDING_MODEL=llama-text-embed-v2

# Memory Settings
MAX_MEMORIES=200
DEDUP_THRESHOLD=0.95
MIN_CONFIDENCE=0.5
RELEVANCE_THRESHOLD=0.6
```

### 3. Create docker-compose.yml

```yaml
version: '3.8'

services:
  smartmemory:
    build: .
    container_name: smartmemory-api
    ports:
      - "8099:8099"
    env_file:
      - .env
    restart: unless-stopped
```

### 4. Start Service

```bash
docker-compose up -d
```

API available at `http://localhost:8099`

Verify: `curl http://localhost:8099/health`

## API Endpoints

### Extract Memories
```http
POST /extract
{
  "user_message": "I just adopted a cat named Whiskers",
  "recent_history": []
}
```

### Search Memories
```http
POST /search
{
  "query": "what pets do I have",
  "limit": 5
}
```

### Get Relevant Memories
```http
POST /relevant
{
  "current_message": "Tell me about my hobbies",
  "limit": 5
}
```

### Delete Memory
```http
DELETE /memory/{memory_id}
```

### System Status
```http
GET /status
GET /health
```

### Documentation
Interactive docs: `http://localhost:8099/docs`

OpenAPI spec: `http://localhost:8099/openapi.json`

## Configuration Options

### Embedding Providers

**Pinecone Inference (Recommended)**
```env
EMBEDDING_PROVIDER=pinecone
EMBEDDING_MODEL=llama-text-embed-v2
```

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
MAX_MEMORIES=200           # Max memories before pruning
DEDUP_THRESHOLD=0.95       # Similarity threshold for duplicates (0-1)
MIN_CONFIDENCE=0.5         # Min confidence to store memory (0-1)
RELEVANCE_THRESHOLD=0.6    # Min relevance to return memory (0-1)
```

## Open WebUI Integration

1. Go to **Workspace > Functions**
2. Click **Import Function from URL**
3. Enter: `http://localhost:8099/openapi.json`

SmartMemory functions will be available in all chats.

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
docker-compose logs -f smartmemory
```

## Troubleshooting

### Service Won't Start

Check logs: `docker-compose logs smartmemory`

Common issues:
- Missing or invalid API keys in `.env`
- Port 8099 already in use
- Docker daemon not running

### Memory Extraction Not Working

- Verify LLM API key and endpoint
- Check LLM model name matches provider
- Review logs for API errors

### Embedding Errors

- Pinecone: verify API key and model name
- Local: ensure sufficient disk space for model download
- API: verify embedding endpoint and key

## Architecture

```
Client (Open WebUI, etc)
      │
      └──── HTTP REST API (FastAPI)
                  │
                  ├──── Memory Engine
                  ├──── LLM Client (OpenAI API)
                  ├──── Embedding Provider
                  │
                  └──── Pinecone Vector Database
```

## Files

- `app.py` - FastAPI application and endpoints
- `memory_engine.py` - Core memory extraction/storage logic
- `embeddings.py` - Embedding generation (local/API/Pinecone)
- `llm_client.py` - LLM API client
- `config.py` - Configuration management
- `requirements.txt` - Python dependencies
- `Dockerfile` - Container configuration

## License

MIT
