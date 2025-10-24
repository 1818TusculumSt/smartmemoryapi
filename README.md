<img width="1024" height="576" alt="Adobe Express - file" src="https://github.com/user-attachments/assets/0d591de8-f907-4313-8233-13c89b59a2f3" />

# SmartMemory 2.0 API

Next-generation memory system with smart extraction, semantic search, and auto-recall powered by Pinecone.

## Features

- **Smart Memory Extraction** - Detects facts, preferences, and context from conversation
- **Smart Updates** - Memories evolve (0.85-0.94 similarity) instead of duplicating
- **Hybrid Search** - Semantic + keyword matching with 15% boost per match
- **Auto-Recall** - Proactive retrieval based on conversation context
- **Memory Consolidation** - Merges fragmented memories automatically
- **Flexible Embeddings** - Local, OpenAI API, or Pinecone inference
- **High Performance** - HTTP/2, connection pooling, async throughout

**Built on:** [gramanoid's Adaptive Memory filter](https://github.com/gramanoid/owui-adaptive-memory)

## Prerequisites

- Docker & Docker Compose
- Pinecone API key ([get one free](https://www.pinecone.io/))
- OpenAI-compatible LLM (OpenAI, Ollama, LiteLLM, etc.)

## Quick Start

```bash
# 1. Clone and configure
git clone https://github.com/1818TusculumSt/smartmemoryapi.git
cd smartmemoryapi
cp .env.example .env  # Edit with your API keys

# 2. Start service
docker-compose up -d

# 3. Verify
curl http://localhost:8099/health
```

### Minimal `.env` Configuration

```env
# Required
PINECONE_API_KEY=your-key
PINECONE_INDEX_NAME=smartmemory-v2
LLM_API_URL=https://api.openai.com/v1/chat/completions
LLM_API_KEY=your-key
LLM_MODEL=gpt-4o-mini

# Recommended
EMBEDDING_PROVIDER=pinecone
EMBEDDING_MODEL=llama-text-embed-v2
```

API runs at `http://localhost:8099` | Docs at `http://localhost:8099/docs`

## Open WebUI Integration

### Via MCPO (Recommended)

```bash
# Install and run MCPO proxy
uvx mcpo --port 8100 --api-key "your-secret" -- http://localhost:8099
```

In Open WebUI:
1. **Settings → External Tools**
2. Add server: `http://localhost:8100`
3. API Key: `your-secret`
4. Enable in chat - done!

### Direct Import

1. **Workspace → Functions**
2. **Import from URL**: `http://localhost:8099/openapi.json`

## API Overview

| Endpoint | Purpose |
|----------|---------|
| `POST /add` | Extract and store memories from conversation |
| `POST /search` | Semantic search with hybrid keyword matching |
| `POST /relevant` | Get memories above relevance threshold |
| `POST /recent` | Recent memories by timestamp |
| `POST /auto-recall` | Proactive context-based retrieval |
| `POST /consolidate` | Merge fragmented memories |
| `DELETE /memory/{id}` | Delete specific memory |
| `POST /batch/delete` | Delete multiple memories |
| `GET /health` | Health check |
| `GET /status` | System status |

**Full API docs:** `http://localhost:8099/docs`

## Configuration Options

<details>
<summary>Embedding Providers</summary>

**Pinecone (Recommended):**
```env
EMBEDDING_PROVIDER=pinecone
EMBEDDING_MODEL=llama-text-embed-v2
```

**Local:** `EMBEDDING_PROVIDER=local` | **API:** `EMBEDDING_PROVIDER=api`

[Full configuration guide →](docs/CONFIGURATION.md)
</details>

<details>
<summary>Memory Tuning</summary>

```env
MIN_CONFIDENCE=0.5         # Lower = more memories captured
RELEVANCE_THRESHOLD=0.55   # Lower = more memories recalled
DEDUP_THRESHOLD=0.95       # Similarity for duplicates
```
</details>

## Development

```bash
# Local development
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload

# View logs
docker-compose logs -f smartmemory-v2
```

**GPU support:** Use `requirements-gpu.txt` for CUDA acceleration (~12GB image vs ~600MB CPU)

## Support

**Issues:** [GitHub Issues](https://github.com/1818TusculumSt/smartmemoryapi/issues)
**License:** MIT
