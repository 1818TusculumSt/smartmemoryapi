<img width="1024" height="576" alt="Adobe Express - file" src="https://github.com/user-attachments/assets/0d591de8-f907-4313-8233-13c89b59a2f3" />

# SmartMemory 2.0 API

Next-generation memory system with smart extraction, semantic search, and auto-recall powered by Pinecone.

## Quick Start

```bash
git clone https://github.com/1818TusculumSt/smartmemoryapi.git
cd smartmemoryapi
cp .env.example .env  # Edit with your keys
docker-compose up -d
```

### Required `.env`

```env
PINECONE_API_KEY=your-key
PINECONE_INDEX_NAME=smartmemory-v2
LLM_API_URL=https://api.openai.com/v1/chat/completions
LLM_API_KEY=your-key
LLM_MODEL=gpt-4o-mini
EMBEDDING_PROVIDER=pinecone
EMBEDDING_MODEL=llama-text-embed-v2
```

**API:** `http://localhost:8099` | **Docs:** `http://localhost:8099/docs`

## What's New in v2.0

### Currently Implemented ✅

- **Auto-categorization**: 10 types (achievement, frustration, idea, fact, event, conversation, relationship, technical, personal, misc)
- **Importance scoring**: 1-10 scale, auto-assigned by LLM
- **Sentiment detection**: positive, negative, neutral, mixed
- **Tags**: Array-based organization for flexible categorization
- **Pinning & archiving**: Flag critical memories, soft-delete others
- **Smart updates**: Memories evolve (0.85-0.94 similarity) instead of duplicating
- **Memory consolidation**: Merge related memory fragments
- **Hybrid search**: Semantic + keyword matching
- **Recent memories**: Time-sorted retrieval

**Backward compatible** - all existing memories still work.

### Future Roadmap 🚀

See [smartmemory-v2-omnibus-spec.md](smartmemory-v2-omnibus-spec.md) for the complete v2.0 roadmap including:

- **Advanced time-based retrieval** - Timeline views, gaps analysis, period summaries
- **Enhanced tag management** - Tag statistics, bulk tagging, tag-based search
- **Relationship queries** - Semantic similarity, explicit memory linking
- **Analytics & insights** - Comprehensive stats, pattern detection, trend analysis
- **Advanced search** - Multi-dimensional filtering (tags, dates, importance, sentiment, entities)
- **Quality curation** - Duplicate detection, importance management, bulk operations
- **Smart summaries** - AI-generated recaps of time periods
- **Export capabilities** - JSON, Markdown, CSV formats

The omnibus spec provides detailed implementation plans, API designs, and MCP tool definitions for all planned features.

## Features

- Smart memory extraction from conversation
- Hybrid search (semantic + keyword)
- Auto-recall for proactive context
- Memory consolidation
- Flexible embeddings (local/API/Pinecone)
- HTTP/2, connection pooling, async

## Open WebUI Integration

**Via MCPO (recommended):**
```bash
uvx mcpo --port 8100 --api-key "your-secret" -- http://localhost:8099
```
Settings → External Tools → Add `http://localhost:8100`

**Direct:** Workspace → Functions → Import `http://localhost:8099/openapi.json`

## API Endpoints

| Endpoint | Purpose |
|----------|---------|
| `POST /add` | Extract and store memories |
| `POST /search` | Semantic search with filters |
| `POST /relevant` | Get memories above threshold |
| `POST /recent` | Recent memories by time |
| `POST /auto-recall` | Proactive retrieval |
| `POST /consolidate` | Merge fragments |
| `DELETE /memory/{id}` | Delete memory |
| `POST /batch/delete` | Batch delete |

Full docs at `http://localhost:8099/docs`

## Configuration

<details>
<summary>Memory Thresholds</summary>

```env
MIN_CONFIDENCE=0.5         # Lower = more captured
RELEVANCE_THRESHOLD=0.55   # Lower = more recalled
DEDUP_THRESHOLD=0.95       # Duplicate detection
```

**Memory updates:**
- ≥0.95 similarity: Skip (duplicate)
- 0.85-0.94: Update existing
- <0.85: Store new

</details>

<details>
<summary>Embedding Providers</summary>

**Pinecone (recommended):**
```env
EMBEDDING_PROVIDER=pinecone
EMBEDDING_MODEL=llama-text-embed-v2
```

**Local:**
```env
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

**API:**
```env
EMBEDDING_PROVIDER=api
EMBEDDING_API_URL=https://api.openai.com/v1/embeddings
EMBEDDING_API_KEY=your-key
```

</details>

## Troubleshooting

**Service won't start:** `docker-compose logs smartmemory-v2`

**No memories extracted:** Lower `MIN_CONFIDENCE=0.4` in `.env`

**Memories not recalled:** Lower `RELEVANCE_THRESHOLD=0.5`, ensure matching `user_id`

**Embedding errors:** Check API keys, model names

**Full troubleshooting:** See logs and status at `http://localhost:8099/status`

## Development

```bash
# Local
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload

# Docker rebuild
docker-compose down && docker-compose build --no-cache && docker-compose up -d
```

## Support

**Issues:** [GitHub Issues](https://github.com/1818TusculumSt/smartmemoryapi/issues)
**License:** MIT
