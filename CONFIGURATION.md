# Configuration Guide

Complete configuration reference for SmartMemory 2.0 API.

## Embedding Providers

### Pinecone Inference (Recommended)

```env
EMBEDDING_PROVIDER=pinecone
EMBEDDING_MODEL=llama-text-embed-v2
```

**Available Pinecone models:**
- `llama-text-embed-v2` - Flexible dimensions: 384, 512, 768, 1024, or 2048 (auto-configured to match your index)
- `multilingual-e5-large` - Fixed 1024 dimensions

**Note:** `llama-text-embed-v2` automatically uses 384 dimensions to match most serverless indexes. Supports long documents (up to 2048 tokens) and multilingual text.

### Local (sentence-transformers)

```env
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

Downloads models to local cache on first run.

### API (OpenAI-compatible)

```env
EMBEDDING_PROVIDER=api
EMBEDDING_API_URL=https://api.openai.com/v1/embeddings
EMBEDDING_API_KEY=your-key
EMBEDDING_MODEL=text-embedding-3-small
```

## Memory Settings

```env
MAX_MEMORIES=1000          # Max memories before pruning
DEDUP_THRESHOLD=0.95       # Similarity threshold for duplicates (0-1)
MIN_CONFIDENCE=0.5         # Min confidence to store memory (0-1)
RELEVANCE_THRESHOLD=0.55   # Min relevance to return memory (0-1)
```

### Threshold Tuning Guide

**MIN_CONFIDENCE** (what to store):
- `0.4-0.5` - Captures more subtle patterns and implied preferences
- `0.6-0.8` - Only stores explicit statements and clear facts

**RELEVANCE_THRESHOLD** (what to recall):
- `0.5` - More memories in recall, better for sparse data
- `0.7` - Only highly relevant memories, better for dense data

**DEDUP_THRESHOLD** (duplicate detection):
- `0.95` - Standard (recommended)
- `0.90` - More aggressive deduplication
- `0.97` - Less aggressive, keeps more variations

### Memory Update Logic

- **0.95+ similarity**: Exact duplicate (skipped)
- **0.85-0.94 similarity**: Update (delete old, store new)
- **<0.85 similarity**: New memory (stored)

## Performance Settings

```env
MAX_RETRIES=3              # Retry attempts for failed requests
RETRY_DELAY=1.0            # Base delay for exponential backoff (seconds)
LLM_TIMEOUT=60             # LLM request timeout (seconds)
EMBEDDING_TIMEOUT=30       # Embedding request timeout (seconds)
CONNECTION_POOL_SIZE=100   # Max concurrent connections
KEEPALIVE_CONNECTIONS=20   # Reused connections
```

## Pagination Settings

```env
DEFAULT_PAGE_SIZE=50       # Default results per page
MAX_PAGE_SIZE=100          # Maximum allowed page size
```

## Batch Operations

```env
MAX_BATCH_SIZE=1000        # Maximum items in batch operations
```

## Memory Categories

Built-in categories for automatic organization:

| Category | Examples |
|----------|----------|
| `personal_information` | Name, age, location, identity |
| `food_preferences` | Likes, dislikes, dietary restrictions |
| `goals` | Aspirations, plans, objectives |
| `relationships` | Family, friends, colleagues |
| `behavior` | Habits, routines, patterns |
| `preferences` | General likes/dislikes |
| `hobbies` | Interests, activities |
| `work` | Career, projects, responsibilities |
| `health` | Medical info, fitness |
| `likes` | Things user enjoys |
| `dislikes` | Things user avoids |
| `skills` | Abilities, experience |

## Hybrid Search Configuration

Hybrid search combines semantic similarity with keyword matching.

**How it works:**
- Retrieves 3x candidates for re-ranking
- 15% boost per keyword match
- Enable via `use_hybrid: true` parameter in search requests

**Example:**
```json
{
  "query": "favorite programming languages",
  "user_id": "user_123",
  "use_hybrid": true,
  "limit": 5
}
```

## Environment Variables Reference

### Required

```env
PINECONE_API_KEY=          # Pinecone API key
PINECONE_INDEX_NAME=       # Pinecone index name
LLM_API_URL=              # LLM endpoint URL
LLM_API_KEY=              # LLM API key
LLM_MODEL=                # LLM model name
```

### Optional (with defaults)

```env
# Pinecone
PINECONE_ENVIRONMENT=us-east-1-aws

# Embeddings
EMBEDDING_PROVIDER=pinecone
EMBEDDING_MODEL=llama-text-embed-v2
EMBEDDING_API_URL=        # Only if EMBEDDING_PROVIDER=api
EMBEDDING_API_KEY=        # Only if EMBEDDING_PROVIDER=api

# Memory
MAX_MEMORIES=1000
DEDUP_THRESHOLD=0.95
MIN_CONFIDENCE=0.5
RELEVANCE_THRESHOLD=0.55

# Performance
MAX_RETRIES=3
RETRY_DELAY=1.0
LLM_TIMEOUT=60
EMBEDDING_TIMEOUT=30
CONNECTION_POOL_SIZE=100
KEEPALIVE_CONNECTIONS=20

# Pagination
DEFAULT_PAGE_SIZE=50
MAX_PAGE_SIZE=100

# Batch
MAX_BATCH_SIZE=1000
```

## Docker Build Options

### CPU-Only (Default, Recommended)

```bash
docker-compose up -d
```

**Image size:** ~600MB
**Best for:** Most deployments

### GPU Build (CUDA Acceleration)

Edit `docker-compose.yml` to use `requirements-gpu.txt`, then:

```bash
docker-compose build --build-arg REQUIREMENTS_FILE=requirements-gpu.txt
docker-compose up -d
```

**Image size:** ~12GB (includes CUDA libraries)
**Requires:** NVIDIA GPU with CUDA support
**Best for:** High-throughput local embeddings
