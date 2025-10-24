# Troubleshooting Guide

Common issues and solutions for SmartMemory 2.0 API.

## Service Won't Start

### Check the logs
```bash
docker-compose logs smartmemory-v2
```

### Common Issues

**Missing or invalid API keys**
```bash
# Verify your .env file has all required keys
grep -E "PINECONE_API_KEY|LLM_API_KEY" .env
```

**Port 8099 already in use**
```bash
# Check what's using the port
lsof -i :8099  # macOS/Linux
netstat -ano | findstr :8099  # Windows

# Change port in docker-compose.yml
ports:
  - "8100:8099"  # Use 8100 externally
```

**Docker daemon not running**
```bash
# Start Docker Desktop or Docker service
sudo systemctl start docker  # Linux
```

## No Memories Being Extracted

### 1. Verify LLM is responding
```bash
curl http://localhost:8099/status
```

### 2. Lower MIN_CONFIDENCE threshold
```env
# In .env file
MIN_CONFIDENCE=0.4  # Down from 0.5
```

### 3. Check extraction logs
```bash
docker-compose logs -f smartmemory-v2 | grep "Extracted"
```

### 4. Test with explicit statement
```bash
curl -X POST http://localhost:8099/add \
  -H "Content-Type: application/json" \
  -d '{
    "user_message": "My name is John and I love pizza",
    "user_id": "test_user"
  }'
```

## Memories Not Appearing in Recall

### 1. Lower RELEVANCE_THRESHOLD
```env
# In .env file
RELEVANCE_THRESHOLD=0.5  # Down from 0.55
```

### 2. Verify semantic relevance
Make sure your query is semantically related to stored memories:
- ❌ Query: "music preferences" | Memory: "loves Italian food"
- ✅ Query: "music preferences" | Memory: "loves rock music"

### 3. Check user_id matches
```bash
# Add memory
curl -X POST http://localhost:8099/add \
  -d '{"user_message": "...", "user_id": "user123"}'

# Search - must use same user_id
curl -X POST http://localhost:8099/search \
  -d '{"query": "...", "user_id": "user123"}'
```

### 4. Verify memories were stored
```bash
curl -X POST http://localhost:8099/recent \
  -H "Content-Type: application/json" \
  -d '{"limit": 10, "user_id": "test_user"}'
```

## Memory Updates Not Working

### Check similarity thresholds
```env
DEDUP_THRESHOLD=0.95  # Should be 0.95 for smart updates
```

### Look for update logs
```bash
docker-compose logs smartmemory-v2 | grep "Similar memory found"
```

### Verify memory IDs
Update logs should show:
```
Similar memory found (similarity=0.89): mem_xxx
Updating existing memory: mem_xxx
```

## Hybrid Search Not Working

### 1. Enable hybrid search in request
```json
{
  "query": "programming languages",
  "user_id": "user_123",
  "use_hybrid": true,  // Must be explicitly enabled
  "limit": 5
}
```

### 2. Monitor keyword extraction
```bash
docker-compose logs -f smartmemory-v2 | grep "keyword"
```

### 3. Check query terms
Hybrid search works best with specific terms:
- ✅ "python programming languages"
- ❌ "stuff user likes"

## Performance Issues

### Check connection pool settings
```env
CONNECTION_POOL_SIZE=100
KEEPALIVE_CONNECTIONS=20
```

### Monitor HTTP/2 usage
```bash
docker-compose logs smartmemory-v2 | grep "HTTP/2"
```

### Adjust timeouts
```env
LLM_TIMEOUT=60           # Increase if LLM is slow
EMBEDDING_TIMEOUT=30     # Increase if embeddings timeout
```

### Check resource usage
```bash
docker stats smartmemory-v2
```

## Embedding Errors

### Pinecone Provider

**Invalid API key:**
```bash
# Verify your key
curl https://api.pinecone.io/indexes \
  -H "Api-Key: YOUR_KEY"
```

**Model not found:**
```env
# Use exact model names
EMBEDDING_MODEL=llama-text-embed-v2
# OR
EMBEDDING_MODEL=multilingual-e5-large
```

**Dimension mismatch:**
- `llama-text-embed-v2` auto-configures to 384 dims
- Verify your Pinecone index dimension matches

### Local Provider

**Model download fails:**
```bash
# Check disk space
df -h

# Clear cache and retry
rm -rf ~/.cache/huggingface
docker-compose restart smartmemory-v2
```

**Out of memory:**
```yaml
# In docker-compose.yml, increase memory
services:
  smartmemory-v2:
    deploy:
      resources:
        limits:
          memory: 4G  # Up from 2G
```

### API Provider

**Connection errors:**
```bash
# Test endpoint directly
curl https://api.openai.com/v1/embeddings \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": "test", "model": "text-embedding-3-small"}'
```

## Database Issues

### Pinecone connection timeout
```env
PINECONE_ENVIRONMENT=us-east-1-aws  # Verify correct region
```

### Index not found
```bash
# List your indexes
curl https://api.pinecone.io/indexes \
  -H "Api-Key: YOUR_KEY"
```

### Quota exceeded
Check your Pinecone plan limits and upgrade if needed.

## API Response Errors

### 422 Validation Error
- Check request body matches API schema
- View schema at `http://localhost:8099/docs`

### 500 Internal Server Error
```bash
# Check full error in logs
docker-compose logs smartmemory-v2 | tail -50
```

### Timeout Errors
```env
# Increase timeouts
LLM_TIMEOUT=120
EMBEDDING_TIMEOUT=60
```

## Development Issues

### Changes not reflected
```bash
# Rebuild container
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Local development not working
```bash
# Verify Python version (3.11+)
python --version

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

## Getting Help

If issues persist:

1. **Check logs:** `docker-compose logs smartmemory-v2 > logs.txt`
2. **Review status:** Visit `http://localhost:8099/status`
3. **Test health:** `curl http://localhost:8099/health`
4. **Open issue:** [GitHub Issues](https://github.com/1818TusculumSt/smartmemoryapi/issues)

Include:
- Error messages from logs
- Your `.env` configuration (redact API keys)
- Steps to reproduce
- Docker/Python version
