# SmartMemory API

Auto-extracting memory system with semantic search powered by Pinecone and LLM analysis.

## Features

- **Automatic Memory Extraction**: Uses LLM to identify user-specific facts, preferences, and context
- **Semantic Search**: Vector-based similarity search using embeddings
- **Deduplication**: Prevents duplicate memories using embedding similarity
- **Configurable Providers**: Switch between local and API-based embeddings
- **Auto-Pruning**: Maintains memory limit by removing oldest entries
- **OpenAPI Spec**: Ready for integration with Open WebUI

## Quick Start

### 1. Prerequisites

- Docker and Docker Compose
- Pinecone account and API key
- OpenAI API key (or compatible LLM API)

### 2. Setup
```bash
cd ~/openapi-servers/servers/smartmemory

# Copy example env file
cp .env.example .env

# Edit .env with your API keys
nano .env
