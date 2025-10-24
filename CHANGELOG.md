# Changelog

All notable changes to SmartMemory 2.0 API will be documented in this file.

## [2.0.0] - 2025-01

### Added
- Smart memory updates (memories evolve over time based on 0.85-0.94 similarity)
- Hybrid search (semantic + keyword matching with 15% boost per match)
- Memory consolidation (merge fragmented memories)
- Auto-recall (proactive context-based retrieval)
- Temporal context awareness (current date in extraction)
- HTTP/2 and connection pooling
- Enhanced OpenAPI descriptions for LLM integration
- Recent memories retrieval endpoint
- Performance optimizations throughout

### Changed
- Category-based organization (from tag-based)
- Session support via `run_id`
- Lower default thresholds for better capture
- Enhanced extraction prompts

### Performance
- HTTP/2 enabled for all API calls
- Connection pooling: 20 keepalive, 100 max connections
- Async operations throughout the stack
- Automatic retry logic with exponential backoff
- Memory extraction: ~200-500ms
- Search operations: ~100-300ms
- Consolidation: ~2-5s per group

## [1.0.0] - 2024

### Initial Release
- Basic memory extraction and storage
- Simple semantic search
- Pinecone vector database integration
- Tag-based organization
- Simple `/extract` endpoint
