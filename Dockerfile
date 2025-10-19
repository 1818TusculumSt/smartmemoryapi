# ============================================
# Stage 1: Builder - Compile dependencies
# ============================================
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and build wheels (includes all dependencies)
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# ============================================
# Stage 2: Runtime - Minimal final image
# ============================================
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install from pre-built wheels and cleanup in one layer
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && \
    rm -rf /wheels && \
    # Pre-download embedding model for local provider
    python -c "from sentence_transformers import SentenceTransformer; \
    model = SentenceTransformer('all-MiniLM-L6-v2'); \
    print(f'Model loaded: {model.get_sentence_embedding_dimension()}d')" || echo "Model download skipped" && \
    # Clean up all caches
    rm -rf /root/.cache /tmp/* /var/tmp/*

# Copy application code
COPY app.py memory_engine.py embeddings.py llm_client.py config.py ./

# Expose port
EXPOSE 8099

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8099/health', timeout=5)" || exit 1

# Run with multiple workers for better throughput
# Workers = (2 * CPU cores) + 1 is a good rule of thumb
# Using 2 workers as default - adjust based on your CPU
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8099", "--workers", "2"]
