FROM python:3.11-slim-bookworm

WORKDIR /app

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml uv.lock* ./
COPY sanitizer/ sanitizer/
COPY app.py .

# Install dependencies
RUN uv sync --no-dev --frozen 2>/dev/null || uv sync --no-dev

# Create non-root user
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Expose Streamlit default port
EXPOSE 8501

# Health check (python-based — curl is not available in slim image)
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# Run the app — bind address configurable via STREAMLIT_ADDRESS (default: 127.0.0.1)
ENV STREAMLIT_ADDRESS=127.0.0.1
ENTRYPOINT ["sh", "-c", "uv run streamlit run app.py --server.port=8501 --server.address=${STREAMLIT_ADDRESS} --server.headless=true"]
