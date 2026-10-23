FROM python:3.14-slim

# Prevent Python from writing bytecode and buffer-free logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install uv from its official container image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Make the virtual environment available
ENV PATH="/app/.venv/bin:$PATH"

# Install dependencies first to improve Docker layer caching
COPY pyproject.toml uv.lock README.md ./

RUN uv sync --locked --no-dev --no-install-project

# Copy application source
COPY src/ ./src/

# Install the project itself
RUN uv sync --locked --no-dev



EXPOSE 8000

CMD ["uvicorn", "ai_document_intelligence.main:app", "--host", "0.0.0.0", "--port", "8000"]