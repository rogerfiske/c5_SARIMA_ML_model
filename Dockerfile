# Stage 1: Builder - Install dependencies
FROM python:3.11-slim AS builder

# Install Poetry
RUN pip install --no-cache-dir poetry==2.0.1

WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Install dependencies in project virtualenv
RUN poetry config virtualenvs.in-project true && \
    poetry install --only main --no-root --no-interaction

# Stage 2: Runtime - Minimal image with installed deps
FROM python:3.11-slim

WORKDIR /app

# Copy virtualenv from builder
COPY --from=builder /app/.venv /app/.venv

# Copy source code and config files
COPY src/ ./src/
COPY pyproject.toml poetry.lock ./

# Add virtualenv to PATH
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src:$PYTHONPATH"

# Create volume mount points
RUN mkdir -p /app/data /app/configs /app/artifacts

# Run as non-root user for security
RUN useradd -m -u 1000 c5user && chown -R c5user:c5user /app
USER c5user

# Entry point is the CLI
ENTRYPOINT ["python", "-m", "c5_forecasting"]
CMD ["--help"]
