# Build stage
FROM python:3.12.1-slim-bookworm AS builder

# Set working directory
WORKDIR /usr/src/app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
RUN pip install --upgrade pip setuptools wheel

# Copy project files
COPY pyproject.toml ./
COPY src ./src

# Install project dependencies
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /usr/src/app/wheels -e .

# Final stage
FROM python:3.12.1-slim-bookworm

# Create a non-root user
RUN useradd -m appuser

# Set working directory
WORKDIR /usr/src/app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Copy wheels and install dependencies
COPY --from=builder /usr/src/app/wheels /wheels
COPY --from=builder /usr/src/app/src ./src
COPY pyproject.toml ./
RUN pip install --no-cache /wheels/*

# Install the project
RUN pip install -e .

# Copy entrypoint script
COPY src/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Create and set ownership for the application code directory
# (Ensures appuser owns source code mounted later if using bind mount for dev)
# Also create and set ownership for the persistent data directory *before* volume mount
RUN chown -R appuser:appuser /usr/src/app \
 && mkdir -p /home/appuser/data \
 && chown appuser:appuser /home/appuser/data

# Set the entrypoint (runs as root by default here)
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Switch to non-root user for the CMD
USER appuser

# Default command (will be passed to entrypoint and run as appuser)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]