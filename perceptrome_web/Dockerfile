# ── Stage 1: Build the Vite/React client ─────────────────────────────────────
FROM node:22-slim AS client-build

WORKDIR /build/client
COPY perceptrome_web/client/package.json perceptrome_web/client/package-lock.json ./
RUN npm ci
COPY perceptrome_web/client/ ./
RUN npm run build


# ── Stage 2: Python runtime ─────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends libpq5 && \
    rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash perceptrome

WORKDIR /app

# Install Python dependencies
COPY pyproject.toml ./
COPY perceptrome/ ./perceptrome/
COPY requirements/ ./requirements/
RUN pip install --no-cache-dir ".[web]"

# Copy server code
COPY perceptrome_web/ ./perceptrome_web/

# Copy built client from stage 1
COPY --from=client-build /build/client/dist ./perceptrome_web/client/dist

# Copy config and other project files needed at runtime
COPY config/ ./config/

RUN chown -R perceptrome:perceptrome /app
USER perceptrome

EXPOSE 8000

# Default: run the API server (migrations should be run separately)
CMD ["uvicorn", "perceptrome_web.server.app.main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "2", "--proxy-headers", "--forwarded-allow-ips", "*"]
