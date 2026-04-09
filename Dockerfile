# ── Base Image ────────────────────────────────────────────────────────────────
# We start FROM an official Python image on Docker Hub.
# python:3.11-slim = Python 3.11 on minimal Linux (Debian).
# "slim" means smaller image — fewer pre-installed tools, faster uploads.
# We use 3.11 not 3.13 because HuggingFace Spaces has best support for 3.11.
FROM python:3.11-slim

# ── Metadata ──────────────────────────────────────────────────────────────────
# Labels are optional documentation attached to the image.
LABEL maintainer="FinanceGPT"
LABEL description="AI-powered financial data analysis — LangGraph + Groq"
LABEL version="1.0"

# ── System Dependencies ───────────────────────────────────────────────────────
# Some Python packages need C libraries to compile (like numpy, scipy).
# We install them at the OS level first.
# --no-install-recommends = don't install optional extras (keeps image lean)
# rm -rf /var/lib/apt/lists/* = delete apt cache after install (smaller image)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working Directory ─────────────────────────────────────────────────────────
# All subsequent commands run from /app inside the container.
# This is the "home folder" for our app inside the container.
WORKDIR /app

# ── Install Python Dependencies ───────────────────────────────────────────────
# IMPORTANT: We copy requirements.txt BEFORE copying the rest of the code.
#
# Why? Docker builds in layers and caches each layer. If requirements.txt
# hasn't changed, Docker reuses the cached pip install layer — much faster
# rebuilds when you only change your Python code.
#
# This pattern (copy requirements first, then code) is called
# "layer caching optimization" — standard Docker best practice.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy Application Code ─────────────────────────────────────────────────────
# Now copy everything else. .dockerignore controls what gets excluded.
# We do this AFTER pip install so code changes don't bust the dependency cache.
COPY . .

# ── Create Data Directory ─────────────────────────────────────────────────────
# The app writes SQLite databases at runtime.
# We create the /app/data directory and make it writable.
RUN mkdir -p /app/data && chmod 777 /app/data

# ── Environment Variables ─────────────────────────────────────────────────────
# These are DEFAULT values baked into the image — safe, non-secret config.
# Secret values (API keys) are injected at runtime via HuggingFace Secrets.
# NEVER put real API keys in a Dockerfile.
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DB_PATH=/app/data/finance.db
ENV CHECKPOINTS_DB=/app/data/checkpoints.db
ENV MAX_FILE_SIZE_MB=10
ENV EXECUTION_TIMEOUT_SECONDS=5
ENV CACHE_ENABLED=true

# ── Port ──────────────────────────────────────────────────────────────────────
# Tell Docker that this container listens on port 7860.
# Gradio uses 7860 by default.
# HuggingFace Spaces REQUIRES port 7860 — this is mandatory.
EXPOSE 7860

# ── Health Check ──────────────────────────────────────────────────────────────
# Docker periodically pings the app to check if it's alive.
# If the app crashes silently, Docker detects it via this check.
# --interval=30s  → check every 30 seconds
# --timeout=10s   → wait max 10 seconds for response
# --start-period=60s → give app 60 seconds to start before checking
# --retries=3     → mark unhealthy only after 3 consecutive failures
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# ── Start Command ─────────────────────────────────────────────────────────────
# This command runs when the container starts.
# HuggingFace Spaces runs this automatically when your Space starts.
CMD ["python", "app.py"]