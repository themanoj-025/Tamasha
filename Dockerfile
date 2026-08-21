# =====================================================================
#  BUILDER STAGE — install system deps + build wheel
# =====================================================================
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build system deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first (Docker layer caching)
COPY requirements.txt setup.py ./
COPY src/ ./src/

# Install runtime deps into a venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Build the package wheel — regular (non-editable) install so the package
# lands inside /opt/venv. An editable install (-e) writes a .pth pointing at
# /build/src, which is NOT shipped to the runtime stage (import tamasha fails).
COPY . .
RUN pip install --no-cache-dir . && \
    python -c "import nltk; nltk.download('vader_lexicon', quiet=True)"

# =====================================================================
#  RUNTIME STAGE — minimal image, only runtime deps
# =====================================================================
FROM python:3.11-slim

WORKDIR /app

# Copy virtualenv with all deps from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code (no build tools, no .git, no data/raw)
COPY --from=builder /build/app ./app
COPY --from=builder /build/api ./api
COPY --from=builder /build/src ./src
COPY --from=builder /build/packages.txt .
COPY --from=builder /build/render.yaml .
COPY --from=builder /build/setup.py .

# Runtime system deps (OpenCV, etc.) — libgl1 replaces the old
# libgl1-mesa-glx package name on Debian bookworm+ (current slim base).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN addgroup --system app && adduser --system --ingroup app app \
    && chown -R app:app /app

USER app

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default: run Streamlit
STOPSIGNAL SIGTERM
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
