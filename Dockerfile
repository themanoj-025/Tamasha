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

# Build the package wheel
COPY . .
RUN pip install --no-cache-dir -e . && \
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

# Runtime system deps (OpenCV, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Expose ports
EXPOSE 8000 8501

# Default: run Streamlit
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
