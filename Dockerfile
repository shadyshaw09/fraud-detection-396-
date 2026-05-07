# ─────────────────────────────────────────────────────────────────
# Stage 1 – Builder
# Install all heavy dependencies in a separate layer so the final
# image only copies what's needed.
# ─────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build tools (needed for some C-extension packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt

# ─────────────────────────────────────────────────────────────────
# Stage 2 – Runtime
# Lean final image – only Python runtime + installed packages
# ─────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="shadyshaw09"
LABEL project="Fraudulent Transaction Anomaly Detection"
LABEL course="INT332 – DevOps | Lovely Professional University"
LABEL version="1.0.0"

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY . .

# Create outputs directory so pipeline can write results
RUN mkdir -p outputs

# Default command – runs the full ML pipeline
CMD ["python", "main.py"]
