# Multi-stage build for Power Systems Autoencoder using Chainguard for maximum security
FROM cgr.dev/chainguard/python:3.9 as builder

# Install build dependencies with security hardening
RUN apk add --no-cache \
    gcc \
    g++ \
    && rm -rf /var/cache/apk/*

# Create and activate virtual environment with security settings
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Upgrade pip and install Python dependencies securely
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt \
    && find /opt/venv -depth -type d -name "*__pycache__*" -exec rm -rv {} + \
    && find /opt/venv -type f -name "*.pyc" -delete

# Production stage using Chainguard distroless for maximum security
FROM cgr.dev/chainguard/python:3.9-runtime as runtime

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copy application code with security
COPY --from=builder /app/src /app/src
COPY --from=builder /app/deployment/services/power-autoencoder /app/service

# Set working directory
WORKDIR /app

# Create non-root user for additional security
RUN addgroup -g 1001 -S powerids && \
    adduser -u 1001 -S powerids -G powerids && \
    chown -R powerids:powerids /app
USER powerids

# Expose port
EXPOSE 8083

# Health check using Python
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8083/health')" || exit 1

# Run the power systems autoencoder service
CMD ["python", "-u", "service/power_main.py"]
