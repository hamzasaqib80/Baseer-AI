# ==========================================
# MULTI-STAGE ENTERPRISE DOCKERFILE
# Stage 1: Build & Requirements
# Stage 2: Optimized Lightweight Runtime
# ==========================================

# STAGE 1: Dependency Preparation
FROM python:3.10-slim as builder

WORKDIR /build
COPY requirements.txt .

# Install dependencies to a local folder to copy back easily
RUN pip install --no-cache-dir --user -r requirements.txt

# STAGE 2: Final Production Runner
FROM python:3.10-slim as runner

# Setup non-root user for security (DevSecOps requirement)
RUN useradd -m appuser
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH

# Copy Project Structure
COPY src/ ./src
COPY configs/ ./configs
COPY models/ ./models
# Ensure directory exists for inference artifacts
RUN mkdir -p monitoring && chown -R appuser:appuser /app

USER appuser

# Expose FastAPI Port
EXPOSE 8000

# Metadata
LABEL maintainer="Principal AI Architect"
LABEL version="1.0.0"
LABEL description="CIFAR-10 Enterprise Inference microservice"

# Run the API via Uvicorn
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
