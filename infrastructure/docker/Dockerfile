FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# LABEL maintainer="NotebookLM Clone"
LABEL version="1.0-gpu"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    FORCE_CUDA=1

# Set working directory
WORKDIR /app

# Install Python 3.9 and system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    build-essential \
    curl \
    git \
    libpq-dev \
    gcc \
    g++ \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set python3.9 as default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

# Copy requirements first for better caching
COPY requirements.txt /app/

# Install llama-cpp-python with CUDA support first
ENV CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=all"
RUN pip install --no-cache-dir --force-reinstall --no-binary llama-cpp-python 'llama-cpp-python>=0.3.16'

# Install remaining requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app/
# COPY vllm /app/vllm

# # Install vllm from source
# RUN pip install -e /app/vllm

# Create necessary directories
RUN mkdir -p /app/data/documents /app/data/embeddings /app/logs

# Create non-root user
# RUN useradd --create-home --shell /bin/bash --uid 1001 appuser && \
#     chown -R appuser:appuser /app
# USER appuser

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Start application directly (no run.py needed)
CMD ["streamlit", "run", "src/frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--server.enableCORS=false"]
