# Hướng dẫn Setup NotebookLM trên GPU Server từ đầu

> **Mục đích:** Tài liệu này giúp bạn setup lại toàn bộ hệ thống từ đầu nếu server bị restart và mất dữ liệu.

## 📋 Thông tin Server

- **GPU:** NVIDIA H200 (139.7GB VRAM)
- **CUDA Version:** 12.4 (Driver), 12.1 (Toolkit)
- **OS:** Ubuntu 24.04
- **Python:** 3.12.3

---

## 🔧 PHẦN 1: Chỉnh sửa Code để hỗ trợ GPU

### 1.1. Chỉnh sửa Dockerfile

**File:** `infrastructure/docker/Dockerfile`

```dockerfile
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

# Create necessary directories
RUN mkdir -p /app/data/documents /app/data/embeddings /app/logs

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Start application
CMD ["streamlit", "run", "src/frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--server.enableCORS=false"]
```

**Thay đổi chính:**
- Base image: `python:3.9-slim` → `nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04`
- Thêm CUDA environment variables
- Cài Python 3.9 manually
- Cài llama-cpp-python với CMAKE_ARGS cho CUDA

---

### 1.2. Chỉnh sửa docker-compose.yml

**File:** `infrastructure/docker/docker-compose.yml`

Thêm vào service `notebooklm-app`:

```yaml
  notebooklm-app:
    build:
      context: ../..
      dockerfile: infrastructure/docker/Dockerfile
      args:
        - PYTHON_VERSION=3.9
    container_name: notebooklm-app
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
      - ENVIRONMENT=${ENVIRONMENT:-development}
      - DEBUG=${DEBUG:-true}
      # ... các env khác
```

**Thay đổi chính:**
- Thêm `deploy.resources.reservations.devices` để enable GPU
- Thêm NVIDIA environment variables

---

### 1.3. Chỉnh sửa requirements.txt

**File:** `requirements.txt`

```txt
# Core dependencies
streamlit==1.39.0
fastapi>=0.115.0
uvicorn==0.24.0

# LLM & AI
--extra-index-url https://download.pytorch.org/whl/cu121
torch==2.3.1+cu121
torchaudio==2.3.1+cu121
torchvision==0.18.1+cu121
sentence-transformers
transformers

# RAG & Vector DB - CUDA support configured in Dockerfile
llama-cpp-python>=0.3.16
langchain>=0.1.0
langchain-community>=0.0.38
langchain-qdrant>=0.1.0
qdrant-client
tenacity>=8.2.0

# Reranking
rank-bm25

# Document processing
pymupdf==1.23.8
pdfplumber==0.10.3
paddlepaddle-gpu==2.6.1  # GPU version
paddleocr==2.7.3
Pillow>=10.3.0
python-docx>=1.0.0
docx2txt>=0.8

# Database
psycopg2-binary==2.9.7
sqlalchemy==2.0.23
alembic==1.13.1

# Utils
pydantic>=2.11.7,<3.0
python-multipart==0.0.6
python-dotenv==1.0.0
requests>=2.32.2
beautifulsoup4==4.12.2
numpy
pandas==2.1.4
uuid==1.30
tqdm
cachetools>=5.3.0
sortedcontainers>=2.4.0
pydantic-settings
chardet
aiofiles
httpx

# Development
jupyter==1.0.0
ipykernel>=6.29.3,<7
pytest==7.4.3
black==23.11.0
```

**Thay đổi chính:**
- Thêm PyTorch với CUDA 12.1
- Thay `paddlepaddle` → `paddlepaddle-gpu`
- Thêm extra index URL cho PyTorch

---

### 1.4. Chỉnh sửa config/settings.py

**File:** `config/settings.py`

Tìm và sửa phần LLM configuration:

```python
# llama.cpp specific settings
LLAMACPP_N_GPU_LAYERS: int = -1  # -1 = offload all layers to GPU (0 = CPU only)
LLAMACPP_N_BATCH: int = 512      # Increased for better GPU performance
LLAMACPP_N_THREADS: int = 4      # CPU threads for non-GPU operations
LLAMACPP_VERBOSE: bool = False
```

**Thay đổi chính:**
- `LLAMACPP_N_GPU_LAYERS`: 0 → -1 (enable GPU)
- `LLAMACPP_N_BATCH`: 64 → 512 (tăng batch size cho GPU)

---

### 1.5. Tạo file .env

**File:** `.env`

```bash
# Environment
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=notebooklm
POSTGRES_USER=postgres
POSTGRES_PASSWORD=password123

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=

# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_GRPC_PORT=6334
QDRANT_COLLECTION_NAME=document_embeddings

# LLM Configuration - GPU Enabled
LLM_MODEL_PATH=/root/Notebooklm/infrastructure/docker/models/Qwen3-8B-Q4_K_M.gguf
LLAMACPP_N_GPU_LAYERS=-1
LLAMACPP_N_BATCH=512
LLAMACPP_N_THREADS=4
LLM_CONTEXT_LENGTH=2048

# Streamlit
STREAMLIT_PORT=8501
```

---

## 🚀 PHẦN 2: Cài đặt hệ thống

### 2.1. Cài đặt System Dependencies

```bash
# Update system
sudo apt-get update

# Cài Python và build tools
sudo apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    build-essential \
    gcc \
    g++ \
    make \
    cmake \
    wget \
    curl \
    git
```

---

### 2.2. Cài đặt NVIDIA Container Toolkit (cho Docker)

```bash
# Add NVIDIA GPG key
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor --batch --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# Add repository
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Update và install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker
sudo nvidia-ctk runtime configure --runtime=docker
```

---

### 2.3. Cài đặt CUDA Toolkit (cho native run)

```bash
# Cài CUDA Toolkit từ Ubuntu repos
sudo apt-get install -y nvidia-cuda-toolkit

# Verify installation
nvcc --version
# Expected: release 12.0, V12.0.140

# Check GPU
nvidia-smi
# Expected: NVIDIA H200
```

---

### 2.4. Setup Python Virtual Environment

```bash
# Navigate to project
cd /root/Notebooklm

# Create virtual environment
python3 -m venv venv

# Activate
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

---

### 2.5. Cài đặt PyTorch với CUDA

```bash
# Make sure venv is activated
source venv/bin/activate

# Install PyTorch with CUDA 12.1
pip install torch==2.3.1+cu121 \
    torchaudio==2.3.1+cu121 \
    torchvision==0.18.1+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Verify CUDA availability
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
# Expected: CUDA available: True
```

---

### 2.6. Cài đặt llama-cpp-python với CUDA

```bash
# Make sure venv is activated
source venv/bin/activate

# Set CMAKE arguments
export CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=all"

# Install llama-cpp-python
pip install --force-reinstall --no-binary llama-cpp-python 'llama-cpp-python>=0.3.16'

# This will take 5-10 minutes to compile
```

---

### 2.7. Cài đặt các packages còn lại

```bash
# Make sure venv is activated
source venv/bin/activate

# Install ML/AI packages
pip install sentence-transformers transformers
pip install langchain>=0.1.0 langchain-community>=0.0.38 langchain-qdrant>=0.1.0
pip install qdrant-client tenacity>=8.2.0 rank-bm25

# Install document processing
pip install pymupdf==1.23.8 pdfplumber==0.10.3
pip install python-docx docx2txt
pip install paddlepaddle-gpu==2.6.1 paddleocr==2.7.3

# Install database packages
pip install libpq-dev  # Nếu chưa có
pip install psycopg2-binary==2.9.7 sqlalchemy==2.0.23

# Install web framework
pip install streamlit==1.39.0 fastapi uvicorn

# Install utilities
pip install 'pydantic>=2.11.7,<3.0' python-dotenv==1.0.0 pydantic-settings
pip install requests beautifulsoup4 pandas numpy tqdm
```

---

### 2.8. Setup Databases

#### Redis

```bash
# Install Redis
sudo apt-get install -y redis-server

# Start Redis
redis-server --daemonize yes

# Test
redis-cli ping
# Expected: PONG
```

#### Qdrant

```bash
cd /root/Notebooklm

# Download Qdrant binary
wget https://github.com/qdrant/qdrant/releases/download/v1.15.0/qdrant-x86_64-unknown-linux-gnu.tar.gz

# Extract
tar -xzf qdrant-x86_64-unknown-linux-gnu.tar.gz

# Make executable
chmod +x qdrant

# Run in background
nohup ./qdrant > qdrant.log 2>&1 &

# Wait and test
sleep 5
curl http://localhost:6333
# Expected: {"title":"qdrant - vector search engine","version":"1.15.0",...}
```

#### PostgreSQL (Optional)

```bash
# Install PostgreSQL
sudo apt-get install -y postgresql postgresql-contrib

# Start (if systemd available)
sudo systemctl start postgresql

# Or manually
sudo -u postgres /usr/lib/postgresql/16/bin/pg_ctl -D /var/lib/postgresql/16/main start
```

---

### 2.9. Download Model

```bash
cd /root/Notebooklm

# Tạo thư mục models
mkdir -p infrastructure/docker/models

# Download model Qwen3-8B (khoảng 4.6GB)
cd infrastructure/docker/models
wget https://huggingface.co/bartowski/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf

# Verify download
ls -lh Qwen3-8B-Q4_K_M.gguf
```

---

## ▶️ PHẦN 3: Chạy ứng dụng

### 3.1. Verify tất cả đã cài đúng

```bash
cd /root/Notebooklm
source venv/bin/activate

# Check GPU
nvidia-smi

# Check CUDA trong Python
python3 -c "
import torch
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

# Check llama-cpp-python installed
python3 -c "from llama_cpp import Llama; print('llama-cpp-python OK')"

# Check services
redis-cli ping
curl http://localhost:6333

# Check model exists
ls -lh infrastructure/docker/models/Qwen3-8B-Q4_K_M.gguf
```

---

### 3.2. Chạy Streamlit Application

```bash
cd /root/Notebooklm
source venv/bin/activate

# Chạy app
streamlit run src/frontend/app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    2>&1 | tee streamlit.log &

# Đợi app khởi động (10-15 giây)
sleep 15

# Kiểm tra app đã chạy
curl -s http://localhost:8501 | head -10
```

**Truy cập:** `http://localhost:8501`

---

## 📝 PHẦN 4: Scripts tự động

### 4.1. Script start_all.sh

```bash
cat > /root/Notebooklm/start_all.sh << 'EOF'
#!/bin/bash
set -e

echo "Starting NotebookLM with GPU Support"
cd /root/Notebooklm

# Start Redis
echo "1. Starting Redis..."
redis-server --daemonize yes
redis-cli ping

# Start Qdrant
echo "2. Starting Qdrant..."
if ! pgrep -f qdrant > /dev/null; then
    nohup ./qdrant > qdrant.log 2>&1 &
    sleep 5
fi
curl -s http://localhost:6333 | head -1

# Start Streamlit
echo "3. Starting Streamlit..."
source venv/bin/activate
streamlit run src/frontend/app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    2>&1 | tee streamlit.log &

sleep 15
echo "✅ Application running at http://localhost:8501"
EOF

chmod +x /root/Notebooklm/start_all.sh
```

### 4.2. Script stop_all.sh

```bash
cat > /root/Notebooklm/stop_all.sh << 'EOF'
#!/bin/bash

echo "Stopping NotebookLM Services"
cd /root/Notebooklm

pkill -f streamlit
pkill -f qdrant
redis-cli shutdown 2>/dev/null

echo "All services stopped."
EOF

chmod +x /root/Notebooklm/stop_all.sh
```

### 4.3. Script check_status.sh

```bash
cat > /root/Notebooklm/check_status.sh << 'EOF'
#!/bin/bash

echo "NotebookLM Services Status"
echo "=========================="

echo "GPU:"
nvidia-smi | grep "H200"

echo ""
echo "Redis:"
redis-cli ping 2>/dev/null || echo "❌ Not running"

echo ""
echo "Qdrant:"
curl -s http://localhost:6333 > /dev/null 2>&1 && echo "✅ Running" || echo "❌ Not running"

echo ""
echo "Streamlit:"
curl -s http://localhost:8501 > /dev/null 2>&1 && echo "✅ Running at http://localhost:8501" || echo "❌ Not running"
EOF

chmod +x /root/Notebooklm/check_status.sh
```

---

## 🔄 PHẦN 5: Quy trình khi server restart

### Checklist khi server mới restart:

```bash
# 1. Check GPU
nvidia-smi

# 2. Navigate to project
cd /root/Notebooklm

# 3. Start services
./start_all.sh

# 4. Check status
./check_status.sh

# 5. Monitor logs
tail -f streamlit.log
```

---

## 🐛 Troubleshooting

### Lỗi: CUDA not available

```bash
# Reinstall PyTorch
source venv/bin/activate
pip uninstall torch torchvision torchaudio -y
pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121
```

### Lỗi: llama-cpp-python không có CUDA

```bash
# Reinstall với CUDA
source venv/bin/activate
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=all" \
    pip install --force-reinstall --no-binary llama-cpp-python 'llama-cpp-python>=0.3.16'
```

### Lỗi: Port already in use

```bash
# Kill process on port
lsof -i :8501
kill -9 <PID>
```

### Lỗi: Qdrant không start

```bash
# Re-download Qdrant
cd /root/Notebooklm
rm -f qdrant qdrant-x86_64-unknown-linux-gnu.tar.gz
wget https://github.com/qdrant/qdrant/releases/download/v1.15.0/qdrant-x86_64-unknown-linux-gnu.tar.gz
tar -xzf qdrant-x86_64-unknown-linux-gnu.tar.gz
chmod +x qdrant
nohup ./qdrant > qdrant.log 2>&1 &
```

---

## 📊 Verification Commands

```bash
# GPU status
nvidia-smi

# CUDA version
nvcc --version

# Python packages
source venv/bin/activate
pip list | grep -E "torch|llama|streamlit|langchain"

# CUDA trong Python
python3 -c "import torch; print(torch.cuda.is_available())"

# Services
redis-cli ping
curl http://localhost:6333
curl http://localhost:8501

# Processes
ps aux | grep -E "streamlit|qdrant|redis"
```

---

## 📁 File Structure

```
/root/Notebooklm/
├── src/
│   ├── frontend/
│   │   └── app.py              # Main Streamlit app
│   ├── serving/
│   │   └── llm_service.py      # LLM service with GPU
│   └── rag/
│       └── ...                 # RAG components
├── config/
│   └── settings.py             # Config with GPU settings
├── infrastructure/
│   └── docker/
│       ├── Dockerfile          # GPU-enabled Dockerfile
│       ├── docker-compose.yml  # With GPU config
│       └── models/
│           └── Qwen3-8B-Q4_K_M.gguf  # LLM model
├── venv/                       # Python virtual environment
├── .env                        # Environment variables
├── requirements.txt            # Python dependencies (GPU)
├── qdrant                      # Qdrant binary
├── start_all.sh               # Start script
├── stop_all.sh                # Stop script
├── check_status.sh            # Status check script
└── streamlit.log              # Application logs
```

---

## ✅ Final Checklist

Sau khi setup xong, verify:

- [ ] `nvidia-smi` hiển thị GPU H200
- [ ] `nvcc --version` hiển thị CUDA 12.0+
- [ ] PyTorch: `torch.cuda.is_available()` = True
- [ ] Redis: `redis-cli ping` = PONG
- [ ] Qdrant: `curl localhost:6333` có response
- [ ] Model: File GGUF tồn tại tại đường dẫn đúng
- [ ] Streamlit: `curl localhost:8501` có HTML response
- [ ] GPU được sử dụng khi chạy: `nvidia-smi` hiển thị VRAM usage

---

## 🎯 Tóm tắt các bước chính

1. **Chỉnh sửa code** (Dockerfile, docker-compose, requirements, settings)
2. **Cài system dependencies** (Python, build tools)
3. **Cài CUDA Toolkit** (`nvidia-cuda-toolkit`)
4. **Setup Python venv** và activate
5. **Cài PyTorch+CUDA** (torch 2.3.1+cu121)
6. **Compile llama-cpp-python** với CMAKE_ARGS
7. **Cài các packages còn lại**
8. **Setup databases** (Redis, Qdrant)
9. **Download model GGUF**
10. **Chạy Streamlit app**

**Thời gian:** ~30-45 phút (phụ thuộc vào tốc độ mạng và compile)

---

**GPU H200 sẵn sàng! 🚀**
