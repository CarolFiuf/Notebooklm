#!/bin/bash
set -e

echo "=== Installing dependencies with GPU support ==="

# Activate virtual environment
source /root/Notebooklm/venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1 support
echo "Installing PyTorch with CUDA 12.1..."
pip install torch==2.3.1+cu121 torchaudio==2.3.1+cu121 torchvision==0.18.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121

# Install llama-cpp-python with CUDA support
echo "Installing llama-cpp-python with CUDA support..."
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=all" pip install --force-reinstall --no-binary llama-cpp-python 'llama-cpp-python>=0.3.16'

# Install other ML packages
echo "Installing other ML packages..."
pip install sentence-transformers transformers

# Install RAG & Vector DB packages
echo "Installing RAG & Vector DB packages..."
pip install langchain>=0.1.0 langchain-community>=0.0.38 langchain-qdrant>=0.1.0
pip install qdrant-client tenacity>=8.2.0 rank-bm25

# Install document processing packages
echo "Installing document processing packages..."
pip install pymupdf==1.23.8 pdfplumber==0.10.3 Pillow>=10.3.0
pip install python-docx>=1.0.0 docx2txt>=0.8

# Install PaddlePaddle GPU version
echo "Installing PaddlePaddle GPU version..."
pip install paddlepaddle-gpu==2.6.1 -i https://mirror.baidu.com/pypi/simple || pip install paddlepaddle-gpu==2.6.1
pip install paddleocr==2.7.3

# Install database packages
echo "Installing database packages..."
pip install psycopg2-binary==2.9.7 sqlalchemy==2.0.23 alembic==1.13.1

# Install web framework
echo "Installing web framework..."
pip install streamlit==1.39.0 fastapi>=0.115.0 uvicorn==0.24.0

# Install utilities
echo "Installing utilities..."
pip install pydantic>=2.11.7,<3.0 python-multipart==0.0.6 python-dotenv==1.0.0
pip install requests>=2.32.2 beautifulsoup4==4.12.2 numpy pandas==2.1.4
pip install uuid==1.30 tqdm cachetools>=5.3.0 sortedcontainers>=2.4.0
pip install pydantic-settings chardet aiofiles httpx

# Install development tools
echo "Installing development tools..."
pip install jupyter==1.0.0 ipykernel>=6.29.3,<7 pytest==7.4.3 black==23.11.0

echo "=== Installation complete! ==="
echo ""
echo "To verify GPU support, run:"
echo "  python -c 'import torch; print(f\"CUDA: {torch.cuda.is_available()}\")'  "
