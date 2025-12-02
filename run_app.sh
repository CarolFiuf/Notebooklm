#!/bin/bash
set -e

echo "=== Starting NotebookLM Application with GPU Support ==="

# Activate virtual environment
source /root/Notebooklm/venv/bin/activate

# Navigate to project directory
cd /root/Notebooklm

# Check GPU availability
echo "Checking GPU..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

# Check if model file exists
if [ ! -f "data/models/Qwen3-8B-Q4_K_M.gguf" ]; then
    echo ""
    echo "WARNING: Model file not found at data/models/Qwen3-8B-Q4_K_M.gguf"
    echo "Please download the model first:"
    echo "  mkdir -p data/models"
    echo "  wget -O data/models/Qwen3-8B-Q4_K_M.gguf <model_url>"
    echo ""
    read -p "Do you want to continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Start Streamlit application
echo ""
echo "Starting Streamlit on port 8501..."
echo "Access the app at: http://localhost:8501"
echo ""

streamlit run src/frontend/app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false
