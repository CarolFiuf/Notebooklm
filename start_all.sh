#!/bin/bash
set -e

echo "=========================================="
echo "Starting NotebookLM with GPU Support"
echo "=========================================="

cd /root/Notebooklm

# Check GPU
echo ""
echo "1. Checking GPU..."
nvidia-smi | grep H200

# Start Redis
echo ""
echo "2. Starting Redis..."
redis-server --daemonize yes
sleep 2
redis-cli ping

# Start Qdrant
echo ""
echo "3. Starting Qdrant..."
if ! pgrep -f qdrant > /dev/null; then
    nohup ./qdrant > qdrant.log 2>&1 &
    sleep 5
fi
curl -s http://localhost:6333 | head -1

# Activate venv and start app
echo ""
echo "4. Starting Streamlit Application..."
source venv/bin/activate

# Check CUDA
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Start Streamlit
streamlit run src/frontend/app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    2>&1 | tee streamlit.log &

echo ""
echo "=========================================="
echo "Waiting for application to start..."
sleep 15

# Check if running
if curl -s http://localhost:8501 > /dev/null; then
    echo "✅ Application is running at http://localhost:8501"
else
    echo "❌ Application failed to start. Check logs:"
    echo "   tail -50 streamlit.log"
fi

echo "=========================================="
