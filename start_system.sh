#!/bin/bash
# Complete system startup script for NotebookLM with GPU support
# Run this after server restart to start all services

set -e

echo "================================================================="
echo "  NotebookLM GPU System Startup Script"
echo "================================================================="
echo ""

cd /root/Notebooklm

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to check if a process is running
check_process() {
    pgrep -f "$1" > /dev/null 2>&1
}

# Function to wait for a port to be ready
wait_for_port() {
    local port=$1
    local max_wait=30
    local count=0

    echo -n "  Waiting for port $port to be ready"
    while ! nc -z localhost $port 2>/dev/null; do
        sleep 1
        count=$((count + 1))
        echo -n "."
        if [ $count -ge $max_wait ]; then
            echo -e " ${RED}TIMEOUT${NC}"
            return 1
        fi
    done
    echo -e " ${GREEN}OK${NC}"
    return 0
}

echo "Step 1: Starting PostgreSQL..."
if pg_isready -h localhost -p 5432 > /dev/null 2>&1; then
    echo -e "  ${GREEN}✓${NC} PostgreSQL already running"
else
    sudo pg_ctlcluster 16 main start > /dev/null 2>&1 || {
        echo -e "  ${RED}✗${NC} Failed to start PostgreSQL"
        exit 1
    }
    wait_for_port 5432
    echo -e "  ${GREEN}✓${NC} PostgreSQL started"
fi

# Create database if not exists
sudo -u postgres psql -lqt | cut -d \| -f 1 | grep -qw notebooklm || {
    echo "  Creating database 'notebooklm'..."
    sudo -u postgres psql -c "CREATE DATABASE notebooklm;" > /dev/null 2>&1
    sudo -u postgres psql -c "ALTER USER postgres WITH PASSWORD 'password123';" > /dev/null 2>&1
    echo -e "  ${GREEN}✓${NC} Database created"
}

echo ""
echo "Step 2: Starting Redis..."
if check_process "redis-server"; then
    echo -e "  ${GREEN}✓${NC} Redis already running"
else
    redis-server --daemonize yes > /dev/null 2>&1
    wait_for_port 6379
    echo -e "  ${GREEN}✓${NC} Redis started"
fi

echo ""
echo "Step 3: Starting Qdrant..."
if check_process "qdrant" && nc -z localhost 6333 2>/dev/null; then
    echo -e "  ${GREEN}✓${NC} Qdrant already running"
else
    if [ ! -f "./qdrant" ]; then
        echo "  Downloading Qdrant v1.15.0..."
        wget -q https://github.com/qdrant/qdrant/releases/download/v1.15.0/qdrant-x86_64-unknown-linux-gnu.tar.gz
        tar -xzf qdrant-x86_64-unknown-linux-gnu.tar.gz
        chmod +x qdrant
        rm -f qdrant-x86_64-unknown-linux-gnu.tar.gz
    fi
    nohup ./qdrant > qdrant.log 2>&1 &
    wait_for_port 6333
    echo -e "  ${GREEN}✓${NC} Qdrant started"
fi

echo ""
echo "Step 4: Starting Streamlit Application..."
# Kill any existing streamlit processes
pkill -9 -f "streamlit run" > /dev/null 2>&1 || true
sleep 2

# Clear Python cache
find /root/Notebooklm -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Activate virtual environment and start Streamlit
source venv/bin/activate
export PYTHONPATH=/root/Notebooklm:$PYTHONPATH

nohup streamlit run src/frontend/app.py \
    --server.port=8888 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    > streamlit.log 2>&1 &

wait_for_port 8888
echo -e "  ${GREEN}✓${NC} Streamlit started on port 8888"

echo ""
echo "Step 5: Starting Cloudflare Tunnel..."
# Kill any existing cloudflared processes
pkill -9 cloudflared > /dev/null 2>&1 || true
sleep 2

nohup cloudflared tunnel --url http://localhost:8888 > cloudflare_tunnel.log 2>&1 &
echo "  Waiting for tunnel to initialize..."
sleep 8

# Extract tunnel URL
TUNNEL_URL=$(grep -o "https://.*\.trycloudflare\.com" cloudflare_tunnel.log 2>/dev/null | head -1)

if [ -n "$TUNNEL_URL" ]; then
    echo -e "  ${GREEN}✓${NC} Cloudflare Tunnel active"
else
    echo -e "  ${YELLOW}⚠${NC} Tunnel URL not yet available, check cloudflare_tunnel.log"
    TUNNEL_URL="(Check cloudflare_tunnel.log)"
fi

echo ""
echo "================================================================="
echo -e "  ${GREEN}✓ ALL SERVICES STARTED SUCCESSFULLY${NC}"
echo "================================================================="
echo ""
echo "Service Status:"
echo "  • PostgreSQL     : localhost:5432"
echo "  • Redis          : localhost:6379"
echo "  • Qdrant         : localhost:6333"
echo "  • Streamlit      : localhost:8888"
echo ""
echo "Public Access URL:"
echo -e "  ${GREEN}$TUNNEL_URL${NC}"
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader | \
    awk -F', ' '{printf "  • %s: %s total, %s used\n", $1, $2, $3}'
echo ""
echo "Quick Commands:"
echo "  • View Streamlit logs : tail -f streamlit.log"
echo "  • View Tunnel logs    : tail -f cloudflare_tunnel.log"
echo "  • Check Qdrant status : curl http://localhost:6333/collections"
echo "  • Stop all services   : pkill -f 'streamlit|cloudflared'"
echo ""
echo "================================================================="
