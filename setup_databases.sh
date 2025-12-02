#!/bin/bash
set -e

echo "=== Setting up databases for NotebookLM ==="

# Check if PostgreSQL is installed
if ! command -v psql &> /dev/null; then
    echo "Installing PostgreSQL..."
    sudo apt-get update
    sudo apt-get install -y postgresql postgresql-contrib
    sudo systemctl start postgresql || sudo pg_ctlcluster 14 main start || echo "PostgreSQL already running"
fi

# Check if Qdrant is running
if ! curl -s http://localhost:6333 > /dev/null 2>&1; then
    echo "Qdrant not running. You can either:"
    echo "1. Run Qdrant with Docker:"
    echo "   docker run -d -p 6333:6333 -p 6334:6334 -v qdrant_data:/qdrant/storage qdrant/qdrant:v1.15"
    echo ""
    echo "2. Or download and run Qdrant binary from https://github.com/qdrant/qdrant/releases"
    echo ""
fi

# Check if Redis is installed
if ! command -v redis-cli &> /dev/null; then
    echo "Installing Redis..."
    sudo apt-get install -y redis-server
    sudo systemctl start redis-server || redis-server --daemonize yes
fi

echo ""
echo "Database setup complete!"
echo ""
echo "PostgreSQL: Check with 'sudo -u postgres psql -c \"\\l\"'"
echo "Redis: Check with 'redis-cli ping'"
echo "Qdrant: Check with 'curl http://localhost:6333'"
