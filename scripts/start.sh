#!/bin/bash
# =============================================================================
# Startup Script for Chemical Property Prediction
# =============================================================================
# Starts both the FastAPI backend and Streamlit dashboard
# Usage: bash scripts/start.sh
# =============================================================================

set -e

echo "=========================================="
echo "Chemical Property Prediction System"
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running in Docker
if [ -f /.dockerenv ]; then
    echo -e "${BLUE}Running inside Docker container${NC}"
    IN_DOCKER=true
else
    IN_DOCKER=false
fi

# Function to check if a port is available
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 1
    else
        return 0
    fi
}

# Check Python
if ! command -v python &> /dev/null; then
    echo "Error: Python not found"
    exit 1
fi

echo -e "${GREEN}Python version:${NC}"
python --version

echo ""
echo -e "${YELLOW}Starting services...${NC}"

# Check ports
API_PORT=${API_PORT:-8000}
STREAMLIT_PORT=${STREAMLIT_PORT:-8501}

if ! check_port $API_PORT; then
    echo "Warning: Port $API_PORT is already in use"
fi

if ! check_port $STREAMLIT_PORT; then
    echo "Warning: Port $STREAMLIT_PORT is already in use"
fi

# Create necessary directories
mkdir -p data/raw data/processed models/saved logs

echo ""
echo -e "${GREEN}1. Starting FastAPI Backend${NC}"
echo "   URL: http://localhost:$API_PORT"
echo "   Docs: http://localhost:$API_PORT/docs"
echo ""

# Start API in background
python -m uvicorn api.main:app \
    --host 0.0.0.0 \
    --port $API_PORT \
    --reload &

API_PID=$!
echo "   API PID: $API_PID"

# Wait for API to be ready
echo "   Waiting for API to start..."
sleep 5

# Check if API started successfully
if ! kill -0 $API_PID 2>/dev/null; then
    echo "Error: API failed to start"
    exit 1
fi

echo -e "${GREEN}   API is running!${NC}"

echo ""
echo -e "${GREEN}2. Starting Streamlit Dashboard${NC}"
echo "   URL: http://localhost:$STREAMLIT_PORT"
echo ""

# Start Streamlit
python -m streamlit run frontend/app.py \
    --server.port $STREAMLIT_PORT \
    --server.address 0.0.0.0 &

STREAMLIT_PID=$!
echo "   Streamlit PID: $STREAMLIT_PID"

echo ""
echo "=========================================="
echo -e "${GREEN}All services are starting!${NC}"
echo "=========================================="
echo ""
echo -e "FastAPI Backend:    ${BLUE}http://localhost:$API_PORT${NC}"
echo -e "API Documentation:  ${BLUE}http://localhost:$API_PORT/docs${NC}"
echo -e "Streamlit UI:       ${BLUE}http://localhost:$STREAMLIT_PORT${NC}"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Wait for both processes
wait $API_PID $STREAMLIT_PID
