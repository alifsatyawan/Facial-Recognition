#!/bin/bash

# Start the web-based facial recognition system

echo "Starting Facial Recognition Web Interface..."

# Check if we're in conda environment
if [[ -z "${CONDA_DEFAULT_ENV}" ]]; then
    echo "Activating conda environment..."
    eval "$(conda shell.bash hook)"
    conda activate facial_recognition
fi

# Install backend dependencies if needed
echo "Checking backend dependencies..."
cd backend
pip install -r requirements.txt > /dev/null 2>&1

# Start backend server
echo "Starting Flask backend server..."
python app.py &
BACKEND_PID=$!

# Wait for backend to start
sleep 3

# Start frontend on port 3001
echo "Starting React frontend on port 3001..."
cd ../frontend
PORT=3001 npm start &
FRONTEND_PID=$!

echo ""
echo "================================================"
echo "Facial Recognition Web Interface is running!"
echo "================================================"
echo "Backend API: http://localhost:5000"
echo "Frontend UI: http://localhost:3001"
echo ""
echo "Press Ctrl+C to stop both servers"
echo "================================================"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down servers..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}

# Set up cleanup on interrupt
trap cleanup INT

# Wait for processes
wait
