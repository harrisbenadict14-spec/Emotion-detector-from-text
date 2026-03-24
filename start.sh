#!/bin/bash

echo "========================================"
echo "   AI Emotion Detector - Startup Script"
echo "========================================"
echo

echo "[1/3] Starting Backend Server..."
cd backend

echo "Installing dependencies..."
pip install -r requirements.txt

echo
echo "[2/3] Training ML Model..."
cd model
python train.py

echo
echo "[3/3] Starting API Server..."
cd ..
python main.py

echo
echo "========================================"
echo "Backend is running on http://localhost:8000"
echo "Open frontend/index.html in your browser"
echo "========================================"
read -p "Press Enter to continue..."
