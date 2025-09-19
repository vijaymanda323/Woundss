#!/bin/bash

# Expo React Native + Flask API Starter
# Quick start script for development

echo "🚀 Starting Expo React Native + Flask API Starter"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if Python is installed
if ! command_exists python3; then
    echo "❌ Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

# Check if Node.js is installed
if ! command_exists node; then
    echo "❌ Node.js is not installed. Please install Node.js first."
    exit 1
fi

# Check if npm is installed
if ! command_exists npm; then
    echo "❌ npm is not installed. Please install npm first."
    exit 1
fi

echo "✅ Prerequisites check passed"
echo ""

# Start Flask server in background
echo "🐍 Starting Flask server..."
cd server

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Copy environment file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cp env.example .env
fi

# Start Flask server
echo "Starting Flask server on http://localhost:5000"
python app.py &
FLASK_PID=$!

# Go back to root directory
cd ..

# Start Expo app
echo ""
echo "📱 Starting Expo React Native app..."
cd mobile

# Install dependencies
echo "Installing Node.js dependencies..."
npm install

# Start Expo
echo "Starting Expo development server..."
echo "Press 'w' for web, 'a' for Android, 'i' for iOS"
echo ""
npx expo start

# Cleanup function
cleanup() {
    echo ""
    echo "🛑 Shutting down..."
    kill $FLASK_PID 2>/dev/null
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Wait for user to stop
wait





