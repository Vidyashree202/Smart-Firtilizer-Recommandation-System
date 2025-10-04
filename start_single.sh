#!/bin/bash

echo "Starting Smart Fertilizer Recommendation System (Single Terminal)"
echo "================================================================"
echo

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 is not installed or not in PATH"
    echo "Please install Python3 and try again"
    exit 1
fi

# Check if Node.js is available
if ! command -v node &> /dev/null; then
    echo "Error: Node.js is not installed or not in PATH"
    echo "Please install Node.js and try again"
    exit 1
fi

# Check if chatbot_integrated directory exists
if [ ! -d "chatbot_integrated" ]; then
    echo "Error: chatbot_integrated directory not found"
    echo "Please make sure the React chatbot files are copied"
    exit 1
fi

# Check if node_modules exists in chatbot_integrated
if [ ! -d "chatbot_integrated/node_modules" ]; then
    echo "Installing React dependencies..."
    cd chatbot_integrated
    npm install
    cd ..
    echo
fi

echo "Starting both servers in single terminal..."
echo
python3 run_single_terminal.py
