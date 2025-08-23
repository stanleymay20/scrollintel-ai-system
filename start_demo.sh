#!/bin/bash

echo "Starting ScrollIntel Full Stack Demo..."
echo

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "Python is not installed or not in PATH"
        echo "Please install Python from https://python.org"
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

# Run the demo launcher
$PYTHON_CMD start_full_demo.py