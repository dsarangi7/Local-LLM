#!/bin/bash

# Wait for Ollama to be ready
echo "Waiting for Ollama server to be ready..."
while ! curl -s http://ollama:11434/api/health > /dev/null; do
    sleep 1
done
echo "Ollama server is ready!"

# Start the GUI application
python gui.py 