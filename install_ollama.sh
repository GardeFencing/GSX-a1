#!/bin/bash

# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull the Llama 3.2 Vision model
ollama pull llama3.2-vision

echo "Ollama installation completed. Starting Ollama service..."
ollama serve 