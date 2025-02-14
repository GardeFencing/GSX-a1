#!/bin/bash

# Set Hugging Face token
export HUGGING_FACE_HUB_TOKEN=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

# Create models directory if it doesn't exist
mkdir -p llama.cpp/models

# Download model files with authentication
echo "Downloading model files..."

# Clone the llama.cpp repository with MiniCPM-V support
cd llama.cpp
git checkout minicpmv-main

# Download the quantized model files
echo "Downloading quantized model files..."
wget -P models https://huggingface.co/TheBloke/MiniCPM-V-2_6-GGUF/resolve/main/minicpm-v2-q4_k.gguf
wget -P models https://huggingface.co/TheBloke/MiniCPM-V-2_6-GGUF/resolve/main/mmproj-model-f16.gguf

echo "Download complete. Checking files..."

# Verify downloads
for file in "minicpm-v2-q4_k.gguf" "mmproj-model-f16.gguf"; do
    if [ -f "models/$file" ]; then
        echo "✓ $file downloaded successfully"
        ls -lh "models/$file"
    else
        echo "✗ Failed to download $file"
        exit 1
    fi
done

echo "All model files downloaded successfully!"
