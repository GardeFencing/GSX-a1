#!/bin/bash

# Exit on error
set -e

echo "Setting up VidCLIP Chatbot..."

# Build llama.cpp
echo "Building llama.cpp..."
cd llama.cpp
mkdir -p build && cd build
cmake .. -DLLAMA_CUBLAS=OFF
make -j4
cd ../..

# Create necessary directories
echo "Creating directories..."
mkdir -p uploads temp_frames

# Set permissions
echo "Setting permissions..."
chmod -R 755 uploads temp_frames
chmod +x llama.cpp/build/bin/*

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cat > .env << EOL
OPENAI_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
ACCESS_TOKEN=your_token_here
EOL
fi

# Activate virtual environment and install requirements
echo "Installing Python requirements..."
source venv_py310/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete! Next steps:"
echo "1. Update your API keys in .env"
echo "2. Run 'python server.py' to start the server" 