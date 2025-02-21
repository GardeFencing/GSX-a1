### GSX-a1 (Generative Sports Expert - alpha 1)



A SOTA Quantized Multimodal LLM for Generative Sports Analytics. 

# VidCLIP Chatbot

A video analysis chatbot powered by MiniCPM-V that can analyze videos and answer questions about them.

## Server Setup Instructions

1. **System Requirements**
   - Ubuntu 20.04 or later
   - Python 3.10+
   - At least 8GB RAM
   - At least 10GB free disk space

2. **Install System Dependencies**
```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake pkg-config python3.10-dev python3.10-venv
```

3. **Clone and Build llama.cpp**
```bash
# Clone llama.cpp with MiniCPM-V support
git clone -b minicpm-v2.5 https://github.com/OpenBMB/llama.cpp.git
cd llama.cpp

# Build llama.cpp
mkdir build && cd build
cmake .. -DLLAMA_CUBLAS=OFF
make -j4
cd ../..
```

4. **Download MiniCPM-V Models**
```bash
# Create model directory
mkdir -p MiniCPM-V-2_6-gguf

# Download models from HuggingFace
wget https://huggingface.co/openbmb/MiniCPM-V-2_6-gguf/resolve/main/ggml-model-Q4_K_M.gguf -O MiniCPM-V-2_6-gguf/ggml-model-Q4_K_M.gguf
wget https://huggingface.co/openbmb/MiniCPM-V-2_6-gguf/resolve/main/mmproj-model-f16.gguf -O MiniCPM-V-2_6-gguf/mmproj-model-f16.gguf
```

5. **Setup Python Environment**
```bash
# Create and activate virtual environment
python3.10 -m venv venv_py310
source venv_py310/bin/activate

# Install requirements
pip install -r requirements.txt
```

6. **Run the Server**
```bash
# Start the server
python server.py
```

## Testing the Setup

1. **Test llama.cpp Installation**
```bash
cd llama.cpp
./llama-minicpmv-cli -m ../MiniCPM-V-2_6-gguf/ggml-model-Q4_K_M.gguf --mmproj ../MiniCPM-V-2_6-gguf/mmproj-model-f16.gguf -c 8192 --temp 0.7 --top-p 0.8 --top-k 100 --repeat-penalty 1.05 --video path/to/video.mp4 -p "What is happening in this video?"
```

2. **Test Web Interface**
- Open browser and go to `http://localhost:5000`
- Upload a video
- Ask questions about the video

## API Usage

The server exposes the following endpoints:

1. `/upload` - Upload a video file
   - Method: POST
   - Input: Form data with video file
   - Returns: Initial video analysis

2. `/chat` - Ask questions about the uploaded video
   - Method: POST
   - Input: JSON with query
   - Returns: AI response about the video

## Troubleshooting

1. If you get "No such file or directory" for llama-minicpmv-cli:
   - Make sure you're in the llama.cpp directory
   - Rebuild with `make clean && make`

2. If you get model loading errors:
   - Verify model files exist in MiniCPM-V-2_6-gguf directory
   - Check file permissions

3. For memory errors:
   - Reduce context size with -c parameter
   - Close other memory-intensive applications 
