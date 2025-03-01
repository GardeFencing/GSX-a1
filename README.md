# GSX-a1 (Generative Sports Expert - alpha 1)

**A State-of-the-Art Quantized Multimodal LLM for Generative Sports Analytics**  
**VidCLIP Chatbot** — A video analysis chatbot powered by **MiniCPM-V** that can analyze videos and answer questions about them.

Reference the MiniCPM Docs: [MiniCPM Docs](https://modelbest.feishu.cn/wiki/C2BWw4ZP0iCDy7kkCPCcX2BHnOf)

---

## 📚 Table of Contents

- [🌟 Features](#features)
- [🚀 System Requirements](#system-requirements)
- [⚙️ Installation Guide](#installation-guide)
  - [1. Install System Dependencies](#1-install-system-dependencies)
  - [2. Clone the Repository](#2-clone-the-repository)
  - [3. Clone and Build llama.cpp](#3-clone-and-build-llamacpp)
  - [4. Download MiniCPM-V Models](#4-download-minicpm-v-models)
  - [5. Setup Python Environment](#5-setup-python-environment)
  - [6. Run the Server](#6-run-the-server)
- [🧪 Testing the Setup](#testing-the-setup)
  - [1. Test llama.cpp Installation](#1-test-llamacpp-installation)
  - [2. Test Web Interface](#2-test-web-interface)
- [🌐 API Usage](#api-usage)
  - [1. `/upload`](#1-upload)
  - [2. `/chat`](#2-chat)
- [❓ Troubleshooting](#troubleshooting)

---

## Features

- Multimodal AI that can analyze videos and answer complex questions about sports events.
- Powered by **MiniCPM-V**, a state-of-the-art quantized language model.
- Uses **llama-minicpmv-cli** for efficient video understanding.
- Lightweight and optimized for resource-constrained environments.

---

## System Requirements

- **Operating System:** Ubuntu 20.04 or later
- **Python Version:** Python 3.10+
- **Memory:** At least 8GB RAM
- **Storage:** At least 10GB free disk space

---

## Installation Guide

### 1. Install System Dependencies

- Install system deps

- Linux

  ```bash
  sudo apt update
  sudo apt install -y build-essential cmake pkg-config python3.10-dev python3.10-venv ffmpeg libavcodec-dev libavformat-dev libavutil-dev libswscale-dev ccache redis-server supervisor
  ```

- MaxOS

  ```bash
  brew update
  brew install cmake pkg-config python@3.10 ffmpeg ccache redis supervisor
  ```


- Start redis cache (optional)

- Linux

  ```bash
  sudo systemctl enable redis-server
  sudo systemctl start redis-server
  sudo systemctl enable supervisor
  sudo systemctl start supervisor
  ```

- MaxOS

  ```bash
  brew services start redis
  brew services start supervisor
  ```

---

### 2. Clone the Repository

- Clone GSX-a1 repo

  ```bash
  git clone -b vidiQA_EC2_Branch https://github.com/GardeFencing/GSX-a1.git
  cd GSX-a1
  ```

---

### 3. Clone and Build llama.cpp

- Clone llama.cpp repo with video understanding

  ```bash
  git clone -b minicpmv-main https://github.com/OpenBMB/llama.cpp.git
  cd llama.cpp
  make LLAMA_FFMPEG=1
  ```

---

### 4. Download MiniCPM-V Models

- Install quantized model (Q4)

  ```bash
  cd ..
  mkdir -p MiniCPM-V-2_6-gguf
  wget https://huggingface.co/openbmb/MiniCPM-V-2_6-gguf/resolve/main/ggml-model-Q4_K_M.gguf -O MiniCPM-V-2_6-gguf/ggml-model-Q4_K_M.gguf
  wget https://huggingface.co/openbmb/MiniCPM-V-2_6-gguf/resolve/main/mmproj-model-f16.gguf -O MiniCPM-V-2_6-gguf/mmproj-model-f16.gguf
  ```

- If low memory, install more quantized model (Q2)

  ```bash
  cd ..
  mkdir -p MiniCPM-V-2_6-gguf
  wget https://huggingface.co/openbmb/MiniCPM-V-2_6-gguf/resolve/main/ggml-model-Q2_K.gguf -O MiniCPM-V-2_6-gguf/ggml-model-Q2_K.gguf
  wget https://huggingface.co/openbmb/MiniCPM-V-2_6-gguf/resolve/main/mmproj-model-f16.gguf
  ```

---

### 5. Setup Python Environment

- Install python deps (server)

  ```bash
  python3.10 -m venv venv_py310
  source venv_py310/bin/activate
  ```

  - Linux

  ```bash
  pip install -r requirements.txt
  ```

  - MacOS

  ```bash
  pip install -r requirements-osx.txt
  ```

---

### 6. Run the Server

- Run the dev server

  ```bash
  python server.py
  ```

- Run the prod server

  ```bash
  export GSX_A1_PATH=$(pwd)
  sudo -E ./supervisor_setup.sh
  ```

- Stop the prod server

  ```bash
  sudo supervisorctl shutdown
  ```

---

## Testing the Setup

### 1. Test llama.cpp Installation

- Test command (Q4)

  ```bash
  cd llama.cpp
  ./llama-minicpmv-cli \
    -m ../MiniCPM-V-2_6-gguf/ggml-model-Q4_K_M.gguf \
    --mmproj ../MiniCPM-V-2_6-gguf/mmproj-model-f16.gguf \
    -c 4096 \
    --temp 0.7 \
    --top-p 0.8 \
    --top-k 100 \
    --repeat-penalty 1.05 \
    --video ../fencing_trimmed.mp4 \
    -p "What is happening in this video?"
  ```

- Test command (Q2)

  ```bash
  cd llama.cpp
  ./llama-minicpmv-cli \
    -m ../MiniCPM-V-2_6-gguf/ggml-model-Q2_K.gguf \
    --mmproj ../MiniCPM-V-2_6-gguf/mmproj-model-f16.gguf \
    -c 4096 \
    --temp 0.7 \
    --top-p 0.8 \
    --top-k 100 \
    --repeat-penalty 1.05 \
    --video ../fencing_trimmed.mp4 \
    -p "What is happening in this video?"
  ```

---

### 2. Test Web Interface

- Open your web browser and navigate to: `http://localhost:5000`
- **Upload a video**: Select a video file to analyze.
- **Ask questions**: Interact with the chatbot to ask questions about the video.

---

## API Usage

### 1. `/upload`
- **Method**: `POST`
- **Description**: Upload a video file for analysis.
- **Input**: Form data with video file
- **Returns**: Initial video analysis

---

### 2. `/chat`
- **Method**: `POST`
- **Description**: Ask questions about the uploaded video.
- **Input**: JSON with query
- **Returns**: AI-generated response about the video

---

## Troubleshooting

1. **`No such file or directory` for `llama-minicpmv-cli`**:
   - Ensure you are in the `llama.cpp` directory.
   - Rebuild using:
     ```bash
     make clean && make
     ```

2. **Model Loading Errors**:
   - Confirm model files are present in `MiniCPM-V-2_6-gguf`.
   - Check file permissions using:
     ```bash
     ls -l MiniCPM-V-2_6-gguf/
     ```

3. **Memory Errors**:
   - Reduce context size using the `-c` parameter.
   - Close other memory-intensive applications.


Q2 V model

Run #1: 3m2.320s
Run #2: 3m0.208s
Run #3: 3m4.236s
Average Real Time for Q2_K: 182.25 seconds

Q4 V model

Run #1: 3m18.254s
Run #2: 3m4.463s
Run #3: 3m15.103s
Average Real Time for Q4_K_M: 192.60 seconds