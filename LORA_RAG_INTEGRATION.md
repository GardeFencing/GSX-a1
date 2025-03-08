# Integrating LoRA Fine-tuning with RAG for Fencing Video Analysis

This guide explains how to use the LoRA fine-tuned MiniCPM-V model with a Retrieval-Augmented Generation (RAG) system to provide enhanced fencing video analysis.

## Overview

This integration combines:

1. **LoRA Fine-tuned Model**: A MiniCPM-V model fine-tuned on fencing images and expert knowledge
2. **RAG System**: Retrieves relevant information from fencing books to enhance responses
3. **Video Analysis**: Processes entire videos rather than frame-by-frame analysis

## Setup Process

### Prerequisites

- AWS EC2 instance with GPU (recommended: g4dn.xlarge or better)
- CUDA and PyTorch installed
- Basic understanding of Python and shell scripting

### Quick Setup

1. **Run the integration script**:
   ```bash
   ./setup_lora_integration.sh
   ```

   This script will:
   - Install required dependencies
   - Extract knowledge from fencing PDFs for RAG
   - Update environment variables
   - Optionally run the LoRA fine-tuning process
   - Integrate everything with the web application

2. **Start the web server**:
   ```bash
   python web_app.py
   ```

## How It Works

### 1. Video Upload and Analysis

1. When a video is uploaded through the `/analyze/upload` endpoint, it's processed by the VideoAnalyzer class
2. The analyzer uses llama.cpp's `llama-minicpmv-cli` to process the entire video
3. If `USE_LORA=true`, it uses the LoRA fine-tuned model instead of the base model

### 2. RAG Enhancement

1. Before processing, the system retrieves relevant passages from fencing books
2. These passages are added to the prompt to provide domain-specific context
3. The enhanced prompt helps the model generate more accurate and detailed responses

### 3. Chat Functionality

The same integration applies to the chat endpoint, where:
1. The system maintains conversation history
2. Each new user query is processed with RAG enhancement
3. The LoRA fine-tuned model provides fencing-specific responses

## Configuration Options

You can adjust the following settings in the `.env` file:

```
# Model Settings
MODEL=ggml-model-Q4_K_M.gguf         # Base model for non-LoRA mode
CONTEXT_SIZE=4096                     # Context window size
TEMPERATURE=0.7                       # Temperature (higher = more creative)
TOP_P=0.8                             # Top-p sampling
TOP_K=100                             # Top-k sampling
REPEAT_PENALTY=1.05                   # Penalty for repeating tokens

# LoRA Settings
USE_LORA=true                         # Whether to use LoRA model
LORA_MODEL_PATH=path/to/model         # Path to merged LoRA model
```

## Troubleshooting

### LoRA Model Not Found

If you see an error about the LoRA model not being found:

1. Check that the model has been fine-tuned and merged
2. Verify the path in `LORA_MODEL_PATH` in `.env`
3. Set `USE_LORA=false` to fallback to the base model

### RAG Not Working

If the RAG enhancement isn't working:

1. Check that `extract_fencing_knowledge.py` has been run successfully
2. Verify that `GardeRAG/fencing_passages.json` exists
3. Check logs for any errors with the RAG system

## Next Steps

- **Add more fencing resources** to the `GardeRAG/books` directory to expand knowledge
- **Re-train the LoRA model** with more specific fencing data
- **Adjust RAG parameters** in `VideoAnalyzer._enhance_prompt_with_rag()` to control retrieval

## References

- LoRA fine-tuning guide: See `lora.txt`
- MiniCPM-V repository: https://github.com/OpenBMB/MiniCPM-V/tree/main 