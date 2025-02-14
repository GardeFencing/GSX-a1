# VidCLIPChatbot

A video analysis and chat system for fencing technique analysis using MiniCPM-V and llama.cpp.

## Prerequisites

- Python 3.10 or higher
- Git
- macOS with Apple Silicon (M1/M2/M3) or Intel processor

## Installation

1. Clone the repository and enter the directory:
```bash
git clone https://github.com/yourusername/VidCLIPChatbot.git
cd VidCLIPChatbot
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install Python dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. Download and set up the models:
```bash
# This script will download the required model files
python download_models.py
```

## MiniCPM-V Integration

This project leverages [MiniCPM-V](https://github.com/OpenBMB/MiniCPM-o/tree/main?tab=readme-ov-file#chat-with-video), a powerful multimodal model capable of analyzing video content with high precision. Our implementation uses the following optimized parameters for fencing analysis:

```bash
./llama-minicpmv-cli -m models/ggml-model-Q2_K.gguf --mmproj models/mmproj-model-f16.gguf -c 2048 --temp 0.1 --top-p 0.9 --top-k 40 --repeat-penalty 1.1 --n-gpu-layers 1 --threads 8 --batch-size 512 --mlock
```

Key features of our MiniCPM-V integration:
- Optimized for Apple Silicon with Metal acceleration
- Efficient frame analysis with 2048 context window
- Low temperature (0.1) for precise, deterministic responses
- Memory-efficient batch processing with mlock support
- GPU acceleration with configurable layer count

## Project Structure

- `web_app.py`: Main Flask application for video upload and analysis
- `GardeRAG/`: Core analysis system
  - `model.py`: MiniCPM-V integration and frame analysis
  - `frame.py`: Frame processing utilities

## Model Capabilities

Our implementation of MiniCPM-V is specifically tuned for fencing analysis:

1. **Frame Analysis**
   - Detailed analysis of fencing techniques and positions
   - Recognition of specific moves (e.g., lunges, parries, ripostes)
   - Form and posture evaluation
   - Technical precision assessment

2. **Interactive Chat**
   - Real-time Q&A about specific frames
   - Technical explanations of observed techniques
   - Coaching insights and recommendations
   - Tactical analysis of movements

3. **Video Processing**
   - Intelligent frame selection for analysis
   - Temporal context understanding
   - Efficient batch processing of video segments
   - High-quality frame extraction and storage

## Running the Application

1. Start the Flask server:
```bash
python web_app.py
```

2. Open a web browser and navigate to:
```
http://localhost:5000
```

3. Upload a fencing video to analyze techniques and positions.

## Hardware Requirements

- Recommended:
  - macOS with Apple Silicon (M1/M2/M3)
  - 16GB RAM or more
  - 5GB free disk space for models and temporary files

## Troubleshooting

1. If you encounter model loading issues:
   - Verify models are downloaded correctly in `llama.cpp/models/`
   - Check that both `ggml-model-Q2_K.gguf` and `mmproj-model-f16.gguf` exist

2. For MPS (Metal Performance Shaders) issues on macOS:
   - Ensure `PYTORCH_ENABLE_MPS_FALLBACK=1` is set in your environment
   - Try running with CPU if GPU acceleration fails

## Future Features

- MoveNet integration for pose estimation (coming soon)
- Biomechanical analysis of fencing movements
- Advanced frame selection using pose data

## License

[Your License Here]
