# VidCLIPChatbot

A video analysis and chat system that combines pose estimation, biomechanical analysis, and RAG (Retrieval-Augmented Generation) for fencing technique analysis.

## Prerequisites

- Python 3.10 (recommended)
- Git
- macOS with Apple Silicon (M1/M2/M3) or Intel processor

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/VidCLIPChatbot.git
cd VidCLIPChatbot
```

2. Create and activate a virtual environment:
```bash
# Create virtual environment
python3.10 -m venv venv_tf

# Activate virtual environment
source venv_tf/bin/activate
```

3. Install dependencies:
```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

4. Install MMPT package:
```bash
# First, ensure you're in the project root
cd examples/MMPT

# Clean any previous installations
rm -rf build dist *.egg-info

# Install in development mode
pip install -e .

# Verify installation
python -c "from mmpt.processors.processor import MMPTProcessor; print('MMPT installed successfully')"

# Return to project root
cd ../..
```

5. Set up MoveNet Thunder model:
```bash
# Run the setup script
python setup_movenet_thunder.py
```

## Environment Variables

Create a `.env` file in the project root with the following variables:
```
PYTORCH_ENABLE_MPS_FALLBACK=1  # For macOS with Apple Silicon
```

## Project Structure

- `web_app.py`: Main Flask application
- `pose_estimator.py`: Pose estimation and biomechanical analysis
- `GardeRAG/`: RAG system implementation
  - `model.py`: Core RAG model
  - `frame.py`: Frame processing utilities
  - `prompt.py`: Prompt templates
- `examples/MMPT/`: Video feature extraction
  - `mmpt/`: Core MMPT package
    - `processors/`: Video processing modules
    - `models/`: Model implementations
    - `utils/`: Utility functions

## Running the Application

1. Start the Flask server:
```bash
python web_app.py
```

2. Open a web browser and navigate to:
```
http://localhost:5000
```

## Hardware Recommendations

- For optimal performance:
  - macOS with Apple Silicon (M1/M2/M3)
  - 16GB RAM or more
  - 10GB free disk space

- Minimum requirements:
  - macOS (Intel/Apple Silicon)
  - 8GB RAM
  - 5GB free disk space

## Troubleshooting

1. If you encounter memory issues:
   - Reduce batch size in video processing
   - Close other memory-intensive applications

2. For MPS (Metal Performance Shaders) issues on macOS:
   - Ensure PYTORCH_ENABLE_MPS_FALLBACK=1 is set
   - Try falling back to CPU if needed

3. If MMPT import fails:
   - Verify MMPT installation: `pip list | grep mmpt`
   - Reinstall MMPT if needed: `cd examples/MMPT && pip install -e .`

## License

[Your License Here]
