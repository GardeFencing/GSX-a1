import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration for EC2 deployment"""
    # Server settings
    HOST = '0.0.0.0'  # Listen on all network interfaces
    PORT = int(os.getenv('PORT', 5000))
    DEBUG = False
    
    # Processing settings
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max-limit
    UPLOAD_FOLDER = 'uploads'
    TEMP_FOLDER = 'temp_frames'
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
    
    # Model settings
    MODEL_PATH = os.getenv('MODEL_PATH', 'MiniCPM-V-2_6-gguf/ggml-model-Q4_K_M.gguf')
    MMPROJ_PATH = os.getenv('MMPROJ_PATH', 'MiniCPM-V-2_6-gguf/mmproj-model-f16.gguf')
    CLI_PATH = os.getenv('CLI_PATH', './llama.cpp/llama-minicpmv-cli')
    
    # Resource limits
    MAX_WORKERS = int(os.getenv('MAX_WORKERS', 2))  # Number of worker processes
    MAX_THREADS = int(os.getenv('MAX_THREADS', 4))  # Threads per worker
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', 4))    # Frames to process at once
    MAX_QUEUE_SIZE = int(os.getenv('MAX_QUEUE_SIZE', 10))  # Max concurrent requests
    
    # Timeouts
    REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', 300))  # 5 minutes
    PROCESS_TIMEOUT = int(os.getenv('PROCESS_TIMEOUT', 60))   # 1 minute per frame batch
    
    @classmethod
    def init_app(cls, app):
        """Initialize Flask app with config"""
        # Create required directories
        os.makedirs(cls.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(cls.TEMP_FOLDER, exist_ok=True)
        
        # Set Flask configs
        app.config['MAX_CONTENT_LENGTH'] = cls.MAX_CONTENT_LENGTH
        app.config['UPLOAD_FOLDER'] = cls.UPLOAD_FOLDER 