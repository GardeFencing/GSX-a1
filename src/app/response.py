import os
import logging
from typing import Optional
from PIL import Image

# Local imports
from src.app.model import load_model_and_tokenizer, generate_response
from src.utils.video_processing import encode_video

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model and tokenizer globally
model, tokenizer = load_model_and_tokenizer()

def describe_video(video_path: str, question: str) -> str:
    """
    Process a video and generate a response to a question about it.
    
    Args:
        video_path (str): Path to the video file
        question (str): Question about the video
        
    Returns:
        str: Generated response or error message
    """
    try:
        # Validate inputs
        if not os.path.exists(video_path):
            return "Error: Video file not found"
        if not question:
            return "Error: No question provided"
            
        # Extract frames from video
        frames = encode_video(video_path)
        if not frames:
            return "Error: Could not extract frames from video"
            
        # Generate response
        response = generate_response(model, tokenizer, frames, question)
        if response is None:
            return "Error: Could not generate response"
            
        return response
        
    except Exception as e:
        logger.error(f"Error describing video: {str(e)}")
        return f"Error: {str(e)}" 