import os
import logging
from typing import Optional

# Local imports
from src.app.model import model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            
        # Generate response using CLI
        response = model.generate_response(video_path, question)
        if response is None:
            return "Error: Could not generate response"
            
        return response
        
    except Exception as e:
        logger.error(f"Error describing video: {str(e)}")
        return f"Error: {str(e)}" 