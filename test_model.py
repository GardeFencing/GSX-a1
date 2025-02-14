import os
import logging
from pathlib import Path
from GardeRAG.model import GardeRAG
from GardeRAG.frame import Frame

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_frame_analysis():
    try:
        # Initialize model
        model = GardeRAG()
        
        # Create a Frame object with proper Path
        frame_path = Path("test_frames/frame_0.jpg")
        frame = Frame(0, frame_path, 0.0)  # number, path, timestamp
        
        # Test frame analysis
        logger.info("Testing frame analysis...")
        result = model.analyze_frame(frame)
        
        # Print result
        logger.info("Analysis result:")
        logger.info(result)
        
    except Exception as e:
        logger.error(f"Error in test: {str(e)}", exc_info=True)

if __name__ == "__main__":
    test_frame_analysis() 