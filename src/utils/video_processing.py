import logging
from PIL import Image
import cv2
from typing import List
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_NUM_FRAMES = 32  # Maximum number of frames to process

def uniform_sample(l: List, n: int) -> List:
    """Uniformly samples n elements from list l"""
    if not l:
        return []
    gap = len(l) / n
    return [l[int(i * gap + gap / 2)] for i in range(n)]

def encode_video(video_path: str) -> List[Image.Image]:
    """
    Encodes a video file into a list of frames.
    
    Args:
        video_path (str): Path to the video file
        
    Returns:
        List[Image.Image]: List of video frames as PIL Images
    """
    try:
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Calculate frame sampling interval
        sample_interval = max(1, fps // 2)  # Sample at 1/2 fps
        frame_indices = list(range(0, total_frames, sample_interval))
        
        # Limit number of frames
        if len(frame_indices) > MAX_NUM_FRAMES:
            frame_indices = uniform_sample(frame_indices, MAX_NUM_FRAMES)
        
        # Extract frames
        pil_frames = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count in frame_indices:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(frame_rgb)
                pil_frames.append(pil_frame)
                
            frame_count += 1
            
            if len(pil_frames) >= len(frame_indices):
                break
        
        cap.release()
        logger.info(f"Processed {len(pil_frames)} frames from video")
        return pil_frames
        
    except Exception as e:
        logger.error(f"Error encoding video: {str(e)}")
        raise 