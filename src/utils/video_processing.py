import logging
from PIL import Image
import cv2
from typing import List
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_NUM_FRAMES = 64  # If CUDA OOM, set a smaller number

def uniform_sample(l: List, n: int) -> List:
    """
    Uniformly samples elements from a list.
    
    Args:
        - l (list): The input list.
        - n (int): The number of elements to sample.
        
    Returns:
        list: A list of sampled elements.
    """
    gap = len(l) / n
    idxs = [int(i * gap + gap / 2) for i in range(n)]
    return [l[i] for i in idxs]

def encode_video(video_path: str) -> List[Image.Image]:
    """
    Encodes a video file into a list of frames.
    
    Args:
        video_path (str): The path to the video file.
        
    Returns:
        list: A list of frames, where each frame is represented as an Image object.
    """
    try:
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate frame sampling interval (1 frame per second)
        sample_interval = max(1, int(fps))
        frame_indices = list(range(0, total_frames, sample_interval))
        
        # Uniformly sample frames if the number of frames is too large
        if len(frame_indices) > MAX_NUM_FRAMES:
            frame_indices = uniform_sample(frame_indices, MAX_NUM_FRAMES)
            
        # Extract frames
        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame_rgb)
            frames.append(pil_frame)
            
        # Release video capture
        cap.release()
        
        if not frames:
            raise ValueError(f"No frames could be extracted from video: {video_path}")
            
        # Log the successful encoding of the video
        logger.info(f"Video encoded successfully. Extracted {len(frames)} frames.")
        
        return frames
        
    except Exception as e:
        logger.error(f"Error encoding video: {str(e)}")
        raise 