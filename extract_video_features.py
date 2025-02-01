import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_video_features(video_path, output_dir, frame_rate=1, batch_size=32):
    """
    Extract video features using CLIP.
    
    Args:
        video_path (str): Path to the video file.
        output_dir (str): Directory to save extracted features.
        frame_rate (int): Number of frames to extract per second.
        batch_size (int): Batch size for processing frames.
        
    Returns:
        torch.Tensor: Aggregated (pooled) video features.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Loading CLIP model for video feature extraction...")
        # Use consistent CLIP variant
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model.eval()
        
        logger.info(f"Reading video from {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("Error opening video file.")
            return None
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"Total frames: {total_frames}, FPS: {fps}")
        
        # Compute the interval between frames to extract based on desired frame_rate
        frame_interval = int(fps / frame_rate) if fps > 0 else 1
        frames = []
        frame_count = 0
        
        logger.info("Extracting frames...")
        with tqdm(total=total_frames // frame_interval) as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_count % frame_interval == 0:
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                    pbar.update(1)
                frame_count += 1
        cap.release()
        logger.info(f"Extracted {len(frames)} frames.")
        
        all_features = []
        logger.info("Extracting CLIP features from frames...")
        for i in tqdm(range(0, len(frames), batch_size)):
            batch = frames[i:i + batch_size]
            inputs = processor(images=batch, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = model.get_image_features(**inputs)
                outputs = F.normalize(outputs, p=2, dim=1)
                all_features.append(outputs)
        video_features = torch.cat(all_features, dim=0)
        
        # Average pooling for feature aggregation
        video_features = torch.mean(video_features, dim=0)
        
        output_path = os.path.join(output_dir, 'clip_features.pt')
        torch.save(video_features, output_path)
        logger.info(f"Saved video features to {output_path}")
        return video_features
        
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        import traceback
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    try:
        video_path = os.path.abspath("En-Garde+Lunge - Made with Clipchamp.mp4")
        output_dir = "data/local_video"
        frame_rate = 1  # Adjust this value to control frame sampling rate
        logger.info(f"Starting feature extraction from {video_path}")
        extract_video_features(video_path, output_dir, frame_rate=frame_rate)
    except Exception as e:
        logger.error(f"Main error: {str(e)}")
        import sys
        sys.exit(1) 