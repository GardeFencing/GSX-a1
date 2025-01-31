import os
import torch
import numpy as np
import cv2
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

def extract_video_features(video_path, output_dir):
    """Extract features using CLIP model."""
    try:
        # Check if video file exists
        print(f"Checking video file at: {os.path.abspath(video_path)}")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory created/verified: {output_dir}")
        
        # Load CLIP model
        print("Loading CLIP model...")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model.eval()
        print("CLIP model loaded successfully")
        
        # Read video
        print(f"\nReading video from {video_path}")
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")
        
        # Get video info
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Video properties - Total frames: {total_frames}, FPS: {fps}")
        
        if total_frames <= 0 or fps <= 0:
            cap.release()
            raise ValueError(f"Invalid video properties: frames={total_frames}, fps={fps}")
            
        print(f"Total frames: {total_frames}, FPS: {fps}")
        
        # Sample frames (1 frame per second)
        frame_interval = max(1, int(fps))  # Ensure frame_interval is at least 1
        frames = []
        frame_count = 0
        
        print(f"Frame interval set to: {frame_interval}")
        print("Extracting frames...")
        
        expected_frames = total_frames // frame_interval
        with tqdm(total=expected_frames) as pbar:
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
        
        if not frames:
            raise ValueError("No frames were extracted from the video")
            
        print(f"Extracted {len(frames)} frames")
        
        # Process frames in batches
        batch_size = 32
        all_features = []
        
        print("Extracting CLIP features...")
        for i in tqdm(range(0, len(frames), batch_size)):
            batch = frames[i:i + batch_size]
            # Process frames with CLIP
            inputs = processor(images=batch, return_tensors="pt", padding=True)
            
            with torch.no_grad():
                outputs = model.get_image_features(**inputs)
                all_features.append(outputs)
        
        # Combine all features
        video_features = torch.cat(all_features, dim=0)
        # Average pool features across frames
        video_features = torch.mean(video_features, dim=0)
        
        # Save features
        output_path = os.path.join(output_dir, 'clip_features.pt')
        torch.save(video_features, output_path)
        print(f"Saved features to {output_path}")
        return video_features
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print("Full traceback:")
        print(traceback.format_exc())
        raise

if __name__ == "__main__":
    try:
        # Use absolute path to the video file
        video_path = os.path.abspath("En-Garde+Lunge - Made with Clipchamp.mp4")
        output_dir = "data/local_video"
        print(f"Starting feature extraction from {video_path}")
        extract_video_features(video_path, output_dir)
    except Exception as e:
        print(f"Main error: {str(e)}")
        import sys
        sys.exit(1) 