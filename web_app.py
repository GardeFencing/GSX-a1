import os
import sys

# Set MPS configuration before any imports
if sys.platform == 'darwin':
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.7'

from flask import Flask, request, jsonify, render_template, send_file
import logging
import cv2
import base64
import numpy as np
from PIL import Image
import io
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import time
import asyncio

# Import ML models and local RAG system
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
# from pose_estimator import PoseEstimator, JointAngles  # Temporarily disabled
from GardeRAG.model import GardeRAG  # Import our local RAG system
from GardeRAG.frame import Frame, FrameStorage  # Import Frame and FrameStorage classes

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
video_processor = None  # Global VideoProcessor instance
rag_system = None  # Global GardeRAG instance
frame_storage = None  # Global FrameStorage instance

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Initialize ML models
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
sent_transformer = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# Initialize GardeRAG and FrameStorage
logger.info("Initializing GardeRAG system and FrameStorage...")
frame_storage = FrameStorage()
rag_system = GardeRAG()

# ------------------------------------------------------------------------------
# BIOMECHANICAL ANALYSIS - Temporarily disabled
# ------------------------------------------------------------------------------

# @dataclass
# class BiomechanicalMetrics:
#     """Store biomechanical analysis results"""
#     power_generation: float = 0.0
#     balance_control: float = 0.0
#     movement_efficiency: float = 0.0
#     coordination: float = 0.0
#     technical_mastery: float = 0.0

# class BiomechanicalAnalyzer:
#     """Analyze biomechanical aspects of fencing movements"""
#     def analyze_movement(self, frame_data: Dict) -> BiomechanicalMetrics:
#         metrics = BiomechanicalMetrics()
#         return metrics

# class MovementAnalyzer:
#     """Analyze movement sequences in video frames"""
#     def analyze_sequence(self, angle_changes: Dict[str, List[float]]) -> Dict[str, Any]:
#         sequence_type = self._identify_sequence(angle_changes)
#         phase = self._identify_phase(angle_changes)
#         confidence = self._calculate_confidence(angle_changes)
#         context = self._generate_temporal_context(sequence_type, phase, angle_changes)
#         
#         return {
#             'sequence_type': sequence_type,
#             'phase': phase,
#             'confidence': confidence,
#             'context': context
#         }
#     
#     def _identify_sequence(self, angle_changes: Dict[str, List[float]]) -> str:
#         # Simplified sequence identification
#         total_change = sum(abs(sum(changes)) for changes in angle_changes.values())
#         if total_change > 100:
#             return "lunge"
#         elif total_change > 50:
#             return "advance"
#         else:
#             return "unknown"
#     
#     def _identify_phase(self, angle_changes: Dict[str, List[float]]) -> str:
#         # Simplified phase identification
#         recent_changes = [changes[-3:] for changes in angle_changes.values() if len(changes) >= 3]
#         if not recent_changes:
#             return "unknown"
#         
#         avg_recent_change = sum(sum(changes) for changes in recent_changes) / len(recent_changes)
#         if avg_recent_change > 5:
#             return "preparation"
#         elif avg_recent_change < -5:
#             return "recovery"
#         else:
#             return "execution"
#     
#     def _calculate_confidence(self, angle_changes: Dict[str, List[float]]) -> float:
#         # Calculate confidence based on consistency and magnitude of changes
#         total_change = sum(abs(sum(changes)) for changes in angle_changes.values())
#         consistency = 1.0 - (
#             sum(abs(max(changes, default=0) - min(changes, default=0)) 
#                 for changes in angle_changes.values()) / (total_change + 1e-6)
#         )
#         return min(1.0, max(0.0, consistency))
#     
#     def _generate_temporal_context(self, sequence_type: str, phase: str, 
#                                  angle_changes: Dict[str, List[float]]) -> str:
#         # Generate natural language description of movement context
#         if sequence_type == "unknown":
#             return "Movement not recognized"
#         
#         magnitude = sum(abs(sum(changes)) for changes in angle_changes.values())
#         speed = "quickly" if magnitude > 50 else "steadily" if magnitude > 20 else "slowly"
#         
#         return f"Performing a {sequence_type} {speed}, currently in {phase} phase"

class VideoProcessor:
    """Process video files and extract frames"""
    def __init__(self):
        self.frames = []
        self.frame_captions = []
        
    def process_frame(self, frame: np.ndarray, frame_count: int, rag_system: GardeRAG, frame_storage: FrameStorage) -> Optional[str]:
        """Process a single frame and return its caption"""
        try:
            # Save frame to storage
            frame_obj = frame_storage.save_frame(frame, frame_count, time.time())
            
            # Generate caption using GardeRAG - use frame_obj.path instead of frame_obj
            analysis = rag_system.analyze_frame(str(frame_obj.path))
            caption = analysis.get('description', 'No caption available')
            logger.info(f"Generated caption for frame {frame_count}: {caption}")
            return caption
        except Exception as e:
            logger.error(f"Error processing frame {frame_count}: {str(e)}")
            return f"Error: {str(e)}"

# ------------------------------------------------------------------------------
# ROUTES
# ------------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    global video_processor, frame_storage, rag_system
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file uploaded'}), 400
            
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        # Save the uploaded file with absolute path
        upload_folder = os.path.abspath(app.config['UPLOAD_FOLDER'])
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
            
        video_path = os.path.join(upload_folder, video_file.filename)
        video_file.save(video_path)
        
        logger.info(f"Saved video file to: {video_path}")
        
        try:
            # Initialize video processing
            video_processor = VideoProcessor()
            
            # Process video and store frames
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
                
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"Processing video: {video_path}")
            logger.info(f"FPS: {fps}, Total frames: {total_frames}")
            
            # More aggressive frame sampling - 1 frame every 3 seconds
            sample_rate = int(fps * 3)
            target_frames = min(8, total_frames // sample_rate)  # Limit to 8 frames max for better memory management
            
            # Calculate actual sample rate to get target number of frames
            sample_rate = max(int(total_frames / target_frames), 1)
            
            logger.info(f"Processing video with {total_frames} total frames, sampling every {sample_rate} frames")
            
            # Downscale frames for faster processing - using 448x448 as recommended for CLIP
            target_size = (448, 448)  # Match CLIP model's expected size
            
            frame_count = 0
            frames = []
            frame_captions = []
            
            # Process frames in batches for better memory efficiency
            batch_size = 2  # Process 2 frames at a time
            current_batch = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_count % sample_rate == 0:
                    # Resize frame
                    frame = cv2.resize(frame, target_size)
                    current_batch.append((frame, frame_count))
                    
                    # Process batch when full
                    if len(current_batch) >= batch_size:
                        for batch_frame, batch_count in current_batch:
                            frames.append(batch_frame)
                            caption = video_processor.process_frame(batch_frame, batch_count, rag_system, frame_storage)
                            frame_captions.append(caption)
                        current_batch = []
                        
                        # Break if we have enough frames
                        if len(frames) >= target_frames:
                            break
                    
                frame_count += 1
                
            # Process remaining frames in the last batch
            if current_batch and len(frames) < target_frames:
                for batch_frame, batch_count in current_batch:
                    frames.append(batch_frame)
                    caption = video_processor.process_frame(batch_frame, batch_count, rag_system, frame_storage)
                    frame_captions.append(caption)
                    
            cap.release()
            
            # Store processed frames and captions
            video_processor.frames = frames
            video_processor.frame_captions = frame_captions
            
            logger.info(f"Successfully processed {len(frames)} frames with {len(frame_captions)} captions")
            
            return jsonify({
                'message': 'Video processed successfully',
                'total_frames': len(frames),
                'total_captions': len(frame_captions)
            })
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            return jsonify({'error': f'Error processing video: {str(e)}'}), 500
            
    except Exception as e:
        logger.error(f"Error in upload: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    global video_processor, rag_system, frame_storage
    try:
        if not video_processor:
            return jsonify({'error': 'Please upload a video first'}), 400
            
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'No query provided'}), 400
            
        query = data['query']
        current_frame = data.get('current_frame', 0)
        
        logger.info(f"Processing chat query: {query} for frame {current_frame}")
        
        try:
            # Get response from RAG system
            response = rag_system.chat(query, frame_index=current_frame)
            logger.info(f"Generated response: {response}")
            
            if isinstance(response, str):
                return jsonify({'response': response})
            else:
                return jsonify({'error': 'Invalid response format'}), 500
                
        except Exception as e:
            logger.error(f"Error in chat processing: {str(e)}")
            return jsonify({'error': f'Error processing chat: {str(e)}'}), 500
            
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_frame/<int:frame_idx>')
def get_frame(frame_idx):
    global video_processor
    try:
        if not video_processor or frame_idx >= len(video_processor.frames):
            return jsonify({'error': 'Invalid frame index'}), 400
            
        frame = video_processor.frames[frame_idx]
        
        # Get caption for this frame
        caption = "No caption available"
        if hasattr(video_processor, 'frame_captions') and frame_idx < len(video_processor.frame_captions):
            caption = video_processor.frame_captions[frame_idx]
            logger.info(f"Retrieved caption for frame {frame_idx}: {caption}")
        else:
            logger.warning(f"No caption available for frame {frame_idx}")
        
        try:
            # Convert frame to base64
            if isinstance(frame, Image.Image):
                img_byte_arr = io.BytesIO()
                frame.save(img_byte_arr, format='JPEG')
                img_byte_arr = img_byte_arr.getvalue()
                frame_b64 = base64.b64encode(img_byte_arr).decode('utf-8')
            else:
                _, buffer = cv2.imencode('.jpg', frame)
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
            
            return jsonify({
                'frame': frame_b64,
                'caption': caption,
                'frame_number': frame_idx,
                'total_frames': len(video_processor.frames)
            })
            
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            return jsonify({'error': 'Error processing frame'}), 500
            
    except Exception as e:
        logger.error(f"Error getting frame: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/view_video')
def view_video():
    """Render the video viewer page"""
    if not video_processor:
        return render_template('index.html', error="Please upload a video first")
    return render_template('viewer.html', total_frames=len(video_processor.frames))

if __name__ == '__main__':
    app.run(debug=True, port=5000) 