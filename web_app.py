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
            
            # Maintain aspect ratio while resizing frames - target a wider view
            def resize_maintaining_aspect_ratio(frame, max_width=800, max_height=600):
                height, width = frame.shape[:2]
                
                # Calculate scaling factor to maintain aspect ratio
                scale_width = max_width / width
                scale_height = max_height / height
                scale = min(scale_width, scale_height)
                
                # Calculate new dimensions
                new_width = int(width * scale)
                new_height = int(height * scale)
                
                # Resize frame
                resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
                return resized
            
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
                    # Resize frame maintaining aspect ratio - store the properly sized frame
                    frame = resize_maintaining_aspect_ratio(frame)
                    
                    # Save frame to storage and get Frame object
                    frame_obj = frame_storage.save_frame(frame, frame_count, time.time())
                    frames.append(frame)  # Store the display frame
                    
                    # Create a copy for model processing that needs square input
                    model_frame = cv2.resize(frame, (448, 448), interpolation=cv2.INTER_AREA)
                    
                    # Generate caption using the model frame
                    analysis = rag_system.analyze_frame(str(frame_obj.path))
                    caption = analysis.get('description', 'No caption available')
                    frame_captions.append(caption)
                    
                    logger.info(f"Processed frame {frame_count} with caption: {caption}")
                    
                    # Break if we have enough frames
                    if len(frames) >= target_frames:
                        break
                    
                frame_count += 1

            cap.release()
            
            # Store processed frames and captions
            video_processor.frames = frames
            video_processor.frame_captions = frame_captions
            
            # Store frame objects in frame_storage
            frame_storage.store_frames([frame_storage.get_frame(i) for i in range(len(frames))])
            
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
        if video_processor is None or len(getattr(video_processor, 'frames', [])) == 0:
            return jsonify({'response': "Please upload a video first. I need to analyze the fencing techniques before I can answer your questions."}), 400
            
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'No query provided'}), 400
            
        query = data['query']
        current_frame_idx = data.get('current_frame', 0)
        
        logger.info(f"Processing chat query: {query} for frame {current_frame_idx}")
        
        try:
            # If asking about the whole video
            if "video" in query.lower() or "happening" in query.lower():
                # Get all frames and their captions
                frames = video_processor.frames
                captions = video_processor.frame_captions
                
                if not frames or not captions:
                    return jsonify({'response': "I haven't analyzed any frames from this video yet."}), 400
                
                # Create a comprehensive prompt for video analysis
                prompt = "<image>\nAs a fencing expert, analyze this sequence of frames focusing on:\n"
                prompt += "1. Technical execution and form\n"
                prompt += "2. Movement progression and timing\n"
                prompt += "3. Tactical implications\n\n"
                prompt += f"User's question: {query}"
                
                # Analyze current frame for detailed context
                if 0 <= current_frame_idx < len(frames):
                    frame = frames[current_frame_idx]
                    # Create model-ready frame
                    model_frame = cv2.resize(frame, (448, 448), interpolation=cv2.INTER_AREA)
                    # Save temporary frame
                    temp_frame_path = os.path.join(frame_storage.session_dir, f"temp_analysis_{current_frame_idx}.jpg")
                    cv2.imwrite(temp_frame_path, model_frame)
                    
                    # Get detailed analysis
                    analysis = rag_system.analyze_frame(
                        frame_path=temp_frame_path,
                        prompt=prompt
                    )
                    
                    # Format comprehensive response
                    response = "Based on my analysis of the fencing sequence:\n\n"
                    
                    # Add frame-by-frame progression
                    response += "Movement Progression:\n"
                    for i, caption in enumerate(captions):
                        response += f"Frame {i}: {caption}\n"
                    
                    # Add detailed current frame analysis
                    if isinstance(analysis, dict) and 'description' in analysis:
                        response += f"\nDetailed Analysis (Frame {current_frame_idx}):\n"
                        response += analysis['description']
                    
                    # Clean up temporary file
                    if os.path.exists(temp_frame_path):
                        os.remove(temp_frame_path)
                    
                    return jsonify({'response': response})
                
            # For specific frame analysis
            elif 0 <= current_frame_idx < len(video_processor.frames):
                frame = video_processor.frames[current_frame_idx]
                # Create model-ready frame
                model_frame = cv2.resize(frame, (448, 448), interpolation=cv2.INTER_AREA)
                # Save temporary frame
                temp_frame_path = os.path.join(frame_storage.session_dir, f"temp_analysis_{current_frame_idx}.jpg")
                cv2.imwrite(temp_frame_path, model_frame)
                
                # Create detailed prompt for specific analysis
                prompt = f"<image>\nAs a fencing expert, analyze this frame focusing on:\n"
                prompt += "1. Technical execution and form\n"
                prompt += "2. Position and footwork\n"
                prompt += "3. Tactical implications\n\n"
                prompt += f"User's question: {query}"
                
                # Get detailed analysis
                analysis = rag_system.analyze_frame(
                    frame_path=temp_frame_path,
                    prompt=prompt
                )
                
                # Clean up temporary file
                if os.path.exists(temp_frame_path):
                    os.remove(temp_frame_path)
                
                if isinstance(analysis, dict) and 'description' in analysis:
                    return jsonify({'response': analysis['description']})
            
            return jsonify({'error': 'Could not analyze the frame(s)'}), 400
                
        except Exception as e:
            logger.error(f"Error in chat processing: {str(e)}", exc_info=True)
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