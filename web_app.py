from flask import Flask, request, jsonify, render_template, send_file
import os
import logging
import cv2
import base64
import numpy as np
from PIL import Image
import io
import requests
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import time
import asyncio
import aiohttp
import subprocess
from pathlib import Path
from werkzeug.utils import secure_filename

# Import ML models
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
from pose_estimator import PoseEstimator, JointAngles
from src.app.model import MiniCPMVProcessor

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
video_processor = None  # Global VideoProcessor instance

# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # Change to WARNING to suppress most logs
    format='%(message)s',  # Simplified format
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress verbose logs from other modules
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("clip_model").setLevel(logging.ERROR)
logging.getLogger("llama_model").setLevel(logging.ERROR)
logging.getLogger("werkzeug").setLevel(logging.ERROR)  # Suppress Flask logs
logging.getLogger("PIL").setLevel(logging.ERROR)  # Suppress PIL logs
logging.getLogger("tqdm").setLevel(logging.ERROR)  # Suppress tqdm logs
logging.getLogger("absl").setLevel(logging.ERROR)  # Suppress TensorFlow logs

# Disable TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging

# Ensure upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Initialize ML models
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
sent_transformer = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# Initialize model
model_processor = MiniCPMVProcessor()

# ------------------------------------------------------------------------------
# MINICPMV INTEGRATION
# ------------------------------------------------------------------------------

class MiniCPMVProcessor:
    """Handle MiniCPM-V model integration for video captioning and chat"""
    def __init__(self, model_path: str = "MiniCPM-V-2_6-gguf/ggml-model-Q4_K_M.gguf",
                 mmproj_path: str = "MiniCPM-V-2_6-gguf/mmproj-model-f16.gguf"):
        self.model_path = model_path
        self.mmproj_path = mmproj_path
        self.cli_path = "/home/ubuntu/GSX-a1/llama.cpp/build/bin/llama-minicpmv-cli"
        
        # Optimized parameters
        self.temp = 0.1
        self.top_p = 0.7
        self.top_k = 40
        self.repeat_penalty = 1.1
        self.context_size = 2048
        self.n_threads = max(1, os.cpu_count() // 2)
        
        # Create temp directory
        self.temp_dir = os.path.join(os.getcwd(), 'temp_frames')
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir, exist_ok=True)
        
        # Ensure model files exist
        if not os.path.exists(model_path):
            raise ValueError(f"Model file not found: {model_path}")
        if not os.path.exists(mmproj_path):
            raise ValueError(f"MMPROJ file not found: {mmproj_path}")
        if not os.path.exists(self.cli_path):
            raise ValueError(f"CLI executable not found: {self.cli_path}")
    
    def process_frames(self, frames: List[np.ndarray], system_prompt: str = None, query: str = "Describe these frames in detail.") -> str:
        """Process multiple frames with MiniCPM-V"""
        temp_paths = []
        try:
            print(f"\nMiniCPMV processing {len(frames)} frames")
            print(f"CLI path: {self.cli_path}")
            print(f"Model path: {self.model_path}")
            print(f"MMPROJ path: {self.mmproj_path}")
            
            # Save frames as temporary images
            for i, frame in enumerate(frames):
                temp_path = os.path.join(self.temp_dir, f"temp_frame_{int(time.time() * 1000)}_{i}.jpg")
                print(f"Saving frame {i} to {temp_path}")
                success = cv2.imwrite(temp_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                                    [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                if not success or not os.path.exists(temp_path):
                    print(f"Failed to save frame {i}")
                    continue
                    
                temp_paths.append(temp_path)
            
            if not temp_paths:
                print("No frames were saved successfully")
                return "Error: Failed to save frames"
            
            print(f"Successfully saved {len(temp_paths)} frames")
            
            # Construct command with multi-image support
            cmd = [
                self.cli_path,
                "-m", os.path.abspath(self.model_path),
                "--mmproj", os.path.abspath(self.mmproj_path),
                "-t", str(self.n_threads),
                "-n", "256",  # Increased token limit for multi-image
                "--temp", str(self.temp),
                "--ctx-size", "8192",  # Increased context for multi-image
                "--top-p", str(self.top_p),
                "--top-k", str(self.top_k)
            ]
            
            # Add system prompt if provided
            if system_prompt:
                cmd.extend(["--system", system_prompt])
            
            # Add images
            for temp_path in temp_paths:
                cmd.extend(["--image", os.path.abspath(temp_path)])
            
            # Add query
            cmd.extend(["-p", query])
            
            print("\nExecuting command:")
            print(" ".join(cmd))
            
            # Run the command with retry mechanism
            max_retries = 3
            retry_count = 0
            while retry_count < max_retries:
                try:
                    print(f"\nAttempt {retry_count + 1}/{max_retries}")
                    result = subprocess.run(
                        cmd, 
                        capture_output=True, 
                        text=True, 
                        timeout=60,  # Increased timeout for multi-image
                        cwd=os.getcwd()
                    )
                    
                    print(f"Return code: {result.returncode}")
                    print("Stdout:", result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)
                    print("Stderr:", result.stderr[:500] + "..." if len(result.stderr) > 500 else result.stderr)
                    
                    # Process output
                    if result.returncode == 0 and result.stdout:
                        response = result.stdout.strip()
                        # Clean up the response
                        response = ' '.join(
                            line.strip() for line in response.split('\n') 
                            if line.strip() 
                            and not any(x in line.lower() for x in 
                                      ['loading', 'llama_print', 'log start', 'clip_model', 
                                       'user:', 'assistant:', '<user>', '<assistant>'])
                        )
                        print(f"\nProcessed response: {response[:200]}...")
                        return response if response else "Error: No caption generated"
                    else:
                        retry_count += 1
                        if retry_count < max_retries:
                            print(f"Command failed, retrying in 1 second...")
                            time.sleep(1)  # Wait before retrying
                            continue
                        return f"Error: Command failed with code {result.returncode}"
                        
                except subprocess.TimeoutExpired:
                    print("Command timed out")
                    retry_count += 1
                    if retry_count < max_retries:
                        print(f"Timeout, retrying in 1 second...")
                        time.sleep(1)  # Wait before retrying
                        continue
                    return "Error: Processing timeout after retries"
                    
        except Exception as e:
            print(f"Error in process_frames: {str(e)}")
            return f"Error: {str(e)}"
            
        finally:
            # Clean up temp files
            print("\nCleaning up temporary files...")
            for temp_path in temp_paths:
                try:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                        print(f"Removed {temp_path}")
                except Exception as e:
                    print(f"Failed to remove {temp_path}: {str(e)}")
    
    def process_frame(self, frame: np.ndarray, query: str = "Describe this image briefly.") -> str:
        """Process a single frame with MiniCPM-V (legacy method)"""
        return self.process_frames([frame], query=query)

# ------------------------------------------------------------------------------
# BIOMECHANICAL ANALYSIS
# ------------------------------------------------------------------------------

@dataclass
class BiomechanicalMetrics:
    """Store biomechanical analysis results"""
    power_generation: float = 0.0
    balance_control: float = 0.0
    movement_efficiency: float = 0.0
    coordination: float = 0.0
    technical_mastery: float = 0.0

class BiomechanicalAnalyzer:
    """Advanced biomechanical analysis for fencing movements"""
    def __init__(self):
        self.kinetic_chains = {
            'power_generation': ['back_leg', 'hip_rotation', 'torso', 'sword_arm'],
            'balance_control': ['stance_width', 'hip_alignment', 'torso_angle'],
            'movement_efficiency': ['joint_coordination', 'momentum_transfer']
        }
        
    def analyze_movement(self, frame_data: Dict[str, Any]) -> BiomechanicalMetrics:
        """Analyze movement biomechanics from frame data"""
        metrics = BiomechanicalMetrics()
        
        # Analyze power generation through kinetic chain
        if 'angles' in frame_data:
            angles = frame_data['angles']
            metrics.power_generation = self._analyze_power_chain(angles)
            metrics.balance_control = self._analyze_balance(angles)
            metrics.movement_efficiency = self._analyze_efficiency(angles)
            metrics.coordination = self._analyze_coordination(angles)
            
            # Calculate overall technical mastery
            metrics.technical_mastery = np.mean([
                metrics.power_generation,
                metrics.balance_control,
                metrics.movement_efficiency,
                metrics.coordination
            ])
        
        return metrics
    
    def _analyze_power_chain(self, angles: Dict[str, float]) -> float:
        """Analyze power generation through kinetic chain"""
        # Check proper sequence activation (proximal to distal)
        power_scores = []
        ideal_sequence = ['back_knee', 'front_knee', 'torso', 'sword_arm']
        
        for i, joint in enumerate(ideal_sequence[:-1]):
            if joint in angles and ideal_sequence[i+1] in angles:
                # Check if proximal joint moves before distal
                score = 1.0 if angles[joint] > angles[ideal_sequence[i+1]] else 0.5
                power_scores.append(score)
        
        return np.mean(power_scores) if power_scores else 0.0
    
    def _analyze_balance(self, angles: Dict[str, float]) -> float:
        """Analyze balance control"""
        balance_scores = []
        
        # Check torso alignment
        if 'torso' in angles:
            torso_score = 1.0 - min(1.0, abs(angles['torso'] - 90) / 45.0)
            balance_scores.append(torso_score)
        
        # Check knee alignment
        if 'front_knee' in angles and 'back_knee' in angles:
            knee_diff = abs(angles['front_knee'] - angles['back_knee'])
            knee_score = 1.0 - min(1.0, knee_diff / 45.0)
            balance_scores.append(knee_score)
        
        return np.mean(balance_scores) if balance_scores else 0.0
    
    def _analyze_efficiency(self, angles: Dict[str, float]) -> float:
        """Analyze movement efficiency"""
        # Check for excessive joint angles
        efficiency_scores = []
        
        if 'sword_arm' in angles:
            # Penalize over-extension or under-extension
            sword_efficiency = 1.0 - min(1.0, abs(angles['sword_arm'] - 180) / 90.0)
            efficiency_scores.append(sword_efficiency)
        
        if 'front_knee' in angles:
            # Optimal front knee angle around 90-100 degrees
            knee_efficiency = 1.0 - min(1.0, abs(angles['front_knee'] - 95) / 45.0)
            efficiency_scores.append(knee_efficiency)
        
        return np.mean(efficiency_scores) if efficiency_scores else 0.0
    
    def _analyze_coordination(self, angles: Dict[str, float]) -> float:
        """Analyze inter-joint coordination"""
        coordination_scores = []
        
        # Check sword arm and front knee coordination
        if 'sword_arm' in angles and 'front_knee' in angles:
            # Ideal relationship: as sword extends, knee bends
            coord_score = 1.0 if (angles['sword_arm'] > 160 and angles['front_knee'] < 110) else 0.5
            coordination_scores.append(coord_score)
        
        # Check torso and leg coordination
        if 'torso' in angles and 'front_knee' in angles and 'back_knee' in angles:
            torso_coord = 1.0 if (80 <= angles['torso'] <= 100) else 0.5
            coordination_scores.append(torso_coord)
        
        return np.mean(coordination_scores) if coordination_scores else 0.0

# ------------------------------------------------------------------------------
# MOVEMENT SEQUENCE ANALYSIS
# ------------------------------------------------------------------------------

class MovementAnalyzer:
    """Advanced movement sequence analysis"""
    def __init__(self, memory_window: int = 5):
        self.memory_window = memory_window
        self.movement_history = []
        self.phase_transitions = []
        
        # Define movement phases and their characteristics
        self.movement_phases = {
            'preparation': {
                'sword_arm_range': (130, 150),
                'front_knee_range': (140, 160),
                'characteristics': ['stable_base', 'loaded_position']
            },
            'extension': {
                'sword_arm_range': (170, 180),
                'front_knee_range': (80, 100),
                'characteristics': ['forward_drive', 'arm_extension']
            },
            'recovery': {
                'sword_arm_range': (130, 150),
                'front_knee_range': (140, 160),
                'characteristics': ['weight_shift', 'reset_position']
            }
        }
    
    def analyze_sequence(self, frame_window: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze movement sequence from frame window"""
        if not frame_window:
            return {
                "sequence": "unknown",
                "phase": "unknown",
                "confidence": 0.0,
                "temporal_context": "No movement data"
            }
        
        # Update movement history
        self.movement_history = frame_window[-self.memory_window:]
        
        # Detect current phase
        current_phase = self._detect_movement_phase(frame_window[-1])
        
        # Analyze phase transition
        if self.phase_transitions and current_phase != self.phase_transitions[-1]:
            self.phase_transitions.append(current_phase)
        elif not self.phase_transitions:
            self.phase_transitions.append(current_phase)
        
        # Calculate confidence based on characteristic matches
        confidence = self._calculate_phase_confidence(frame_window[-1], current_phase)
        
        # Determine sequence type
        sequence_type = self._determine_sequence_type()
        
        return {
            "sequence": sequence_type,
            "phase": current_phase,
            "confidence": confidence,
            "temporal_context": self._get_temporal_context()
        }
    
    def _detect_movement_phase(self, frame_data: Dict[str, Any]) -> str:
        """Detect current movement phase"""
        if 'angles' not in frame_data:
            return "unknown"
        
        angles = frame_data['angles']
        phase_scores = {}
        
        for phase, characteristics in self.movement_phases.items():
            score = 0
            total_checks = 0
            
            # Check sword arm angle
            if 'sword_arm' in angles:
                min_angle, max_angle = characteristics['sword_arm_range']
                if min_angle <= angles['sword_arm'] <= max_angle:
                    score += 1
                total_checks += 1
            
            # Check front knee angle
            if 'front_knee' in angles:
                min_angle, max_angle = characteristics['front_knee_range']
                if min_angle <= angles['front_knee'] <= max_angle:
                    score += 1
                total_checks += 1
            
            phase_scores[phase] = score / total_checks if total_checks > 0 else 0
        
        # Return phase with highest score
        return max(phase_scores.items(), key=lambda x: x[1])[0]
    
    def _calculate_phase_confidence(self, frame_data: Dict[str, Any], phase: str) -> float:
        """Calculate confidence in phase detection"""
        if phase == "unknown" or phase not in self.movement_phases:
            return 0.0
        
        if 'angles' not in frame_data:
            return 0.0
        
        angles = frame_data['angles']
        characteristics = self.movement_phases[phase]
        confidence_scores = []
        
        # Check angle ranges
        for joint in ['sword_arm', 'front_knee']:
            if joint in angles:
                min_angle, max_angle = characteristics[f'{joint}_range']
                angle = angles[joint]
                # Calculate how close the angle is to the ideal range
                if min_angle <= angle <= max_angle:
                    confidence_scores.append(1.0)
                else:
                    distance = min(abs(angle - min_angle), abs(angle - max_angle))
                    confidence_scores.append(1.0 - min(1.0, distance / 45.0))
        
        return np.mean(confidence_scores) if confidence_scores else 0.0
    
    def _determine_sequence_type(self) -> str:
        """Determine the type of movement sequence"""
        if len(self.phase_transitions) < 2:
            return "position_hold"
        
        # Look for specific sequences
        recent_phases = self.phase_transitions[-3:]
        
        if "preparation" in recent_phases and "extension" in recent_phases:
            return "lunge"
        elif "extension" in recent_phases and "recovery" in recent_phases:
            return "recovery"
        else:
            return "transition"
    
    def _get_temporal_context(self) -> str:
        """Generate temporal context description"""
        if len(self.phase_transitions) < 2:
            return "Initial position"
        
        transitions = " -> ".join(self.phase_transitions[-3:])
        return f"Movement sequence: {transitions}"

# ------------------------------------------------------------------------------
# VIDEO PROCESSING
# ------------------------------------------------------------------------------

class VideoProcessor:
    """Process video files and extract frames"""
    _instance = None  # Singleton instance
    
    def __new__(cls):
        if cls._instance is None:
            print("Creating new VideoProcessor instance")
            cls._instance = super(VideoProcessor, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        print("Initializing VideoProcessor")
        self.frames = []
        self.frame_captions = []
        self.processing_complete = False
        
        # Enhanced batch processing settings
        self.batch_size = 16  # Process more frames at once
        self.max_workers = min(32, (os.cpu_count() or 1) * 2)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Create temp directory
        self.temp_dir = os.path.join(os.getcwd(), 'temp_frames')
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir, exist_ok=True)
            
        # Initialize model and tokenizer
        try:
            from src.app.model import load_model_and_tokenizer
            self.model, self.tokenizer = load_model_and_tokenizer()
            print("Model and tokenizer initialized successfully")
            self._initialized = True
        except Exception as e:
            print(f"Error initializing model: {str(e)}")
            raise
    
    def _process_frame_batch(self, frames_data):
        """Process a batch of frames using multi-image inference"""
        try:
            print(f"\nProcessing batch of {len(frames_data)} frames")
            
            # Preprocess frames
            processed_frames = []
            frame_indices = []
            
            for frame, frame_idx in frames_data:
                print(f"Preprocessing frame {frame_idx}")
                frame_processed = self._preprocess_frame(frame)
                if frame_processed is not None:
                    # Convert to PIL Image for model input
                    pil_image = Image.fromarray(frame_processed)
                    processed_frames.append(pil_image)
                    frame_indices.append(frame_idx)
            
            if not processed_frames:
                print("No frames were preprocessed successfully")
                return []
            
            print(f"Successfully preprocessed {len(processed_frames)} frames")
            
            # Example few-shot prompts for better context
            few_shot_examples = [
                {
                    "question": "What is happening in this sequence?",
                    "answer": "The sequence shows a continuous movement or action across multiple frames."
                },
                {
                    "question": "Describe any changes between frames.",
                    "answer": "There are subtle changes in position and movement between consecutive frames."
                }
            ]
            
            # Generate captions using multi-image inference
            from src.app.model import generate_response
            captions = generate_response(
                self.model,
                self.tokenizer,
                processed_frames,
                "Describe these frames in sequence, noting any changes or movements between them.",
                few_shot_examples=few_shot_examples
            )
            
            if not captions:
                print("Failed to generate captions")
                return []
            
            print(f"\nReceived caption response: {captions[:200]}...")
            
            # Split into individual frame captions
            try:
                frame_captions = captions.split('\n')
                print(f"Split into {len(frame_captions)} individual captions")
                
                # Ensure we have enough captions
                while len(frame_captions) < len(processed_frames):
                    frame_captions.append("No specific description available")
                
                # Trim excess captions
                if len(frame_captions) > len(processed_frames):
                    frame_captions = frame_captions[:len(processed_frames)]
                
            except Exception as e:
                print(f"Error splitting captions: {str(e)}")
                frame_captions = [captions] * len(processed_frames)
            
            # Return results
            results = list(zip(processed_frames, frame_captions))
            print(f"Returning {len(results)} frame-caption pairs")
            return results
            
        except Exception as e:
            print(f"\nError in _process_frame_batch: {str(e)}")
            return []
    
    def process_video(self, video_path: str, target_fps: int = 5) -> None:
        """Process video file and extract frames"""
        try:
            print("\nProcessing video frames...")
            
            # Reset state
            self.frames = []
            self.frame_captions = []
            self.processing_complete = False
            
            # Open video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Failed to open video file")
            
            # Get video properties
            orig_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate frame sampling interval
            sample_interval = max(1, int(orig_fps / target_fps))
            frame_indices = list(range(0, total_frames, sample_interval))
            total_frames_to_process = len(frame_indices)
            
            # Process frames in batches
            frame_batches = []
            current_batch = []
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                
                current_batch.append((frame, frame_idx))
                if len(current_batch) >= self.batch_size:
                    frame_batches.append(current_batch)
                    current_batch = []
            
            if current_batch:
                frame_batches.append(current_batch)
            
            # Process batches in parallel
            processed_frames = 0
            for batch in frame_batches:
                results = self._process_frame_batch(batch)
                for frame, caption in results:
                    self.frames.append(frame)
                    self.frame_captions.append(caption)
                    processed_frames += 1
                    print(f"\rProcessed {processed_frames}/{total_frames_to_process} frames ✓", end="", flush=True)
            
            cap.release()
            print("\nVideo processing completed!")
            print(f"Total frames processed: {len(self.frames)}")
            print(f"Total captions generated: {len(self.frame_captions)}")
            print(f"Success rate: {(len(self.frame_captions) / total_frames_to_process) * 100:.1f}%")
            
            self.processing_complete = True
            
        except Exception as e:
            print(f"\nError processing video: {str(e)}")
            raise
        finally:
            self._cleanup_temp_files()

# ------------------------------------------------------------------------------
# ROUTES
# ------------------------------------------------------------------------------

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        if file and allowed_file(file.filename):
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process frames
            frames = extract_frames(filepath)
            if not frames:
                return jsonify({'error': 'Failed to extract frames'}), 500
                
            # Generate caption
            caption = model_processor.process_frames(frames)
            if caption.startswith('Error:'):
                return jsonify({'error': caption}), 500
                
            return jsonify({
                'caption': caption,
                'filename': filename
            })
            
        return jsonify({'error': 'Invalid file type'}), 400
        
    except Exception as e:
        app.logger.error(f"Error in upload_file: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    global video_processor
    try:
        print("\n=== Chat Request ===")
        print(f"Video processor exists: {video_processor is not None}")
        print(f"Video processor initialized: {video_processor._initialized if video_processor else False}")
        print(f"Processing complete: {video_processor.processing_complete if video_processor else False}")
        print(f"Total frames: {len(video_processor.frames) if video_processor else 0}")
        print(f"Total captions: {len(video_processor.frame_captions) if video_processor else 0}")
        
        if not video_processor or not video_processor._initialized:
            print("Error: Video processor not initialized")
            return jsonify({'error': 'Please upload a video first'}), 400
            
        if not video_processor.processing_complete:
            print("Error: Video still processing")
            return jsonify({'error': 'Video is still processing'}), 400
            
        data = request.get_json()
        print(f"Request data: {data}")
        
        if not data or 'query' not in data:
            print("Error: No query provided")
            return jsonify({'error': 'No query provided'}), 400
            
        query = data['query']
        current_frame = data.get('current_frame', 0)
        print(f"Processing query: '{query}' for frame {current_frame}")
        
        # Ensure current_frame is within bounds
        if current_frame >= len(video_processor.frames):
            print(f"Adjusting frame index from {current_frame} to {len(video_processor.frames) - 1}")
            current_frame = len(video_processor.frames) - 1
            
        # Get frames for context window
        frame_context = []
        start_idx = max(0, current_frame - 2)
        end_idx = min(len(video_processor.frames), current_frame + 3)
        
        print(f"Getting context frames {start_idx} to {end_idx}")
        
        # Convert frames to PIL Images
        for i in range(start_idx, end_idx):
            frame = video_processor.frames[i]
            if isinstance(frame, np.ndarray):
                frame = Image.fromarray(frame)
            frame_context.append(frame)
        
        print(f"Got {len(frame_context)} context frames")
        
        # Create few-shot examples based on previous captions
        few_shot_examples = []
        if len(video_processor.frame_captions) > 0:
            # Add example from start of video
            few_shot_examples.append({
                "question": "What is happening at the start of the video?",
                "answer": video_processor.frame_captions[0]
            })
            
            # Add example from middle of video
            mid_idx = len(video_processor.frame_captions) // 2
            few_shot_examples.append({
                "question": "What is happening in the middle of the video?",
                "answer": video_processor.frame_captions[mid_idx]
            })
            
            # Add temporal context example
            if current_frame > 0:
                few_shot_examples.append({
                    "question": "What changed from the previous frame?",
                    "answer": f"From {video_processor.frame_captions[current_frame-1]} to {video_processor.frame_captions[current_frame]}"
                })
        
        # Generate response using multi-image inference
        from src.app.model import generate_response
        response = generate_response(
            video_processor.model,
            video_processor.tokenizer,
            frame_context,
            query,
            few_shot_examples=few_shot_examples
        )
        
        if not response:
            print("Failed to generate response")
            return jsonify({'error': 'Failed to generate response'}), 500
            
        print(f"Generated response: {response[:200]}...")
        return jsonify({'response': response})
            
    except Exception as e:
        print(f"Error in chat: {str(e)}")
        logger.error(f"Error in chat: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_frame/<int:frame_idx>')
def get_frame(frame_idx):
    global video_processor
    try:
        print("\n=== Get Frame Request ===")
        print(f"Requested frame index: {frame_idx}")
        print(f"Video processor exists: {video_processor is not None}")
        print(f"Video processor initialized: {video_processor._initialized if video_processor else False}")
        
        if not video_processor or not video_processor._initialized:
            print("Error: Video processor not initialized")
            return jsonify({'error': 'Please upload a video first'}), 400
            
        print(f"Total frames: {len(video_processor.frames)}")
        print(f"Total captions: {len(video_processor.frame_captions)}")
        
        if frame_idx >= len(video_processor.frames):
            print(f"Error: Frame index {frame_idx} out of bounds")
            return jsonify({'error': 'Invalid frame index'}), 400
            
        frame = video_processor.frames[frame_idx]
        caption = video_processor.frame_captions[frame_idx] if frame_idx < len(video_processor.frame_captions) else "No caption available"
        
        print(f"Frame shape: {frame.shape if isinstance(frame, np.ndarray) else 'PIL Image'}")
        print(f"Caption: {caption[:100]}...")
        
        try:
            # Convert frame to base64
            if isinstance(frame, Image.Image):
                print("Converting PIL Image to base64")
                img_byte_arr = io.BytesIO()
                frame.save(img_byte_arr, format='JPEG')
                img_byte_arr = img_byte_arr.getvalue()
                frame_b64 = base64.b64encode(img_byte_arr).decode('utf-8')
            else:
                print("Converting numpy array to base64")
                _, buffer = cv2.imencode('.jpg', frame)
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
            
            response = {
                'frame': frame_b64,
                'caption': caption,
                'frame_number': frame_idx,
                'total_frames': len(video_processor.frames)
            }
            print("Successfully prepared frame response")
            return jsonify(response)
            
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            logger.error(f"Error processing frame: {str(e)}")
            return jsonify({'error': 'Error processing frame'}), 500
            
    except Exception as e:
        print(f"Error getting frame: {str(e)}")
        logger.error(f"Error getting frame: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/view_video')
def view_video():
    """Render the video viewer page"""
    if not video_processor:
        return render_template('index.html', error="Please upload a video first")
    return render_template('viewer.html', total_frames=len(video_processor.frames))

@app.route('/test_caption', methods=['GET'])
def test_caption():
    """Test route to verify caption generation"""
    try:
        # Create a simple test image
        test_img = np.zeros((448, 448, 3), dtype=np.uint8)
        cv2.putText(test_img, "Test", (150, 225), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
        
        # Initialize processor
        processor = MiniCPMVProcessor()
        
        # Test caption generation
        caption = processor.process_frame(test_img, "What is in this image?")
        
        return jsonify({
            'status': 'success',
            'caption': caption,
            'message': 'Caption generation test completed'
        })
        
    except Exception as e:
        logger.error(f"Error in test caption: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'message': 'Caption generation test failed'
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 