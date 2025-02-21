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

# Import ML models
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
from pose_estimator import PoseEstimator, JointAngles

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
video_processor = None  # Global VideoProcessor instance

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Change to DEBUG for maximum verbosity
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Keep Flask and werkzeug logs for debugging
logging.getLogger("werkzeug").setLevel(logging.DEBUG)
logging.getLogger("flask").setLevel(logging.DEBUG)

# Only suppress specific noisy modules
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Disable TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Less verbose TF logging

# Ensure upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Initialize ML models
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
sent_transformer = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# ------------------------------------------------------------------------------
# MINICPMV INTEGRATION
# ------------------------------------------------------------------------------

class MiniCPMVProcessor:
    """Handle MiniCPM-V model integration for video captioning and chat"""
    def __init__(self):
        # Get absolute paths
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(base_dir, "MiniCPM-V-2_6-gguf")
        
        self.model_path = os.path.join(model_dir, "ggml-model-Q4_K_M.gguf")
        self.mmproj_path = os.path.join(model_dir, "mmproj-model-f16.gguf")
        self.cli_path = os.path.join(base_dir, "llama.cpp/build/bin/llama-minicpmv-cli")
        
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
        if not os.path.exists(self.model_path):
            raise ValueError(f"Model file not found: {self.model_path}")
        if not os.path.exists(self.mmproj_path):
            raise ValueError(f"MMPROJ file not found: {self.mmproj_path}")
        if not os.path.exists(self.cli_path):
            raise ValueError(f"CLI executable not found: {self.cli_path}")
            
        print(f"Model path: {self.model_path}")
        print(f"MMPROJ path: {self.mmproj_path}")
        print(f"CLI path: {self.cli_path}")
    
    def process_frames(self, frames: List[np.ndarray], system_prompt: str = None, query: str = "Describe these frames in detail.") -> str:
        """Process multiple frames with MiniCPM-V"""
        temp_paths = []
        try:
            # Save frames as temporary images
            for i, frame in enumerate(frames):
                temp_path = os.path.join(self.temp_dir, f"temp_frame_{int(time.time() * 1000)}_{i}.jpg")
                success = cv2.imwrite(temp_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                                    [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                if not success or not os.path.exists(temp_path):
                    continue
                    
                temp_paths.append(temp_path)
            
            if not temp_paths:
                return "Error: Failed to save frames"
            
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
            
            # Run the command with retry mechanism
            max_retries = 3
            retry_count = 0
            while retry_count < max_retries:
                try:
                    result = subprocess.run(
                        cmd, 
                        capture_output=True, 
                        text=True, 
                        timeout=180,  # Increased from 60 to 180 seconds (3 minutes) per frame
                        cwd=os.getcwd()
                    )
                    
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
                        return response if response else "Error: No caption generated"
                    else:
                        retry_count += 1
                        if retry_count < max_retries:
                            time.sleep(1)  # Wait before retrying
                            continue
                        return f"Error: Command failed with code {result.returncode}"
                        
                except subprocess.TimeoutExpired:
                    retry_count += 1
                    if retry_count < max_retries:
                        time.sleep(1)  # Wait before retrying
                        continue
                    return "Error: Processing timeout after retries"
                    
        except Exception as e:
            return f"Error: {str(e)}"
            
        finally:
            # Clean up temp files
            for temp_path in temp_paths:
                try:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                except:
                    pass
    
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
    """Process video files and extract frames using MiniCPM-V's approach"""
    _instance = None  # Singleton instance
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VideoProcessor, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.frames = []
        self.frame_captions = []
        self.processing_complete = False
        
        # Initialize paths
        self.base_dir = os.path.abspath(os.getcwd())
        self.model_path = os.path.join(self.base_dir, "MiniCPM-V-2_6-gguf/ggml-model-Q4_K_M.gguf")
        self.mmproj_path = os.path.join(self.base_dir, "MiniCPM-V-2_6-gguf/mmproj-model-f16.gguf")
        self.cli_path = os.path.join(self.base_dir, "llama.cpp/build/bin/llama-minicpmv-cli")
        
        # Create a more persistent temp directory
        self.temp_dir = os.path.join(self.base_dir, 'temp_frames')
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir, exist_ok=True)
        
        print(f"Model path: {self.model_path}")
        print(f"MMPROJ path: {self.mmproj_path}")
        print(f"CLI path: {self.cli_path}")
        print(f"Temp dir: {self.temp_dir}")
        
        self._initialized = True
    
    def process_video(self, video_path: str) -> None:
        """Process video file and extract frames using MiniCPM-V's approach"""
        try:
            print("\n" + "="*50)
            print("Starting video processing...")
            print("="*50)
            
            # Reset state
            self.frames = []
            self.frame_captions = []
            self.processing_complete = False
            
            # Verify paths
            print(f"Video path: {video_path}")
            print(f"Model path: {self.model_path}")
            print(f"MMPROJ path: {self.mmproj_path}")
            print(f"CLI path: {self.cli_path}")
            
            for path in [self.model_path, self.mmproj_path, self.cli_path]:
                if not os.path.exists(path):
                    raise ValueError(f"Required file not found: {path}")
            
            # Open video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            print(f"\nVideo properties:")
            print(f"- Total frames: {total_frames}")
            print(f"- FPS: {fps}")
            
            # Calculate frame sampling interval (1 frame per second)
            sample_interval = max(1, fps)
            frame_indices = list(range(0, total_frames, sample_interval))
            
            # Uniformly sample frames if too many
            MAX_FRAMES = 15  # Limit to 15 frames
            if len(frame_indices) > MAX_FRAMES:
                gap = len(frame_indices) / MAX_FRAMES
                frame_indices = [int(i * gap + gap / 2) for i in range(MAX_FRAMES)]
            
            print(f"\nFrame extraction plan:")
            print(f"- Frame interval: {sample_interval}")
            print(f"- Selected frames: {len(frame_indices)}")
            
            # Extract frames
            print("\nExtracting frames...")
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frames.append(frame_rgb)
                print(f"Extracted frame {len(self.frames)} at time {frame_idx/fps:.2f}s")
            
            cap.release()
            print(f"\nExtracted {len(self.frames)} frames")
            
            # Process frames one by one
            print("\nProcessing frames with MiniCPM-V...")
            for i, frame in enumerate(self.frames):
                print(f"\nProcessing frame {i+1}/{len(self.frames)}")
                temp_path = os.path.join(self.temp_dir, f"frame_{i+1}.jpg")
                
                try:
                    # Save frame as temporary file with high quality
                    success = cv2.imwrite(temp_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                                        [cv2.IMWRITE_JPEG_QUALITY, 100])
                    if not success or not os.path.exists(temp_path):
                        print(f"Warning: Failed to save frame {i+1}")
                        self.frame_captions.append("Failed to save frame")
                        continue
                    
                    # Verify file exists and has content
                    if not os.path.getsize(temp_path) > 0:
                        print(f"Warning: Empty frame file for frame {i+1}")
                        self.frame_captions.append("Empty frame file")
                        continue
                    
                    # Construct command with absolute paths
                    cmd = [
                        os.path.abspath(self.cli_path),
                        "-m", os.path.abspath(self.model_path),
                        "--mmproj", os.path.abspath(self.mmproj_path),
                        "-t", "4",
                        "-n", "512",
                        "--temp", "0.1",
                        "--ctx-size", "2048",
                        "--repeat-penalty", "1.1",
                        "--top-p", "0.7",
                        "--top-k", "40",
                        "-ngl", "0",  # Disable GPU layers
                        "--image", os.path.abspath(temp_path),
                        "-p", "Describe what is happening in this frame."
                    ]
                    
                    print("Running command:", " ".join(cmd))
                    
                    # Run command with increased timeout
                    logger.info(f"Running command for frame {i+1}")
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=180,  # 2 minutes timeout
                        check=False,  # Don't raise exception on non-zero exit
                        cwd=self.base_dir  # Run from base directory
                    )
                    
                    # Always log command output
                    if result.stdout:
                        logger.info(f"Command stdout for frame {i+1}:\n{result.stdout}")
                    if result.stderr:
                        logger.error(f"Command stderr for frame {i+1}:\n{result.stderr}")
                    
                    if result.returncode != 0:
                        error_msg = f"Command failed with code {result.returncode}"
                        logger.error(f"{error_msg} for frame {i+1}")
                        self.frame_captions.append(f"Error: {error_msg}")
                        continue
                    
                    # Process output only if command succeeded
                    response = None
                    try:
                        # Split output into lines and process
                        lines = result.stdout.split('\n')
                        response_lines = []
                        capture_response = False
                        
                        for line in lines:
                            line = line.strip()
                            # Start capturing after seeing assistant's response
                            if '<assistant>' in line.lower():
                                capture_response = True
                                continue
                            # Stop capturing at the next prompt or system message
                            if capture_response and ('<user>' in line.lower() or 'llama_print' in line.lower()):
                                break
                            if capture_response and line:
                                response_lines.append(line)
                        
                        response = ' '.join(response_lines).strip()
                    except Exception as e:
                        logger.error(f"Error processing output for frame {i+1}: {str(e)}")
                    
                    if response:
                        logger.info(f"Generated caption for frame {i+1}: {response}")
                        self.frame_captions.append(response)
                    else:
                        error_msg = "No valid response generated from model output"
                        logger.warning(f"{error_msg} for frame {i+1}")
                        self.frame_captions.append(error_msg)
                    
                except subprocess.TimeoutExpired:
                    print(f"Error: Command timed out for frame {i+1}")
                    self.frame_captions.append("Timeout error")
                except subprocess.CalledProcessError as e:
                    print(f"Error: Command failed for frame {i+1}: {e.stderr}")
                    self.frame_captions.append("Processing error")
                except Exception as e:
                    print(f"Unexpected error for frame {i+1}: {str(e)}")
                    self.frame_captions.append("Processing error")
                finally:
                    # Keep temp files for debugging
                    print(f"Frame {i+1} saved at: {temp_path}")
            
            if len(self.frames) == 0:
                raise ValueError("No frames were extracted")
            
            self.processing_complete = True
            print("\n" + "="*50)
            print("Video processing completed!")
            print(f"Total frames: {len(self.frames)}")
            print(f"Total captions: {len(self.frame_captions)}")
            print("="*50)
            
        except Exception as e:
            print("\n" + "="*50)
            print(f"Error in process_video: {str(e)}")
            print("="*50)
            self.processing_complete = False
            raise

# ------------------------------------------------------------------------------
# ROUTES
# ------------------------------------------------------------------------------

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    global video_processor
    try:
        logger.info("Upload request received")
        if 'video' not in request.files:
            logger.error("No video file in request")
            return jsonify({'error': 'No video file uploaded'}), 400
        
        video_file = request.files['video']
        if video_file.filename == '':
            logger.error("Empty filename")
            return jsonify({'error': 'No selected file'}), 400
        
        # Save the uploaded file
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
        logger.info(f"Saving video to {video_path}")
        video_file.save(video_path)
        
        # Initialize video processing if not already initialized
        if video_processor is None:
            logger.info("Initializing video processor")
            video_processor = VideoProcessor()
        
        # Process video and wait for completion
        logger.info("Starting video processing")
        try:
            video_processor.process_video(video_path)
            
            # Only return success if processing is complete
            if video_processor.processing_complete:
                logger.info(f"Video processing complete. Total frames: {len(video_processor.frames)}")
                return jsonify({
                    'message': 'Video processed successfully',
                    'total_frames': len(video_processor.frames)
                })
            else:
                logger.error("Video processing incomplete")
                return jsonify({'error': 'Video processing incomplete'}), 500
                
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}", exc_info=True)
            return jsonify({'error': str(e)}), 500
            
    except Exception as e:
        logger.error(f"Error in upload: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    global video_processor
    try:
        logger.info("Chat request received")
        
        # Check video processor state
        if not video_processor:
            logger.error("No video processor initialized")
            return jsonify({'error': 'Please upload a video first'}), 400
            
        logger.info(f"Video processor state - Processing complete: {video_processor.processing_complete}, Frames: {len(video_processor.frames)}, Captions: {len(video_processor.frame_captions)}")
        
        if not video_processor.processing_complete:
            logger.error("Video still processing")
            return jsonify({'error': 'Video is still processing'}), 400
            
        # Validate request data
        data = request.get_json()
        if not data or 'query' not in data:
            logger.error("Invalid request data")
            return jsonify({'error': 'No query provided'}), 400
            
        query = data['query']
        current_frame = data.get('current_frame', 0)
        logger.info(f"Processing query: '{query}' for frame {current_frame}")
        
        # Validate frame index
        if current_frame >= len(video_processor.frames):
            logger.error(f"Invalid frame index {current_frame}")
            return jsonify({'error': 'Invalid frame index'}), 400
            
        # Get current frame
        frame = video_processor.frames[current_frame]
        
        # Save frame as temporary image
        temp_path = os.path.join(video_processor.temp_dir, f"chat_frame_{int(time.time() * 1000)}.jpg")
        try:
            # Save frame
            success = cv2.imwrite(temp_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if not success:
                logger.error("Failed to save frame")
                return jsonify({'error': 'Failed to save frame'}), 500
            
            logger.info(f"Frame saved to {temp_path}")
            
            # Construct command
            cmd = [
                video_processor.cli_path,
                "-m", video_processor.model_path,
                "--mmproj", video_processor.mmproj_path,
                "-t", "4",
                "-n", "512",
                "--temp", "0.1",
                "--ctx-size", "2048",
                "--repeat-penalty", "1.1",
                "--top-p", "0.7",
                "--top-k", "40",
                "--image", temp_path,
                "-p", query
            ]
            
            logger.info(f"Running command: {' '.join(cmd)}")
            
            # Execute command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=180,  # Increased from 60 to 180 seconds (3 minutes) per frame
                check=False  # Don't raise on non-zero exit
            )
            
            # Log command output
            if result.stdout:
                logger.info(f"Command stdout:\n{result.stdout}")
            if result.stderr:
                logger.error(f"Command stderr:\n{result.stderr}")
            
            if result.returncode != 0:
                logger.error(f"Command failed with code {result.returncode}")
                return jsonify({'error': f'Command failed with code {result.returncode}'}), 500
            
            # Process model output
            try:
                # Split output into lines and process
                lines = result.stdout.split('\n')
                response_lines = []
                capture_response = False
                
                for line in lines:
                    line = line.strip()
                    # Start capturing after seeing assistant's response
                    if '<assistant>' in line.lower():
                        capture_response = True
                        continue
                    # Stop capturing at the next prompt or system message
                    if capture_response and ('<user>' in line.lower() or 'llama_print' in line.lower()):
                        break
                    if capture_response and line:
                        response_lines.append(line)
                
                response = ' '.join(response_lines).strip()
                
                if not response:
                    logger.warning("No response generated from model output")
                    return jsonify({'error': 'No response generated'}), 500
                
                logger.info(f"Generated response: {response}")
                return jsonify({'response': response})
                        
            except Exception as e:
                logger.error(f"Error processing model output: {str(e)}", exc_info=True)
                return jsonify({'error': 'Failed to process model output'}), 500
                
        except subprocess.TimeoutExpired as e:
            logger.error("Command timed out", exc_info=True)
            return jsonify({'error': 'Request timed out'}), 504
        except Exception as e:
            logger.error(f"Unexpected error in chat: {str(e)}", exc_info=True)
            return jsonify({'error': 'Internal server error'}), 500
        finally:
            # Clean up temp file
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    logger.info(f"Cleaned up temp file: {temp_path}")
            except Exception as e:
                logger.error(f"Failed to clean up temp file: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/get_frame/<int:frame_idx>')
def get_frame(frame_idx):
    global video_processor
    try:
        if not video_processor or frame_idx >= len(video_processor.frames):
            return jsonify({'error': 'Invalid frame index'}), 400
            
        frame = video_processor.frames[frame_idx]
        caption = video_processor.frame_captions[frame_idx] if frame_idx < len(video_processor.frame_captions) else "No caption available"
        
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