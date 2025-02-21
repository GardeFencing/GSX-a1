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
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    """Process video files and extract frames using optimized approach"""
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
            
        # Initialize thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Constants for frame processing
        self.MAX_FRAMES = 32  # Balance between speed and memory
        self.FRAME_DIFFERENCE_THRESHOLD = 10.0  # Threshold for frame difference detection
        
        print(f"Model path: {self.model_path}")
        print(f"MMPROJ path: {self.mmproj_path}")
        print(f"CLI path: {self.cli_path}")
        print(f"Temp dir: {self.temp_dir}")
        
        self._initialized = True
    
    def uniform_sample(self, frame_indices: List[int], n: int) -> List[int]:
        """Uniformly sample n frames from the list of indices"""
        gap = len(frame_indices) / n
        return [int(i * gap + gap / 2) for i in range(n)]
    
    def _calculate_frame_difference(self, current_frame: np.ndarray, prev_frame: Optional[np.ndarray]) -> float:
        """Calculate difference between consecutive frames to detect significant changes"""
        if prev_frame is None:
            return float('inf')
        
        # Convert to grayscale for faster processing
        curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate absolute difference
        diff = cv2.absdiff(curr_gray, prev_gray)
        return np.mean(diff)
    
    def process_single_frame(self, frame: np.ndarray, frame_idx: int, fps: float) -> str:
        """Process a single frame and return its caption"""
        # Clean up old files first
        try:
            for f in os.listdir(self.temp_dir):
                if f.startswith('temp_frame_') or f.startswith('frame_') or f.startswith('chat_frame_'):
                    try:
                        os.remove(os.path.join(self.temp_dir, f))
                    except:
                        pass
        except:
            pass

        # Use timestamp to ensure unique filename
        timestamp = int(time.time() * 1000)
        temp_path = os.path.join(self.temp_dir, f"temp_frame_{timestamp}_{frame_idx}.jpg")
        
        try:
            # Save frame as temporary file with high quality
            success = cv2.imwrite(temp_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                                [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not success:
                logger.error(f"Failed to save frame {frame_idx} to {temp_path}")
                return "Failed to save frame"
            
            # Verify file was actually written
            if not os.path.exists(temp_path):
                logger.error(f"File {temp_path} was not created")
                return "Failed to create frame file"
                
            file_size = os.path.getsize(temp_path)
            if file_size == 0:
                logger.error(f"File {temp_path} is empty")
                return "Frame file is empty"
            
            logger.info(f"Successfully saved frame {frame_idx} to {temp_path} (size: {file_size} bytes)")
            
            # Log paths for debugging
            logger.info(f"Processing frame {frame_idx} with paths:")
            logger.info(f"CLI path: {self.cli_path}")
            logger.info(f"Model path: {self.model_path}")
            logger.info(f"MMPROJ path: {self.mmproj_path}")
            logger.info(f"Frame path: {temp_path}")
            
            # Verify file existence
            if not os.path.exists(self.cli_path):
                logger.error(f"CLI executable not found at {self.cli_path}")
                return "Error: CLI executable not found"
            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found at {self.model_path}")
                return "Error: Model file not found"
            if not os.path.exists(self.mmproj_path):
                logger.error(f"MMPROJ file not found at {self.mmproj_path}")
                return "Error: MMPROJ file not found"
            
            # Construct command with GPU acceleration
            cmd = [
                self.cli_path,
                "-m", os.path.abspath(self.model_path),
                "--mmproj", os.path.abspath(self.mmproj_path),
                "-t", str(max(1, os.cpu_count() // 2)),
                "-n", "256",
                "--temp", "0.1",
                "--ctx-size", "2048",
                "--top-p", "0.7",
                "--top-k", "40",
                "--image", os.path.abspath(temp_path),
                "-p", "Describe what is happening in this frame."
            ]
            
            logger.info(f"Running command for frame {frame_idx}: {' '.join(cmd)}")
            
            # Run command with retries
            max_retries = 3
            retry_count = 0
            while retry_count < max_retries:
                try:
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=180,
                        check=True
                    )
                    
                    if result.returncode != 0:
                        logger.error(f"Command failed for frame {frame_idx} with code {result.returncode}")
                        logger.error(f"Error output: {result.stderr}")
                        retry_count += 1
                        if retry_count < max_retries:
                            time.sleep(1)
                            continue
                        return f"Error: Command failed with code {result.returncode}"
                    
                    # Process output to extract caption
                    lines = result.stdout.split('\n')
                    caption_lines = []
                    for line in lines:
                        if line and not line.startswith('>') and not line.startswith('llama_print'):
                            caption_lines.append(line.strip())
                    
                    caption = " ".join(caption_lines)
                    if not caption:
                        logger.error(f"No caption generated for frame {frame_idx}")
                        retry_count += 1
                        if retry_count < max_retries:
                            time.sleep(1)
                            continue
                        return "Error: No caption generated"
                    
                    logger.info(f"Successfully generated caption for frame {frame_idx}")
                    return caption
                    
                except subprocess.TimeoutExpired:
                    logger.error(f"Timeout processing frame {frame_idx} (attempt {retry_count + 1})")
                    retry_count += 1
                    if retry_count < max_retries:
                        time.sleep(1)
                        continue
                    return "Error: Frame processing timed out"
                except Exception as e:
                    logger.error(f"Error processing frame {frame_idx}: {str(e)}")
                    retry_count += 1
                    if retry_count < max_retries:
                        time.sleep(1)
                        continue
                    return f"Error: {str(e)}"
                
            return "Error: All retries failed"
            
        except Exception as e:
            logger.error(f"Error processing frame {frame_idx}: {str(e)}")
            return f"Error: {str(e)}"
        finally:
            # Cleanup temporary file
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    logger.info(f"Cleaned up temp file {temp_path}")
            except Exception as e:
                logger.error(f"Failed to clean up temp file {temp_path}: {str(e)}")
    
    def process_video(self, video_path: str) -> None:
        """Process video file with optimized frame extraction and GPU acceleration"""
        try:
            logger.info("\n" + "="*50)
            logger.info("Starting video processing...")
            logger.info("="*50)
            
            # Reset state
            self.frames = []
            self.frame_captions = []
            self.processing_complete = False
            
            # Open video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps
            
            logger.info(f"Video properties:")
            logger.info(f"- Total frames: {total_frames}")
            logger.info(f"- FPS: {fps}")
            logger.info(f"- Duration: {duration:.2f} seconds")
            
            # Sample 1 frame per second
            frame_times = np.arange(0, duration, 1.0)  # One frame per second
            frame_indices = (frame_times * fps).astype(int)
            
            # Cap at MAX_FRAMES if needed
            if len(frame_indices) > self.MAX_FRAMES:
                logger.info(f"Limiting frames from {len(frame_indices)} to {self.MAX_FRAMES}")
                step = len(frame_indices) // self.MAX_FRAMES
                frame_indices = frame_indices[::step][:self.MAX_FRAMES]
            
            logger.info(f"Will process {len(frame_indices)} frames")
            
            # Extract frames
            frames_with_indices = []
            prev_frame = None
            
            for i, frame_idx in enumerate(frame_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    # Skip if too similar to previous frame
                    if prev_frame is not None:
                        diff = self._calculate_frame_difference(frame, prev_frame)
                        if diff < self.FRAME_DIFFERENCE_THRESHOLD:
                            logger.info(f"Skipping frame {i} (too similar to previous)")
                            continue
                    
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames_with_indices.append((frame_rgb, len(self.frames)))
                    self.frames.append(frame_rgb)
                    prev_frame = frame
                    logger.info(f"Processing frame {len(self.frames)-1} at time {frame_idx/fps:.2f}s")
            
            cap.release()
            
            if not frames_with_indices:
                raise ValueError("No frames were extracted")
            
            # Initialize captions list with the correct size
            self.frame_captions = [None] * len(self.frames)
            
            logger.info(f"\nProcessing {len(frames_with_indices)} frames with captions...")
            
            # Process frames in parallel
            future_to_frame = {
                self.executor.submit(self.process_single_frame, frame, idx, fps): idx 
                for frame, idx in frames_with_indices
            }
            
            # Collect results
            for future in as_completed(future_to_frame):
                try:
                    idx = future_to_frame[future]
                    caption = future.result()
                    self.frame_captions[idx] = caption
                    logger.info(f"Caption generated for frame {idx}")
                except Exception as e:
                    logger.error(f"Error processing frame {idx}: {str(e)}")
                    self.frame_captions[idx] = f"Error: {str(e)}"
            
            # Mark processing as complete
            self.processing_complete = True
            
            logger.info("\n" + "="*50)
            logger.info("Video processing completed!")
            logger.info(f"Total frames processed: {len(self.frames)}")
            logger.info(f"Total captions generated: {len(self.frame_captions)}")
            logger.info("="*50)
            
        except Exception as e:
            logger.error(f"Error in process_video: {str(e)}")
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
        if 'video' not in request.files:
            return jsonify({'error': 'No video file uploaded'}), 400
            
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        # Save the uploaded file
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
        video_file.save(video_path)
        
        # Initialize video processing if not already initialized
        if video_processor is None:
            video_processor = VideoProcessor()
        
        # Process video
        try:
            video_processor.process_video(video_path)
            
            if not video_processor.processing_complete:
                return jsonify({'error': 'Video processing failed'}), 500
                
            if not video_processor.frames or not video_processor.frame_captions:
                return jsonify({'error': 'No frames or captions generated'}), 500
                
            return jsonify({
                'message': 'Video processed successfully',
                'total_frames': len(video_processor.frames),
                'captions_generated': len(video_processor.frame_captions)
            })
            
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
            logger.error("Chat request failed: Video processor not initialized")
            return jsonify({'error': 'Please upload a video first'}), 400
            
        if not video_processor.processing_complete:
            logger.error("Chat request failed: Video still processing")
            return jsonify({'error': 'Video is still processing'}), 400
            
        # Validate request data
        if not request.is_json:
            logger.error("Chat request failed: Not JSON")
            return jsonify({'error': 'Request must be JSON'}), 400
            
        data = request.get_json()
        logger.info(f"Chat request data: {data}")
        
        if not data:
            logger.error("Chat request failed: Empty request data")
            return jsonify({'error': 'No request data provided'}), 400
            
        if 'query' not in data:
            logger.error("Chat request failed: No query provided")
            return jsonify({'error': 'No query provided'}), 400
            
        query = data['query']
        current_frame = data.get('current_frame', 0)
        
        # Validate frame index
        if not isinstance(current_frame, int):
            logger.error(f"Chat request failed: Invalid frame index type: {type(current_frame)}")
            return jsonify({'error': 'Frame index must be an integer'}), 400
            
        if current_frame < 0:
            logger.error(f"Chat request failed: Negative frame index: {current_frame}")
            return jsonify({'error': 'Frame index cannot be negative'}), 400
            
        if current_frame >= len(video_processor.frames):
            logger.error(f"Chat request failed: Frame index {current_frame} out of range")
            return jsonify({'error': 'Invalid frame index'}), 400
            
        # Get current frame
        frame = video_processor.frames[current_frame]
        
        # Save frame as temporary image
        temp_path = os.path.join(video_processor.temp_dir, f"chat_frame_{int(time.time() * 1000)}.jpg")
        try:
            # Save frame
            success = cv2.imwrite(temp_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if not success:
                logger.error("Failed to save frame for chat")
                return jsonify({'error': 'Failed to save frame'}), 500
            
            # Construct command
            cmd = [
                video_processor.cli_path,
                "-m", video_processor.model_path,
                "--mmproj", video_processor.mmproj_path,
                "-t", str(max(1, os.cpu_count() // 2)),
                "-n", "256",
                "--temp", "0.1",
                "--ctx-size", "2048",
                "--top-p", "0.7",
                "--top-k", "40",
                "--image", temp_path,
                "-p", query
            ]
            
            logger.info(f"Running chat command: {' '.join(cmd)}")
            
            # Execute command with retries
            max_retries = 3
            retry_count = 0
            last_error = None
            
            while retry_count < max_retries:
                try:
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=180,
                        check=True
                    )
                    
                    # Process output
                    lines = result.stdout.split('\n')
                    response_lines = []
                    capture_response = False
                    
                    for line in lines:
                        line = line.strip()
                        if '<assistant>' in line.lower():
                            capture_response = True
                            continue
                        if capture_response and ('<user>' in line.lower() or 'llama_print' in line.lower()):
                            break
                        if capture_response and line:
                            response_lines.append(line)
                    
                    response = ' '.join(response_lines).strip()
                    
                    if not response:
                        retry_count += 1
                        last_error = "Empty response from model"
                        logger.warning(f"Empty response on attempt {retry_count}")
                        time.sleep(1)
                        continue
                    
                    logger.info("Successfully generated chat response")
                    return jsonify({'response': response})
                    
                except subprocess.TimeoutExpired:
                    retry_count += 1
                    last_error = "Request timed out"
                    logger.warning(f"Timeout on attempt {retry_count}")
                    time.sleep(1)
                    continue
                    
                except subprocess.CalledProcessError as e:
                    retry_count += 1
                    last_error = f"Command failed: {e.stderr}"
                    logger.warning(f"Command failed on attempt {retry_count}: {e.stderr}")
                    time.sleep(1)
                    continue
                    
                except Exception as e:
                    logger.error(f"Unexpected error in chat: {str(e)}")
                    return jsonify({'error': str(e)}), 500
                    
            # If we get here, all retries failed
            logger.error(f"All chat retries failed. Last error: {last_error}")
            return jsonify({'error': last_error or 'Failed to process request'}), 500
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception as e:
                    logger.error(f"Failed to clean up temp file: {str(e)}")
                    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_frame/<int:frame_idx>')
def get_frame(frame_idx):
    global video_processor
    try:
        if not video_processor:
            return jsonify({'error': 'Video processor not initialized'}), 400
            
        if frame_idx >= len(video_processor.frames):
            return jsonify({'error': 'Invalid frame index'}), 400
            
        frame = video_processor.frames[frame_idx]
        
        # Ensure we have captions
        if not video_processor.frame_captions or frame_idx >= len(video_processor.frame_captions):
            caption = "Caption not available"
        else:
            caption = video_processor.frame_captions[frame_idx]
            if not caption or caption.startswith("Error:"):
                caption = "Caption generation failed"
        
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