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

# Import ML models
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
from pose_estimator import PoseEstimator, JointAngles
from GardeRAG.model import GardeRAG  # Import GardeRAG

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
video_processor = None  # Global VideoProcessor instance
garde_rag = None  # Global GardeRAG instance

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

# Initialize GardeRAG
def init_garde_rag():
    global garde_rag
    if garde_rag is None:
        garde_rag = GardeRAG()
        logger.info("GardeRAG initialized successfully")

# Call initialization
init_garde_rag()

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
    """Process video files and extract frames with pose estimation"""
    def __init__(self):
        self.pose_estimator = PoseEstimator()
        self.frames = []
        self.frame_angles = []
        self.frame_captions = []
        self.biomech_analyzer = BiomechanicalAnalyzer()
        self.movement_analyzer = MovementAnalyzer()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def _process_frame_batch(self, frame_batch, start_idx):
        """Process a batch of frames using local GardeRAG"""
        try:
            batch_contexts = []
            batch_prompts = []
            
            # Prepare contexts and prompts for the batch
            for i, (frame, angles) in enumerate(zip(frame_batch, self.frame_angles[start_idx:start_idx + len(frame_batch)])):
                frame_idx = start_idx + i
                
                # Get frame window for sequence context
                window_start = max(0, frame_idx - 2)
                frame_window = []
                for idx in range(window_start, frame_idx + 1):
                    if idx < len(self.frame_angles) and self.frame_angles[idx] is not None:
                        curr_angles = self.frame_angles[idx]
                        frame_window.append({
                            'angles': {
                                'sword_arm': curr_angles.sword_arm_angle if curr_angles.sword_arm_angle is not None else 0.0,
                                'front_knee': curr_angles.front_knee_angle if curr_angles.front_knee_angle is not None else 0.0,
                                'back_knee': curr_angles.back_knee_angle if curr_angles.back_knee_angle is not None else 0.0,
                                'torso': curr_angles.torso_angle if curr_angles.torso_angle is not None else 0.0
                            }
                        })
                
                # Get biomechanical and sequence analysis
                frame_data = {
                    'angles': {
                        'sword_arm': angles.sword_arm_angle if angles.sword_arm_angle is not None else 0.0,
                        'front_knee': angles.front_knee_angle if angles.front_knee_angle is not None else 0.0,
                        'back_knee': angles.back_knee_angle if angles.back_knee_angle is not None else 0.0,
                        'torso': angles.torso_angle if angles.torso_angle is not None else 0.0
                    }
                }
                biomech_analysis = self.biomech_analyzer.analyze_movement(frame_data)
                sequence_analysis = self.movement_analyzer.analyze_sequence(frame_window)
                
                context = {
                    "movement_phase": {
                        "sequence": sequence_analysis["sequence"],
                        "phase": sequence_analysis["phase"],
                        "confidence": sequence_analysis["confidence"],
                        "temporal_context": sequence_analysis["temporal_context"]
                    },
                    "biomechanics": {
                        "power_generation": biomech_analysis.power_generation,
                        "balance_control": biomech_analysis.balance_control,
                        "movement_efficiency": biomech_analysis.movement_efficiency,
                        "coordination": biomech_analysis.coordination,
                        "technical_mastery": biomech_analysis.technical_mastery
                    },
                    "current_angles": frame_data['angles']
                }
                
                prompt = f"""Analyze this fencing frame:
Movement: {context['movement_phase']['sequence']} - {context['movement_phase']['phase']}
Angles: Sword arm {frame_data['angles']['sword_arm']:.1f}°, Front knee {frame_data['angles']['front_knee']:.1f}°
Quality: Power {context['biomechanics']['power_generation']:.2f}, Balance {context['biomechanics']['balance_control']:.2f}

Describe the pose and quality in 2-3 sentences. Focus on key technical aspects."""
                
                batch_contexts.append(context)
                batch_prompts.append(prompt)
            
            # Process batch with local GardeRAG
            responses = []
            for prompt, context in zip(batch_prompts, batch_contexts):
                try:
                    result = garde_rag.query(prompt, context)
                    pose_description = result.get("response", "")
                    responses.append(pose_description)
                except Exception as e:
                    logger.error(f"Error in GardeRAG query: {str(e)}")
                    responses.append("Error analyzing pose")
            
            # Create captions for the batch
            for i, (pose_description, context) in enumerate(zip(responses, batch_contexts)):
                biomech = context["biomechanics"]
                movement = context["movement_phase"]
                
                caption = f"""Technical Metrics:
Power Generation: {biomech['power_generation']:.2f}
Balance Control: {biomech['balance_control']:.2f}
Movement Efficiency: {biomech['movement_efficiency']:.2f}
Coordination: {biomech['coordination']:.2f}
Overall Mastery: {biomech['technical_mastery']:.2f}

Pose Analysis:
{pose_description}

Movement Phase: {movement['sequence']} - {movement['phase']}
{movement['temporal_context']}"""
                
                self.frame_captions.append(caption)
            
        except Exception as e:
            logger.error(f"Error processing frame batch: {str(e)}")
            return ["Error analyzing pose"] * len(frame_batch)
    
    def process_video(self, video_path: str, target_fps: int = 5) -> None:
        """Process video file and extract frames with pose estimation"""
        try:
            logger.info(f"Processing video: {video_path}")
            
            # Open video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Failed to open video file")
            
            # Get video properties
            orig_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate frame sampling interval
            sample_interval = max(1, int(orig_fps / target_fps))
            
            # Process frames
            frame_idx = 0
            self.frames = []
            self.frame_angles = []
            self.frame_captions = []
            
            # Process frames in batches
            batch_size = 5
            current_batch = []
            batch_start_idx = 0
            
            with tqdm(total=total_frames//sample_interval, desc="Processing frames") as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    if frame_idx % sample_interval == 0:
                        try:
                            # Convert BGR to RGB
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            
                            # Get pose estimation
                            keypoints, angles = self.pose_estimator.estimate_pose(frame_rgb)
                            
                            # Draw pose on frame if keypoints exist
                            if keypoints is not None and angles is not None:
                                annotated_frame = frame_rgb.copy()
                                annotated_frame = self.pose_estimator.draw_pose(annotated_frame, keypoints, angles)
                            else:
                                annotated_frame = frame_rgb
                            
                            # Add to current batch
                            current_batch.append(annotated_frame)
                            self.frames.append(annotated_frame)
                            self.frame_angles.append(angles)
                            
                            # Process batch if full
                            if len(current_batch) >= batch_size:
                                # Process the batch
                                self._process_frame_batch(current_batch, batch_start_idx)
                                
                                # Reset batch
                                current_batch = []
                                batch_start_idx = len(self.frames)
                            
                            pbar.update(1)
                            
                        except Exception as e:
                            logger.error(f"Error processing frame {frame_idx}: {str(e)}")
                            continue
                            
                    frame_idx += 1
            
            # Process remaining frames
            if current_batch:
                self._process_frame_batch(current_batch, batch_start_idx)
            
            cap.release()
            logger.info(f"Processed {len(self.frames)} frames")
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            raise

# ------------------------------------------------------------------------------
# ROUTES
# ------------------------------------------------------------------------------

@app.route('/')
def index():
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
        
        try:
            # Initialize video processing
            video_processor = VideoProcessor()
            video_processor.process_video(video_path)
            
            return jsonify({
                'message': 'Video processed successfully',
                'total_frames': len(video_processor.frames)
            })
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            return jsonify({'error': 'Error processing video'}), 500
            
    except Exception as e:
        logger.error(f"Error in upload: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    global video_processor
    try:
        if not video_processor:
            return jsonify({'error': 'Please upload a video first'}), 400
            
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'No query provided'}), 400
            
        query = data['query'].lower()
        current_frame = data.get('current_frame', 0)
        
        # Get frame window for analysis
        frame_window = []
        start_idx = max(0, current_frame - 5)
        for idx in range(start_idx, current_frame + 1):
            frame_data = {
                'angles': {
                    'sword_arm': video_processor.frame_angles[idx].sword_arm_angle,
                    'front_knee': video_processor.frame_angles[idx].front_knee_angle,
                    'back_knee': video_processor.frame_angles[idx].back_knee_angle,
                    'torso': video_processor.frame_angles[idx].torso_angle
                }
            }
            frame_window.append(frame_data)
        
        # Perform movement sequence analysis
        sequence_analysis = video_processor.movement_analyzer.analyze_sequence(frame_window)
        
        # Perform biomechanical analysis
        biomech_analysis = video_processor.biomech_analyzer.analyze_movement(frame_window[-1])
        
        # Create rich context for RAG
        context = {
            "movement_phase": {
                "sequence": sequence_analysis["sequence"],
                "phase": sequence_analysis["phase"],
                "confidence": sequence_analysis["confidence"],
                "temporal_context": sequence_analysis["temporal_context"]
            },
            "biomechanics": {
                "power_generation": biomech_analysis.power_generation,
                "balance_control": biomech_analysis.balance_control,
                "movement_efficiency": biomech_analysis.movement_efficiency,
                "coordination": biomech_analysis.coordination,
                "technical_mastery": biomech_analysis.technical_mastery
            },
            "current_angles": frame_window[-1]['angles']
        }
        
        # Determine the type of question and create appropriate prompt
        if any(word in query for word in ['hi', 'hello', 'hey']):
            prompt = """You are a friendly fencing coach. The student just said hello.
Give a warm, brief greeting and ask how you can help them with their fencing technique today.
Keep it casual and encouraging, like a real coach."""
            
        elif any(word in query for word in ['what am i doing', 'what actions', 'what movement']):
            prompt = f"""You are a friendly fencing coach. The student wants to know what movements they're performing.
Current Movement Sequence: {sequence_analysis['temporal_context']}
Current Phase: {context['movement_phase']['sequence']} - {context['movement_phase']['phase']}

Describe what they're doing in simple, clear terms. Focus on the actual movements first, then maybe ONE key observation about their form.
Keep it conversational, like you're watching them in person.

Example: "You're working on a lunge sequence. I can see you preparing with good back leg positioning, though you might want to adjust your sword arm a bit."
"""
        else:
            # Default coaching prompt for other questions
            prompt = f"""You are a friendly fencing coach having a conversation with your student.

Current Movement: {context['movement_phase']['sequence']} - {context['movement_phase']['phase']}
Technical Context:
- Power Generation: {context['biomechanics']['power_generation']:.2f}
- Balance Control: {context['biomechanics']['balance_control']:.2f}
- Movement Efficiency: {context['biomechanics']['movement_efficiency']:.2f}
- Coordination: {context['biomechanics']['coordination']:.2f}

Student Question: {query}

Answer like a real coach would in person:
1. Be conversational and encouraging
2. Address their question directly
3. Maybe add ONE specific tip if relevant
4. Keep it natural - don't list numbers unless specifically asked

Example: "That's a good question! Your lunge is looking strong, but try focusing on extending your sword arm fully before pushing off."
"""
        
        try:
            # Process in batches to avoid rate limiting
            max_retries = 3
            retry_delay = 2
            
            async def process_rag_request():
                try:
                    result = garde_rag.query(prompt, context)
                    pose_description = result.get("response", "")
                    return pose_description
                except Exception as e:
                    logger.error(f"Error in GardeRAG query: {str(e)}")
                    return None
            
            # Create event loop for async processing
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            for attempt in range(max_retries):
                try:
                    feedback = loop.run_until_complete(process_rag_request())
                    if feedback:
                        # Format the feedback
                        def format_coaching_feedback(text):
                            # Remove verbose patterns
                            patterns = [
                                "As a fencing coach, ",
                                "Based on the video, ",
                                "I can observe that ",
                                "It's important to note that ",
                                "I notice that ",
                                "I see that ",
                                "Your ",
                                "I would ",
                                "Let me "
                            ]
                            for pattern in patterns:
                                text = text.replace(pattern, "")
                            
                            # Clean up any leftover artifacts
                            text = text.strip()
                            if text.lower().startswith("your"):
                                text = text[4:].strip()
                            if text.lower().startswith("i see"):
                                text = text[5:].strip()
                            if text.lower().startswith("i notice"):
                                text = text[8:].strip()
                            
                            # Capitalize first letter
                            text = text[0].upper() + text[1:]
                            
                            return text
                        
                        feedback = format_coaching_feedback(feedback)
                        
                        return jsonify({'response': feedback})
                    
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                    else:
                        return jsonify({'error': 'Error generating coaching feedback'}), 500
                        
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Error in chat: {str(e)}")
                        return jsonify({'error': str(e)}), 500
                    time.sleep(retry_delay * (attempt + 1))
                    
            loop.close()
            
        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            return jsonify({'error': str(e)}), 500
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

if __name__ == '__main__':
    app.run(debug=True, port=5000) 