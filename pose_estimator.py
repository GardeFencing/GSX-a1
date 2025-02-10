import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import logging
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from collections import deque

# Set tokenizers parallelism explicitly
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Force TensorFlow to use CPU only to avoid GPU conflicts with PyTorch/CLIP
tf.config.set_visible_devices([], 'GPU')

# MoveNet keypoint indices remain the same
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

@dataclass
class JointAngles:
    """Class to store joint angles from pose estimation"""
    sword_arm_angle: Optional[float] = None
    front_knee_angle: Optional[float] = None
    back_knee_angle: Optional[float] = None
    torso_angle: Optional[float] = None
    stance_width: Optional[float] = None
    confidence: float = 0.0  # Confidence score for the pose estimation

@dataclass
class MotionSequence:
    """Track motion sequences for advanced analysis"""
    phase: str  # preparation, attack, recovery
    start_frame: int
    end_frame: int
    key_angles: List[JointAngles]
    motion_score: float  # Quality score for the motion
    technical_notes: List[str]

class PoseEstimator:
    def __init__(self, sequence_length: int = 5):
        """Initialize MoveNet pose estimator (SinglePoseLightning)"""
        logger.info("Initializing MoveNet pose estimator (SinglePoseLightning)...")
        
        # Model parameters
        self.input_size = 192  # MoveNet Lightning input size
        self.confidence_threshold = 0.2
        
        # Set up model cache directory
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache")
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["TFHUB_CACHE_DIR"] = cache_dir
        
        try:
            # Model handle for MoveNet Lightning
            model_handle = "https://tfhub.dev/google/movenet/singlepose/lightning/4"
            model_path = os.path.join(cache_dir, "movenet_lightning")
            
            # Always download a fresh model to ensure proper initialization
            logger.info("Downloading MoveNet model...")
            model = hub.load(model_handle)
            
            # Get the concrete function directly
            self.movenet = model.signatures['serving_default']
            
            # Save model for future use
            logger.info("Saving model to cache...")
            tf.saved_model.save(model, model_path)
            
            # Verify input tensor name
            input_details = list(self.movenet.structured_input_signature[1].keys())[0]
            logger.info(f"Model loaded successfully. Input tensor name: {input_details}")
            
            # Store input tensor name for later use
            self.input_tensor_name = input_details
            
        except Exception as e:
            logger.error(f"Error loading MoveNet model: {str(e)}")
            raise RuntimeError(f"Failed to initialize MoveNet model: {str(e)}")
        
        # Initialize motion tracking
        self.sequence_length = sequence_length
        self.keypoint_history = []
        self.sequence_buffer = deque(maxlen=sequence_length)
        self.motion_sequences = []
        
        # Define ideal angles for different phases
        self.ideal_angles = {
            'en_garde': {
                'sword_arm_angle': 135.0,
                'front_knee_angle': 120.0,
                'back_knee_angle': 150.0,
                'torso_angle': 90.0,
                'stance_width': 1.0  # Normalized
            },
            'lunge': {
                'sword_arm_angle': 180.0,
                'front_knee_angle': 90.0,
                'back_knee_angle': 160.0,
                'torso_angle': 85.0,
                'stance_width': 1.5  # Normalized
            }
        }
        logger.info("Pose estimator initialized with motion tracking")

    def _convert_normalized_to_pixel(self, x_norm: float, y_norm: float, orig_width: int, orig_height: int) -> Tuple[int, int]:
        """
        Convert normalized coordinates from the padded input image back to the original frame coordinates.
        """
        target = self.input_size  # e.g., 192
        # Compute scaling factor and padding offsets
        scale = min(target / orig_height, target / orig_width)
        new_height = orig_height * scale
        new_width = orig_width * scale
        pad_top = (target - new_height) / 2
        pad_left = (target - new_width) / 2

        # Convert normalized coordinates to absolute coordinates in the padded image
        x_abs = x_norm * target
        y_abs = y_norm * target

        # Remove padding and scale back to original image dimensions
        x_orig = (x_abs - pad_left) / scale
        y_orig = (y_abs - pad_top) / scale

        # Ensure coordinates are within bounds
        x_orig = int(np.clip(x_orig, 0, orig_width - 1))
        y_orig = int(np.clip(y_orig, 0, orig_height - 1))
        return x_orig, y_orig

    def _filter_keypoints(self, keypoints):
        """Filter keypoints based on confidence threshold and convert to frame coordinates"""
        filtered_keypoints = []
        for kp in keypoints:
            y, x, score = kp  # MoveNet returns [y, x, score]
            if score > self.confidence_threshold:
                filtered_keypoints.append((x, y, score))  # Convert to (x, y, score) format
            else:
                filtered_keypoints.append(None)
        return filtered_keypoints

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame for pose estimation (BGR → RGB and optional resizing)."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb

    def _get_keypoints(self, frame):
        """Get keypoints from frame using MoveNet model"""
        try:
            orig_height, orig_width = frame.shape[:2]
            # Resize and pad the image to keep the aspect ratio and fit the expected size
            input_image = tf.expand_dims(frame, axis=0)
            input_image = tf.image.resize_with_pad(input_image, self.input_size, self.input_size)
            
            # Convert to int32 (expected by the model)
            input_image = tf.cast(input_image, dtype=tf.int32)
            
            # Run model inference with the correct input tensor name
            input_tensor_name = list(self.movenet.structured_input_signature[1].keys())[0]
            outputs = self.movenet(**{input_tensor_name: input_image})
            
            # Process the outputs
            keypoints = outputs['output_0'].numpy()
            keypoints = np.squeeze(keypoints)  # Remove batch dimension
            
            processed_keypoints = []
            
            for kp in keypoints:
                y, x, confidence = kp
                if confidence > self.confidence_threshold:
                    # Convert normalized coordinates to pixel coordinates using the helper function
                    x_px, y_px = self._convert_normalized_to_pixel(x, y, orig_width, orig_height)
                    processed_keypoints.append([x_px, y_px, confidence])
                else:
                    processed_keypoints.append(None)
            
            return processed_keypoints
            
        except Exception as e:
            logger.error(f"Error in pose estimation: {str(e)}")
            return [None] * 17  # Return None for all 17 keypoints

    def _calculate_angle(
        self,
        p1: Optional[Tuple[float, float, float]], 
        p2: Optional[Tuple[float, float, float]], 
        p3: Optional[Tuple[float, float, float]]
    ) -> Tuple[float, float]:
        """Calculate the angle between three points and return the angle plus average confidence."""
        if not all([p1, p2, p3]):
            return 0.0, 0.0
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        confidence = np.mean([p1[2], p2[2], p3[2]])
        return angle, confidence

    def _analyze_fencing_pose(self, keypoints: List[Optional[Tuple[float, float, float]]], frame_idx: int) -> Tuple[JointAngles, dict]:
        """Analyze the fencing pose from the detected keypoints."""
        # Get relevant points (adjust indices as needed)
        right_shoulder = keypoints[KEYPOINT_DICT['right_shoulder']]
        right_elbow = keypoints[KEYPOINT_DICT['right_elbow']]
        right_wrist = keypoints[KEYPOINT_DICT['right_wrist']]
        right_hip = keypoints[KEYPOINT_DICT['right_hip']]
        right_knee = keypoints[KEYPOINT_DICT['right_knee']]
        right_ankle = keypoints[KEYPOINT_DICT['right_ankle']]
        left_hip = keypoints[KEYPOINT_DICT['left_hip']]
        left_knee = keypoints[KEYPOINT_DICT['left_knee']]
        left_ankle = keypoints[KEYPOINT_DICT['left_ankle']]
        
        # Calculate angles using your helper
        sword_arm, sa_conf = self._calculate_angle(right_shoulder, right_elbow, right_wrist)
        front_knee, fk_conf = self._calculate_angle(right_hip, right_knee, right_ankle)
        back_knee, bk_conf = self._calculate_angle(left_hip, left_knee, left_ankle)
        
        # Calculate torso angle (using right shoulder and right hip)
        if right_shoulder and right_hip:
            torso_vector = np.array([right_shoulder[0] - right_hip[0], right_shoulder[1] - right_hip[1]])
            vertical = np.array([0, -1])
            torso_angle = np.degrees(np.arccos(np.clip(
                np.dot(torso_vector, vertical) / (np.linalg.norm(torso_vector) * np.linalg.norm(vertical)),
                -1.0, 1.0
            )))
            torso_conf = (right_shoulder[2] + right_hip[2]) / 2
        else:
            torso_angle, torso_conf = 0.0, 0.0
        
        # Calculate stance width (distance between left and right ankles)
        if right_ankle and left_ankle:
            stance_width = np.linalg.norm([right_ankle[0] - left_ankle[0], right_ankle[1] - left_ankle[1]])
            stance_conf = (right_ankle[2] + left_ankle[2]) / 2
        else:
            stance_width, stance_conf = 0.0, 0.0

        confidence = np.mean([sa_conf, fk_conf, bk_conf, torso_conf, stance_conf])
        angles = JointAngles(
            sword_arm_angle=sword_arm,
            front_knee_angle=front_knee,
            back_knee_angle=back_knee,
            torso_angle=torso_angle,
            stance_width=stance_width,
            confidence=confidence
        )
        scores = self._calculate_biomechanical_scores(angles)
        self.sequence_buffer.append((frame_idx, angles, scores))
        if len(self.sequence_buffer) == self.sequence_buffer.maxlen:
            self._analyze_motion_sequence()
        return angles, scores

    def _calculate_biomechanical_scores(self, angles: JointAngles) -> dict:
        """Calculate biomechanical quality scores (example implementation)."""
        en_garde_scores = {
            'sword_arm': 1 - abs(angles.sword_arm_angle - self.ideal_angles['en_garde']['sword_arm_angle']) / 180,
            'front_knee': 1 - abs(angles.front_knee_angle - self.ideal_angles['en_garde']['front_knee_angle']) / 180,
            'back_knee': 1 - abs(angles.back_knee_angle - self.ideal_angles['en_garde']['back_knee_angle']) / 180,
            'torso': 1 - abs(angles.torso_angle - self.ideal_angles['en_garde']['torso_angle']) / 90
        }
        lunge_scores = {
            'sword_arm': 1 - abs(angles.sword_arm_angle - self.ideal_angles['lunge']['sword_arm_angle']) / 180,
            'front_knee': 1 - abs(angles.front_knee_angle - self.ideal_angles['lunge']['front_knee_angle']) / 180,
            'back_knee': 1 - abs(angles.back_knee_angle - self.ideal_angles['lunge']['back_knee_angle']) / 180,
            'torso': 1 - abs(angles.torso_angle - self.ideal_angles['lunge']['torso_angle']) / 90
        }
        scores = {
            'en_garde_score': np.mean(list(en_garde_scores.values())) * angles.confidence,
            'lunge_score': np.mean(list(lunge_scores.values())) * angles.confidence
        }
        return scores

    def _analyze_motion_sequence(self) -> Optional[MotionSequence]:
        """Analyze a sequence of poses to identify and score fencing actions."""
        if len(self.sequence_buffer) < 2:
            return None
        start_frame, start_angles, _ = self.sequence_buffer[0]
        end_frame, end_angles, _ = self.sequence_buffer[-1]
        angle_velocities = []
        for i in range(1, len(self.sequence_buffer)):
            _, prev_angles, _ = self.sequence_buffer[i-1]
            _, curr_angles, _ = self.sequence_buffer[i]
            sword_arm_vel = curr_angles.sword_arm_angle - prev_angles.sword_arm_angle
            front_knee_vel = curr_angles.front_knee_angle - prev_angles.front_knee_angle
            angle_velocities.append((sword_arm_vel, front_knee_vel))
        avg_sword_vel = np.mean([v[0] for v in angle_velocities])
        avg_knee_vel = np.mean([v[1] for v in angle_velocities])
        if abs(avg_sword_vel) < 5 and abs(avg_knee_vel) < 5:
            phase = "preparation"
        elif avg_sword_vel > 10 or avg_knee_vel < -10:
            phase = "attack"
        else:
            phase = "recovery"
        motion_score = np.mean([angles.confidence for _, angles, _ in self.sequence_buffer])
        technical_notes = []
        if phase == "attack":
            if end_angles.sword_arm_angle < 160:
                technical_notes.append("Extend sword arm fully during attack")
            if end_angles.front_knee_angle > 100:
                technical_notes.append("Deepen front knee bend in lunge")
        sequence = MotionSequence(
            phase=phase,
            start_frame=start_frame,
            end_frame=end_frame,
            key_angles=[angles for _, angles, _ in self.sequence_buffer],
            motion_score=motion_score,
            technical_notes=technical_notes
        )
        self.motion_sequences.append(sequence)
        return sequence

    def estimate_pose(self, frame):
        """Estimate pose in a frame and return keypoints with joint angles"""
        try:
            orig_height, orig_width = frame.shape[:2]
            
            # Process frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize frame to model input size while maintaining aspect ratio
            input_image = tf.expand_dims(frame_rgb, axis=0)
            input_image = tf.image.resize_with_pad(input_image, self.input_size, self.input_size)
            input_image = tf.cast(input_image, dtype=tf.int32)
            
            # Run model inference using stored input tensor name
            outputs = self.movenet(**{self.input_tensor_name: input_image})
            keypoints = outputs['output_0'].numpy()
            keypoints = np.squeeze(keypoints)  # Remove batch dimension
            
            processed_keypoints = []
            
            for kp in keypoints:
                y, x, confidence = kp
                if confidence > self.confidence_threshold:
                    # Convert using the helper so that we correctly account for padding/scale
                    x_px, y_px = self._convert_normalized_to_pixel(x, y, orig_width, orig_height)
                    processed_keypoints.append([x_px, y_px, confidence])
                else:
                    processed_keypoints.append(None)
            
            # Calculate joint angles
            angles = self._calculate_joint_angles(processed_keypoints)
            
            return processed_keypoints, angles
            
        except Exception as e:
            logger.error(f"Error in pose estimation: {str(e)}")
            return [None] * 17, None

    def draw_pose(self, frame: np.ndarray, keypoints: List[Optional[List[float]]], angles: JointAngles) -> np.ndarray:
        """Draw pose keypoints and angles on the frame."""
        try:
            annotated_frame = frame.copy()
            height, width = frame.shape[:2]
            
            # Define colors
            KEYPOINT_COLOR = (0, 255, 0)  # Green
            CONNECTION_COLOR = (0, 255, 255)  # Yellow
            TEXT_COLOR = (255, 255, 255)  # White
            
            # Draw connections first (so they appear behind keypoints)
            connections = [
                # Torso
                ([5, 6], (0, 0, 255)),   # Shoulders (Red)
                ([5, 11], (0, 0, 255)),  # Left torso
                ([6, 12], (0, 0, 255)),  # Right torso
                ([11, 12], (0, 0, 255)), # Hips
                # Arms
                ([5, 7, 9], (255, 0, 0)),   # Left arm (Blue)
                ([6, 8, 10], (255, 0, 0)),  # Right arm
                # Legs
                ([11, 13, 15], (0, 255, 0)), # Left leg (Green)
                ([12, 14, 16], (0, 255, 0))  # Right leg
            ]
            
            # Draw skeleton connections
            for connection in connections:
                points = connection[0]
                color = connection[1]
                
                # Draw lines between consecutive points in each connection
                for i in range(len(points) - 1):
                    if (points[i] < len(keypoints) and points[i+1] < len(keypoints) and
                        keypoints[points[i]] is not None and keypoints[points[i+1]] is not None):
                        pt1 = (int(keypoints[points[i]][0]), int(keypoints[points[i]][1]))
                        pt2 = (int(keypoints[points[i+1]][0]), int(keypoints[points[i+1]][1]))
                        cv2.line(annotated_frame, pt1, pt2, color, 2)
            
            # Draw keypoints
            for i, keypoint in enumerate(keypoints):
                if keypoint is not None:
                    x, y = int(keypoint[0]), int(keypoint[1])
                    cv2.circle(annotated_frame, (x, y), 4, KEYPOINT_COLOR, -1)
                    cv2.circle(annotated_frame, (x, y), 6, (0, 0, 0), 1)
            
            # Draw angles with better positioning and visibility
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            def put_outlined_text(img, text, pos, font, scale, color, thickness):
                # Draw black outline
                cv2.putText(img, text, pos, font, scale, (0, 0, 0), thickness + 1)
                # Draw text in color
                cv2.putText(img, text, pos, font, scale, color, thickness)
            
            # Draw sword arm angle
            if angles.sword_arm_angle is not None and keypoints[8] is not None:
                x, y = int(keypoints[8][0]), int(keypoints[8][1])
                text = f"{angles.sword_arm_angle:.1f}°"
                put_outlined_text(annotated_frame, text, (x + 10, y), font, font_scale, TEXT_COLOR, thickness)
            
            # Draw front knee angle
            if angles.front_knee_angle is not None and keypoints[14] is not None:
                x, y = int(keypoints[14][0]), int(keypoints[14][1])
                text = f"{angles.front_knee_angle:.1f}°"
                put_outlined_text(annotated_frame, text, (x + 10, y), font, font_scale, TEXT_COLOR, thickness)
            
            # Draw back knee angle
            if angles.back_knee_angle is not None and keypoints[13] is not None:
                x, y = int(keypoints[13][0]), int(keypoints[13][1])
                text = f"{angles.back_knee_angle:.1f}°"
                put_outlined_text(annotated_frame, text, (x - 60, y), font, font_scale, TEXT_COLOR, thickness)
            
            # Draw torso angle
            if angles.torso_angle is not None and keypoints[5] is not None and keypoints[6] is not None:
                mid_x = int((keypoints[5][0] + keypoints[6][0]) / 2)
                mid_y = int((keypoints[5][1] + keypoints[6][1]) / 2)
                text = f"{angles.torso_angle:.1f}°"
                put_outlined_text(annotated_frame, text, (mid_x - 30, mid_y - 10), font, font_scale, TEXT_COLOR, thickness)
            
            return annotated_frame
            
        except Exception as e:
            logger.error(f"Error drawing pose: {str(e)}")
            return frame

    def _calculate_joint_angles(self, keypoints):
        """Calculate joint angles from keypoints"""
        try:
            # Create JointAngles object to store results
            angles = JointAngles()
            
            # Helper function to calculate angle between three points
            def calculate_angle(p1, p2, p3):
                if None in (p1, p2, p3):
                    return None
                    
                # Convert to numpy arrays for easier calculation
                p1 = np.array([p1[0], p1[1]])
                p2 = np.array([p2[0], p2[1]])
                p3 = np.array([p3[0], p3[1]])
                
                # Calculate vectors
                v1 = p1 - p2
                v2 = p3 - p2
                
                # Calculate angle in degrees
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                return np.degrees(angle)
            
            # Calculate sword arm angle (right arm)
            # Keypoint indices: 6=right_shoulder, 8=right_elbow, 10=right_wrist
            angles.sword_arm_angle = calculate_angle(
                keypoints[6], keypoints[8], keypoints[10]
            ) if all(kp is not None for kp in [keypoints[6], keypoints[8], keypoints[10]]) else None
            
            # Calculate front knee angle (right leg)
            # Keypoint indices: 12=right_hip, 14=right_knee, 16=right_ankle
            angles.front_knee_angle = calculate_angle(
                keypoints[12], keypoints[14], keypoints[16]
            ) if all(kp is not None for kp in [keypoints[12], keypoints[14], keypoints[16]]) else None
            
            # Calculate back knee angle (left leg)
            # Keypoint indices: 11=left_hip, 13=left_knee, 15=left_ankle
            angles.back_knee_angle = calculate_angle(
                keypoints[11], keypoints[13], keypoints[15]
            ) if all(kp is not None for kp in [keypoints[11], keypoints[13], keypoints[15]]) else None
            
            # Calculate torso angle
            # Use shoulders and hips to determine torso orientation
            if all(kp is not None for kp in [keypoints[5], keypoints[6], keypoints[11], keypoints[12]]):
                left_shoulder = np.array([keypoints[5][0], keypoints[5][1]])
                right_shoulder = np.array([keypoints[6][0], keypoints[6][1]])
                left_hip = np.array([keypoints[11][0], keypoints[11][1]])
                right_hip = np.array([keypoints[12][0], keypoints[12][1]])
                
                # Calculate midpoints
                shoulder_mid = (left_shoulder + right_shoulder) / 2
                hip_mid = (left_hip + right_hip) / 2
                
                # Calculate torso vector and angle with vertical
                torso_vector = shoulder_mid - hip_mid
                vertical = np.array([0, -1])  # Negative because y-axis points down in image coordinates
                
                cos_angle = np.dot(torso_vector, vertical) / (np.linalg.norm(torso_vector) * np.linalg.norm(vertical))
                angles.torso_angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
            else:
                angles.torso_angle = None
            
            # Calculate stance width (distance between feet)
            # Keypoint indices: 15=left_ankle, 16=right_ankle
            if keypoints[15] is not None and keypoints[16] is not None:
                left_foot = np.array([keypoints[15][0], keypoints[15][1]])
                right_foot = np.array([keypoints[16][0], keypoints[16][1]])
                angles.stance_width = np.linalg.norm(right_foot - left_foot)
            else:
                angles.stance_width = None
            
            return angles
            
        except Exception as e:
            logger.error(f"Error calculating joint angles: {str(e)}")
            return JointAngles()
