import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from collections import deque

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
        """Initialize MoveNet pose estimator (SinglePoseThunder)"""
        logger.info("Initializing MoveNet pose estimator (SinglePoseThunder)...")
        
        # Model parameters
        self.input_size = 256  # MoveNet Thunder input size (changed from 192)
        self.confidence_threshold = 0.2
        
        # Load MoveNet Thunder
        model_path = "https://tfhub.dev/google/movenet/singlepose/thunder/4"
        self.model = hub.load(model_path)
        self.movenet = self.model.signatures['serving_default']
        
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
        
        # Define the keypoint pairs for drawing
        self.KEYPOINT_EDGES = {
            (0, 1): 'm',    # nose to left_eye
            (0, 2): 'm',    # nose to right_eye
            (1, 3): 'm',    # left_eye to left_ear
            (2, 4): 'm',    # right_eye to right_ear
            (0, 5): 'm',    # nose to left_shoulder
            (0, 6): 'm',    # nose to right_shoulder
            (5, 6): 'y',    # left_shoulder to right_shoulder
            (5, 7): 'y',    # left_shoulder to left_elbow
            (7, 9): 'y',    # left_elbow to left_wrist
            (6, 8): 'y',    # right_shoulder to right_elbow
            (8, 10): 'y',   # right_elbow to right_wrist
            (5, 11): 'g',   # left_shoulder to left_hip
            (6, 12): 'g',   # right_shoulder to right_hip
            (11, 12): 'g',  # left_hip to right_hip
            (11, 13): 'r',  # left_hip to left_knee
            (13, 15): 'r',  # left_knee to left_ankle
            (12, 14): 'r',  # right_hip to right_knee
            (14, 16): 'r'   # right_knee to right_ankle
        }
        
        # Color mapping
        self.COLOR_MAP = {
            'm': (255, 0, 255),   # magenta
            'y': (255, 255, 0),   # yellow
            'g': (0, 255, 0),     # green
            'r': (0, 0, 255)      # red
        }
        logger.info("Pose estimator initialized with motion tracking")

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
            # Resize and pad the image to keep the aspect ratio and fit the expected size
            input_image = tf.expand_dims(frame, axis=0)
            input_image = tf.image.resize_with_pad(input_image, self.input_size, self.input_size)
            
            # Convert to int32 (expected by the model)
            input_image = tf.cast(input_image, dtype=tf.int32)
            
            # Run model inference
            outputs = self.movenet(input_image)
            
            # Process the outputs
            keypoints = outputs['output_0'].numpy()
            keypoints = np.squeeze(keypoints)  # Remove batch dimension
            
            # Convert keypoints to the correct format
            processed_keypoints = []
            height, width = frame.shape[:2]
            
            for kp in keypoints:
                y, x, confidence = kp
                if confidence > self.confidence_threshold:
                    # Convert normalized coordinates to pixel coordinates
                    x_px = int(x * width)
                    y_px = int(y * height)
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
            # Get original frame dimensions
            height, width = frame.shape[:2]
            
            # Resize frame to model input size while maintaining aspect ratio
            input_image = tf.expand_dims(frame, axis=0)
            input_image = tf.image.resize_with_pad(input_image, self.input_size, self.input_size)
            
            # Convert to int32 (expected by the model)
            input_image = tf.cast(input_image, dtype=tf.int32)
            
            # Run model inference
            outputs = self.movenet(input_image)
            keypoints = outputs['output_0'].numpy()
            keypoints = np.squeeze(keypoints)  # Remove batch dimension
            
            # Convert normalized coordinates to pixel coordinates
            processed_keypoints = []
            
            # Get padding info from resize_with_pad
            scale = min(self.input_size / height, self.input_size / width)
            scaled_height = int(height * scale)
            scaled_width = int(width * scale)
            pad_height = (self.input_size - scaled_height) // 2
            pad_width = (self.input_size - scaled_width) // 2
            
            for kp in keypoints:
                y, x, confidence = kp
                if confidence > self.confidence_threshold:
                    # Remove padding and scale back to original dimensions
                    x_orig = ((x * self.input_size - pad_width) / scaled_width) * width
                    y_orig = ((y * self.input_size - pad_height) / scaled_height) * height
                    
                    # Ensure coordinates are within frame bounds
                    x_orig = np.clip(x_orig, 0, width - 1)
                    y_orig = np.clip(y_orig, 0, height - 1)
                    
                    processed_keypoints.append([x_orig, y_orig, confidence])
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
            
            # Draw keypoints
            for i, keypoint in enumerate(keypoints):
                if keypoint is not None and len(keypoint) >= 2:  # Check if keypoint exists and has x,y coordinates
                    x, y = int(keypoint[0]), int(keypoint[1])
                    cv2.circle(annotated_frame, (x, y), 5, (0, 255, 0), -1)
            
            # Draw connections
            connections = [
                [0, 1],  # nose to left eye
                [0, 2],  # nose to right eye
                [1, 3],  # left eye to left ear
                [2, 4],  # right eye to right ear
                [5, 6],  # left shoulder to right shoulder
                [5, 7],  # left shoulder to left elbow
                [7, 9],  # left elbow to left wrist
                [6, 8],  # right shoulder to right elbow
                [8, 10], # right elbow to right wrist
                [5, 11], # left shoulder to left hip
                [6, 12], # right shoulder to right hip
                [11, 12], # left hip to right hip
                [11, 13], # left hip to left knee
                [13, 15], # left knee to left ankle
                [12, 14], # right hip to right knee
                [14, 16]  # right knee to right ankle
            ]
            
            for start_idx, end_idx in connections:
                if (start_idx < len(keypoints) and end_idx < len(keypoints) and 
                    keypoints[start_idx] is not None and keypoints[end_idx] is not None and
                    len(keypoints[start_idx]) >= 2 and len(keypoints[end_idx]) >= 2):
                    start_point = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
                    end_point = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
                    cv2.line(annotated_frame, start_point, end_point, (255, 255, 0), 2)
            
            # Draw angles
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            
            # Draw sword arm angle
            if (angles.sword_arm_angle is not None and 8 < len(keypoints) and 
                keypoints[8] is not None and len(keypoints[8]) >= 2):
                sword_elbow_idx = 8  # Right elbow
                x = int(keypoints[sword_elbow_idx][0])
                y = int(keypoints[sword_elbow_idx][1])
                cv2.putText(annotated_frame, f"Sword: {angles.sword_arm_angle:.1f}°",
                           (x + 10, y), font, font_scale, (0, 255, 0), thickness)
            
            # Draw front knee angle
            if (angles.front_knee_angle is not None and 14 < len(keypoints) and 
                keypoints[14] is not None and len(keypoints[14]) >= 2):
                front_knee_idx = 14  # Right knee
                x = int(keypoints[front_knee_idx][0])
                y = int(keypoints[front_knee_idx][1])
                cv2.putText(annotated_frame, f"Front knee: {angles.front_knee_angle:.1f}°",
                           (x + 10, y), font, font_scale, (0, 255, 0), thickness)
            
            # Draw back knee angle
            if (angles.back_knee_angle is not None and 13 < len(keypoints) and 
                keypoints[13] is not None and len(keypoints[13]) >= 2):
                back_knee_idx = 13  # Left knee
                x = int(keypoints[back_knee_idx][0])
                y = int(keypoints[back_knee_idx][1])
                cv2.putText(annotated_frame, f"Back knee: {angles.back_knee_angle:.1f}°",
                           (x + 10, y), font, font_scale, (0, 255, 0), thickness)
            
            # Draw torso angle
            if (angles.torso_angle is not None and 5 < len(keypoints) and 6 < len(keypoints) and 
                keypoints[5] is not None and keypoints[6] is not None and
                len(keypoints[5]) >= 2 and len(keypoints[6]) >= 2):
                mid_shoulder_x = int((keypoints[5][0] + keypoints[6][0]) / 2)  # Average of left and right shoulder
                mid_shoulder_y = int((keypoints[5][1] + keypoints[6][1]) / 2)
                cv2.putText(annotated_frame, f"Torso: {angles.torso_angle:.1f}°",
                           (mid_shoulder_x + 10, mid_shoulder_y), font, font_scale, (0, 255, 0), thickness)
            
            return annotated_frame
            
        except Exception as e:
            logger.error(f"[Pose] Error drawing pose: {str(e)}")
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

    def _prepare_image(self, image: np.ndarray) -> tf.Tensor:
        """Prepare image for model input."""
        try:
            # Convert to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            # Convert to float32 in range [0, 1]
            input_image = tf.convert_to_tensor(image_rgb)
            input_image = tf.cast(input_image, tf.float32)
            input_image = input_image / 255.0
            
            # Add batch dimension and resize
            input_image = tf.expand_dims(input_image, axis=0)
            input_image = tf.image.resize_with_pad(input_image, self.input_size, self.input_size)
            
            # Scale back to [0, 255] and convert to int32
            input_image = tf.cast(input_image * 255.0, tf.int32)
            
            return input_image
            
        except Exception as e:
            logger.error(f"Error preparing image: {str(e)}")
            raise

    def process_image(self, image: np.ndarray, 
                     min_confidence: float = 0.3) -> Tuple[np.ndarray, Optional[JointAngles]]:
        """Process image and return annotated image with pose and joint angles."""
        try:
            # Log image properties
            logger.info(f"Processing image with shape: {image.shape}")
            
            # Prepare image
            input_image = self._prepare_image(image)
            logger.info(f"Prepared input image shape: {input_image.shape}, dtype: {input_image.dtype}")
            
            # Run inference - pass tensor directly without dictionary
            outputs = self.movenet(input_image)
            keypoints = outputs['output_0'].numpy().squeeze()
            logger.info(f"Got keypoints with shape: {keypoints.shape}")
            
            # Create output image
            output_image = image.copy()
            height, width = image.shape[:2]
            
            # Convert normalized coordinates to pixel coordinates
            keypoints_px = []
            for kp in keypoints:
                y, x, conf = kp
                if conf > min_confidence:
                    x_px = int(x * width)
                    y_px = int(y * height)
                    keypoints_px.append([x_px, y_px, conf])
                else:
                    keypoints_px.append(None)
            
            # Draw keypoints and connections
            joint_angles = None
            if any(kp is not None for kp in keypoints_px):
                logger.info(f"Found keypoints above confidence threshold")
                # Draw edges
                for (start_idx, end_idx), color_code in self.KEYPOINT_EDGES.items():
                    if (keypoints_px[start_idx] is not None and 
                        keypoints_px[end_idx] is not None):
                        try:
                            start_point = keypoints_px[start_idx][:2]
                            end_point = keypoints_px[end_idx][:2]
                            cv2.line(output_image, 
                                    (int(start_point[0]), int(start_point[1])),
                                    (int(end_point[0]), int(end_point[1])),
                                    self.COLOR_MAP[color_code], 2)
                        except Exception as e:
                            logger.error(f"Error drawing line for keypoints {start_idx}-{end_idx}: {str(e)}")
            
                # Calculate joint angles
                try:
                    joint_angles = self._calculate_joint_angles(keypoints_px)
                    logger.info("Successfully calculated joint angles")
                except Exception as e:
                    logger.error(f"Error calculating joint angles: {str(e)}")
                    joint_angles = JointAngles()
            else:
                logger.warning("No keypoints found above confidence threshold")
            
            return output_image, joint_angles
            
        except Exception as e:
            logger.error(f"Error in pose estimation: {str(e)}", exc_info=True)
            return image.copy(), JointAngles()  # Return original image and empty angles on error
    
    def process_video(self, video_path: str, output_path: str = None,
                     min_confidence: float = 0.3) -> List[JointAngles]:
        """Process video and return list of joint angles for each frame."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"Video properties - Width: {width}, Height: {height}, FPS: {fps}, Total frames: {total_frames}")
            
            # Setup video writer if output path is provided
            writer = None
            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                logger.info(f"Video writer initialized for output: {output_path}")
            
            joint_angles_list = []
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    logger.info(f"Finished reading video after {frame_count} frames")
                    break
                
                frame_count += 1
                if frame_count % 10 == 0:  # Log progress every 10 frames
                    logger.info(f"Processing frame {frame_count}/{total_frames}")
                
                try:
                    # Process frame
                    annotated_frame, joint_angles = self.process_image(frame, min_confidence)
                    
                    if joint_angles:
                        joint_angles_list.append(joint_angles)
                    
                    # Write frame if output path is provided
                    if writer:
                        writer.write(annotated_frame)
                        
                except Exception as e:
                    logger.error(f"Error processing frame {frame_count}: {str(e)}", exc_info=True)
                    continue
                
            logger.info(f"Processed {frame_count} frames, got {len(joint_angles_list)} sets of joint angles")
            
        except Exception as e:
            logger.error(f"Error in video processing: {str(e)}", exc_info=True)
            raise
            
        finally:
            if cap is not None:
                cap.release()
            if writer is not None:
                writer.release()
        
        return joint_angles_list
