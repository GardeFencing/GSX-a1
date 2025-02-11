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

        self.input_size = 256  # Thunder model expects 256x256 input
        self.confidence_threshold = 0.1  # Lower threshold to help detect legs

        # We'll keep the simpler approach now: no advanced "leg side" logic
        # If you want to do smoothing, you can still keep buffers or use tracking
        self.leg_position_buffer = deque(maxlen=5)

        # Set up model cache directory
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache")
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["TFHUB_CACHE_DIR"] = cache_dir

        try:
            # Model handle for MoveNet Thunder - more robust than Lightning (though slower)
            model_handle = "https://tfhub.dev/google/movenet/singlepose/thunder/4"
            model_path = os.path.join(cache_dir, "movenet_thunder")

            logger.info("Downloading MoveNet Thunder model...")
            model = hub.load(model_handle)
            self.movenet = model.signatures['serving_default']

            logger.info("Saving model to cache...")
            tf.saved_model.save(model, model_path)

            input_details = list(self.movenet.structured_input_signature[1].keys())[0]
            logger.info(f"Model loaded successfully. Input tensor name: {input_details}")
            self.input_tensor_name = input_details

        except Exception as e:
            logger.error(f"Error loading MoveNet model: {str(e)}")
            raise RuntimeError(f"Failed to initialize MoveNet model: {str(e)}")

        # Initialize motion tracking
        self.sequence_length = sequence_length
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
        Convert normalized coordinates (0-1) from MoveNet output to original image pixel coordinates,
        taking into account the same resize_with_pad approach that keeps aspect ratio and adds padding.
        """
        target = self.input_size  # 256 for Thunder model
        scale = min(target / orig_height, target / orig_width)

        scaled_height = int(orig_height * scale)
        scaled_width = int(orig_width * scale)

        pad_height = target - scaled_height
        pad_width = target - scaled_width
        pad_top = pad_height // 2
        pad_left = pad_width // 2

        x_padded = x_norm * target
        y_padded = y_norm * target

        x_scaled = x_padded - pad_left
        y_scaled = y_padded - pad_top

        x_orig = x_scaled / scale
        y_orig = y_scaled / scale

        x_orig = int(np.clip(x_orig, 0, orig_width - 1))
        y_orig = int(np.clip(y_orig, 0, orig_height - 1))

        return x_orig, y_orig

    def _calculate_angle(
        self,
        p1: Optional[Tuple[float, float, float]], 
        p2: Optional[Tuple[float, float, float]], 
        p3: Optional[Tuple[float, float, float]]
    ) -> Tuple[float, float]:
        """Calculate the angle between three points and return the angle + average confidence."""
        if not all([p1, p2, p3]):
            return 0.0, 0.0
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        confidence = np.mean([p1[2], p2[2], p3[2]])
        return angle, confidence

    def _calculate_biomechanical_scores(self, angles: JointAngles) -> dict:
        """Calculate biomechanical quality scores (example implementation)."""
        en_garde_scores = {
            'sword_arm': 1 - abs(angles.sword_arm_angle - self.ideal_angles['en_garde']['sword_arm_angle']) / 180 if angles.sword_arm_angle else 0,
            'front_knee': 1 - abs(angles.front_knee_angle - self.ideal_angles['en_garde']['front_knee_angle']) / 180 if angles.front_knee_angle else 0,
            'back_knee': 1 - abs(angles.back_knee_angle - self.ideal_angles['en_garde']['back_knee_angle']) / 180 if angles.back_knee_angle else 0,
            'torso': 1 - abs(angles.torso_angle - self.ideal_angles['en_garde']['torso_angle']) / 90 if angles.torso_angle else 0,
        }
        lunge_scores = {
            'sword_arm': 1 - abs(angles.sword_arm_angle - self.ideal_angles['lunge']['sword_arm_angle']) / 180 if angles.sword_arm_angle else 0,
            'front_knee': 1 - abs(angles.front_knee_angle - self.ideal_angles['lunge']['front_knee_angle']) / 180 if angles.front_knee_angle else 0,
            'back_knee': 1 - abs(angles.back_knee_angle - self.ideal_angles['lunge']['back_knee_angle']) / 180 if angles.back_knee_angle else 0,
            'torso': 1 - abs(angles.torso_angle - self.ideal_angles['lunge']['torso_angle']) / 90 if angles.torso_angle else 0,
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
            sword_arm_vel = (curr_angles.sword_arm_angle or 0) - (prev_angles.sword_arm_angle or 0)
            front_knee_vel = (curr_angles.front_knee_angle or 0) - (prev_angles.front_knee_angle or 0)
            angle_velocities.append((sword_arm_vel, front_knee_vel))

        avg_sword_vel = np.mean([v[0] for v in angle_velocities])
        avg_knee_vel = np.mean([v[1] for v in angle_velocities])

        # Example classification logic
        if abs(avg_sword_vel) < 5 and abs(avg_knee_vel) < 5:
            phase = "preparation"
        elif avg_sword_vel > 10 or avg_knee_vel < -10:
            phase = "attack"
        else:
            phase = "recovery"

        motion_score = np.mean([angles.confidence for _, angles, _ in self.sequence_buffer])
        technical_notes = []
        if phase == "attack":
            if end_angles.sword_arm_angle and end_angles.sword_arm_angle < 160:
                technical_notes.append("Extend sword arm fully during attack")
            if end_angles.front_knee_angle and end_angles.front_knee_angle > 100:
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

    def _analyze_fencing_pose(self, keypoints: List[Optional[Tuple[float, float, float]]], frame_idx: int) -> Tuple[JointAngles, dict]:
        """Analyze the fencing pose from the detected keypoints."""
        right_shoulder = keypoints[KEYPOINT_DICT['right_shoulder']]
        right_elbow = keypoints[KEYPOINT_DICT['right_elbow']]
        right_wrist = keypoints[KEYPOINT_DICT['right_wrist']]
        right_hip = keypoints[KEYPOINT_DICT['right_hip']]
        right_knee = keypoints[KEYPOINT_DICT['right_knee']]
        right_ankle = keypoints[KEYPOINT_DICT['right_ankle']]
        left_hip = keypoints[KEYPOINT_DICT['left_hip']]
        left_knee = keypoints[KEYPOINT_DICT['left_knee']]
        left_ankle = keypoints[KEYPOINT_DICT['left_ankle']]

        # Sword arm angle
        sword_arm, sa_conf = self._calculate_angle(right_shoulder, right_elbow, right_wrist)
        # Front knee angle
        front_knee, fk_conf = self._calculate_angle(right_hip, right_knee, right_ankle)
        # Back knee angle
        back_knee, bk_conf = self._calculate_angle(left_hip, left_knee, left_ankle)

        # Torso angle
        if right_shoulder and right_hip:
            torso_vector = np.array([right_shoulder[0] - right_hip[0], right_shoulder[1] - right_hip[1]])
            vertical = np.array([0, -1])
            torso_cos = np.dot(torso_vector, vertical) / (np.linalg.norm(torso_vector) * np.linalg.norm(vertical))
            torso_angle = np.degrees(np.arccos(np.clip(torso_cos, -1.0, 1.0)))
            torso_conf = (right_shoulder[2] + right_hip[2]) / 2
        else:
            torso_angle, torso_conf = 0.0, 0.0

        # Stance width
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

        # If the sequence buffer is full, analyze the overall movement
        if len(self.sequence_buffer) == self.sequence_buffer.maxlen:
            self._analyze_motion_sequence()

        return angles, scores

    def _calculate_joint_angles(self, keypoints):
        """Calculate joint angles from keypoints (pixel coords, [x, y, conf])."""
        try:
            angles = JointAngles()

            def calc_angle(p1, p2, p3):
                if not p1 or not p2 or not p3:
                    return None
                p1_xy = np.array([p1[0], p1[1]])
                p2_xy = np.array([p2[0], p2[1]])
                p3_xy = np.array([p3[0], p3[1]])

                v1 = p1_xy - p2_xy
                v2 = p3_xy - p2_xy
                dot = np.dot(v1, v2)
                denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
                if denom < 1e-6:
                    return None
                cos_angle = np.clip(dot / denom, -1.0, 1.0)
                angle_rad = np.arccos(cos_angle)
                return np.degrees(angle_rad)

            # Sword arm (right_shoulder=6, right_elbow=8, right_wrist=10)
            angles.sword_arm_angle = calc_angle(keypoints[6], keypoints[8], keypoints[10])

            # Front knee (right_hip=12, right_knee=14, right_ankle=16)
            angles.front_knee_angle = calc_angle(keypoints[12], keypoints[14], keypoints[16])

            # Back knee (left_hip=11, left_knee=13, left_ankle=15)
            angles.back_knee_angle = calc_angle(keypoints[11], keypoints[13], keypoints[15])

            # Torso angle (mid-shoulders vs. mid-hips relative to vertical)
            if all(keypoints[i] is not None for i in [5, 6, 11, 12]):
                shoulder_mid = np.mean([
                    [keypoints[5][0], keypoints[5][1]],
                    [keypoints[6][0], keypoints[6][1]]
                ], axis=0)
                hip_mid = np.mean([
                    [keypoints[11][0], keypoints[11][1]],
                    [keypoints[12][0], keypoints[12][1]]
                ], axis=0)
                torso_vec = shoulder_mid - hip_mid
                vertical = np.array([0, -1])
                dot = np.dot(torso_vec, vertical)
                denom = (np.linalg.norm(torso_vec) * np.linalg.norm(vertical))
                if denom > 1e-6:
                    cos_angle = np.clip(dot / denom, -1.0, 1.0)
                    angles.torso_angle = np.degrees(np.arccos(cos_angle))

            # Stance width (distance between ankles)
            if keypoints[15] is not None and keypoints[16] is not None:
                left_ankle = np.array([keypoints[15][0], keypoints[15][1]])
                right_ankle = np.array([keypoints[16][0], keypoints[16][1]])
                angles.stance_width = np.linalg.norm(right_ankle - left_ankle)

            # Overall confidence
            valid_confs = [kp[2] for kp in keypoints if kp is not None]
            angles.confidence = np.mean(valid_confs) if valid_confs else 0.0

            return angles

        except Exception as e:
            logger.error(f"Error calculating joint angles: {str(e)}")
            return JointAngles()

    def _get_keypoints(self, frame: np.ndarray) -> List[Optional[List[float]]]:
        """
        Get keypoints from frame using MoveNet Thunder with correct preprocessing.
        Note: The Thunder model expects an input tensor of type int32.
        """
        try:
            # Instead of converting to float32 and normalizing, use the raw image.
            # The frame is expected to be in RGB and of type uint8.
            img = tf.expand_dims(frame, axis=0)
            img = tf.image.resize_with_pad(img, 
                target_height=self.input_size,
                target_width=self.input_size
            )
            img = tf.cast(img, dtype=tf.int32)

            # Forward pass
            outputs = self.movenet(**{self.input_tensor_name: img})
            keypoints = outputs['output_0'].numpy()
            keypoints = np.squeeze(keypoints)

            # Scale up to the original frame size
            orig_height, orig_width = frame.shape[:2]
            processed_keypoints = []

            for (y_norm, x_norm, conf) in keypoints:
                if conf > self.confidence_threshold:
                    # Convert to pixel coordinates
                    x_px, y_px = self._convert_normalized_to_pixel(
                        x_norm, y_norm, orig_width, orig_height
                    )
                    processed_keypoints.append([x_px, y_px, conf])
                else:
                    processed_keypoints.append(None)

            return processed_keypoints

        except Exception as e:
            logger.error(f"Error in pose estimation: {str(e)}")
            return [None] * 17

    def estimate_pose(self, frame: np.ndarray) -> Tuple[List[Optional[List[float]]], JointAngles]:
        """
        Estimate pose in a frame and return keypoints with joint angles.
        """
        try:
            # Convert BGR → RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Get keypoints using the fixed function
            keypoints = self._get_keypoints(frame_rgb)

            # Compute angles
            angles = self._calculate_joint_angles(keypoints)
            return keypoints, angles

        except Exception as e:
            logger.error(f"Error in pose estimation: {str(e)}")
            return [None] * 17, JointAngles()

    def draw_pose(self, frame: np.ndarray, keypoints: List[Optional[List[float]]], angles: JointAngles) -> np.ndarray:
        """Draw pose keypoints and angles on the frame exactly matching reference implementation."""
        try:
            KEYPOINT_EDGE_INDS_TO_COLOR = {
                (0, 1): (255, 0, 255),  # magenta
                (0, 2): (0, 255, 255),  # cyan
                (1, 3): (255, 0, 255),
                (2, 4): (0, 255, 255),
                (0, 5): (255, 0, 255),
                (0, 6): (0, 255, 255),
                (5, 7): (255, 0, 255),
                (7, 9): (255, 0, 255),
                (6, 8): (0, 255, 255),
                (8, 10): (0, 255, 255),
                (5, 6): (255, 255, 0),
                (5, 11): (255, 0, 255),
                (6, 12): (0, 255, 255),
                (11, 12): (255, 255, 0),
                (11, 13): (255, 0, 255),
                (13, 15): (255, 0, 255),
                (12, 14): (0, 255, 255),
                (14, 16): (0, 255, 255)
            }

            annotated_frame = frame.copy()

            # Draw edges
            for (v1, v2), edge_color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
                if keypoints[v1] is not None and keypoints[v2] is not None:
                    if (keypoints[v1][2] > self.confidence_threshold and
                        keypoints[v2][2] > self.confidence_threshold):
                        pt1 = (int(keypoints[v1][0]), int(keypoints[v1][1]))
                        pt2 = (int(keypoints[v2][0]), int(keypoints[v2][1]))
                        cv2.line(annotated_frame, pt1, pt2, edge_color, 2)

            # Draw keypoints
            for kp in keypoints:
                if kp is not None and kp[2] > self.confidence_threshold:
                    x, y = int(kp[0]), int(kp[1])
                    cv2.circle(annotated_frame, (x, y), 3, (0, 255, 0), -1)

            # Draw angles
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2

            def put_outlined_text(img, text, pos):
                cv2.putText(img, text, pos, font, font_scale, (0, 0, 0), thickness + 1)
                cv2.putText(img, text, pos, font, font_scale, (255, 255, 255), thickness)

            # Angle annotations near relevant joints
            if angles.sword_arm_angle is not None and keypoints[8] is not None:
                x, y = int(keypoints[8][0]), int(keypoints[8][1])
                txt = f"{angles.sword_arm_angle:.1f}°"
                put_outlined_text(annotated_frame, txt, (x + 10, y))

            if angles.front_knee_angle is not None and keypoints[14] is not None:
                x, y = int(keypoints[14][0]), int(keypoints[14][1])
                txt = f"{angles.front_knee_angle:.1f}°"
                put_outlined_text(annotated_frame, txt, (x + 10, y))

            if angles.back_knee_angle is not None and keypoints[13] is not None:
                x, y = int(keypoints[13][0]), int(keypoints[13][1])
                txt = f"{angles.back_knee_angle:.1f}°"
                put_outlined_text(annotated_frame, txt, (x - 60, y))

            return annotated_frame

        except Exception as e:
            logger.error(f"Error drawing pose: {str(e)}")
            return frame
