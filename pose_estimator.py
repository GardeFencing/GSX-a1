import cv2
import numpy as np
import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class JointAngles:
    """Class to store joint angles from pose estimation"""
    sword_arm_angle: Optional[float] = None
    front_knee_angle: Optional[float] = None
    back_knee_angle: Optional[float] = None
    torso_angle: Optional[float] = None

class PoseEstimator:
    """Estimate poses from video frames for fencing analysis"""
    def __init__(self):
        # Initialize OpenCV's DNN module with COCO model
        self.net = cv2.dnn.readNetFromCaffe(
            "models/pose/pose_deploy.prototxt",
            "models/pose/pose_iter_584000.caffemodel"
        )
        
        # Define body parts
        self.BODY_PARTS = {
            "Nose": 0, "Neck": 1,
            "RShoulder": 2, "RElbow": 3, "RWrist": 4,
            "LShoulder": 5, "LElbow": 6, "LWrist": 7,
            "RHip": 8, "RKnee": 9, "RAnkle": 10,
            "LHip": 11, "LKnee": 12, "LAnkle": 13
        }
        
        self.POSE_PAIRS = [
            ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
            ["RElbow", "RWrist"], ["Neck", "LShoulder"],
            ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
            ["Neck", "RHip"], ["RHip", "RKnee"],
            ["RKnee", "RAnkle"], ["Neck", "LHip"],
            ["LHip", "LKnee"], ["LKnee", "LAnkle"]
        ]
        
        logger.info("Pose estimator initialized with OpenCV DNN")
    
    def estimate_pose(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[JointAngles]]:
        """
        Estimate pose from a video frame
        
        Args:
            frame (np.ndarray): Input video frame
            
        Returns:
            Tuple[Optional[np.ndarray], Optional[JointAngles]]: 
                Detected keypoints and calculated joint angles
        """
        try:
            # Prepare input
            input_height, input_width = 368, 368
            frame_height, frame_width = frame.shape[:2]
            
            # Create blob from image
            input_blob = cv2.dnn.blobFromImage(
                frame, 1.0 / 255,
                (input_width, input_height),
                (0, 0, 0), swapRB=False, crop=False
            )
            
            # Forward pass
            self.net.setInput(input_blob)
            output = self.net.forward()
            
            # Extract keypoints
            keypoints = []
            for i in range(len(self.BODY_PARTS)):
                # Get heatmap
                prob_map = output[0, i, :, :]
                prob_map = cv2.resize(prob_map, (frame_width, frame_height))
                
                # Find global maxima
                _, confidence, _, point = cv2.minMaxLoc(prob_map)
                
                # Add keypoint if confidence is high enough
                if confidence > 0.1:
                    keypoints.append(point)
                else:
                    keypoints.append(None)
            
            # Calculate joint angles
            angles = self._calculate_angles(keypoints)
            
            return np.array(keypoints), angles
            
        except Exception as e:
            logger.error(f"Error in pose estimation: {str(e)}")
            return None, None
    
    def _calculate_angles(self, keypoints: List[Tuple[int, int]]) -> JointAngles:
        """Calculate joint angles from keypoints"""
        angles = JointAngles()
        
        try:
            # Calculate sword arm angle (right arm)
            if all(keypoints[i] for i in [self.BODY_PARTS["RShoulder"], 
                                        self.BODY_PARTS["RElbow"], 
                                        self.BODY_PARTS["RWrist"]]):
                shoulder = keypoints[self.BODY_PARTS["RShoulder"]]
                elbow = keypoints[self.BODY_PARTS["RElbow"]]
                wrist = keypoints[self.BODY_PARTS["RWrist"]]
                
                angles.sword_arm_angle = self._angle_between_points(shoulder, elbow, wrist)
            
            # Calculate front knee angle (right leg)
            if all(keypoints[i] for i in [self.BODY_PARTS["RHip"], 
                                        self.BODY_PARTS["RKnee"], 
                                        self.BODY_PARTS["RAnkle"]]):
                hip = keypoints[self.BODY_PARTS["RHip"]]
                knee = keypoints[self.BODY_PARTS["RKnee"]]
                ankle = keypoints[self.BODY_PARTS["RAnkle"]]
                
                angles.front_knee_angle = self._angle_between_points(hip, knee, ankle)
            
            # Calculate back knee angle (left leg)
            if all(keypoints[i] for i in [self.BODY_PARTS["LHip"], 
                                        self.BODY_PARTS["LKnee"], 
                                        self.BODY_PARTS["LAnkle"]]):
                hip = keypoints[self.BODY_PARTS["LHip"]]
                knee = keypoints[self.BODY_PARTS["LKnee"]]
                ankle = keypoints[self.BODY_PARTS["LAnkle"]]
                
                angles.back_knee_angle = self._angle_between_points(hip, knee, ankle)
            
            # Calculate torso angle
            if all(keypoints[i] for i in [self.BODY_PARTS["Neck"], 
                                        self.BODY_PARTS["RHip"]]):
                neck = keypoints[self.BODY_PARTS["Neck"]]
                hip = keypoints[self.BODY_PARTS["RHip"]]
                
                # Calculate angle with vertical
                vertical = (neck[0], neck[1] - 100)  # Point above neck
                angles.torso_angle = self._angle_between_points(vertical, neck, hip)
            
            return angles
            
        except Exception as e:
            logger.error(f"Error calculating angles: {str(e)}")
            return angles
    
    def _angle_between_points(self, p1: Tuple[int, int], p2: Tuple[int, int], 
                            p3: Tuple[int, int]) -> float:
        """Calculate angle between three points"""
        try:
            # Convert points to vectors
            v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
            v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
            
            # Calculate angle
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            
            return np.degrees(angle)
            
        except Exception as e:
            logger.error(f"Error calculating angle between points: {str(e)}")
            return 0.0
    
    def draw_pose(self, frame: np.ndarray, keypoints: np.ndarray, 
                 angles: JointAngles) -> np.ndarray:
        """Draw pose skeleton and angles on frame"""
        try:
            # Draw connections
            for pair in self.POSE_PAIRS:
                idx1 = self.BODY_PARTS[pair[0]]
                idx2 = self.BODY_PARTS[pair[1]]
                
                if keypoints[idx1] is not None and keypoints[idx2] is not None:
                    pt1 = keypoints[idx1]
                    pt2 = keypoints[idx2]
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
            
            # Draw keypoints
            for i, point in enumerate(keypoints):
                if point is not None:
                    cv2.circle(frame, point, 5, (0, 0, 255), -1)
            
            # Add angle annotations
            if angles.sword_arm_angle is not None:
                cv2.putText(frame, f"Sword: {angles.sword_arm_angle:.1f}°",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if angles.front_knee_angle is not None:
                cv2.putText(frame, f"Front Knee: {angles.front_knee_angle:.1f}°",
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if angles.back_knee_angle is not None:
                cv2.putText(frame, f"Back Knee: {angles.back_knee_angle:.1f}°",
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if angles.torso_angle is not None:
                cv2.putText(frame, f"Torso: {angles.torso_angle:.1f}°",
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            return frame
            
        except Exception as e:
            logger.error(f"Error drawing pose: {str(e)}")
            return frame
