import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        
        try:
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
            
        except Exception as e:
            logger.error(f"Error in biomechanical analysis: {str(e)}")
            return metrics
    
    def _analyze_power_chain(self, angles: Dict[str, float]) -> float:
        """Analyze power generation through kinetic chain"""
        try:
            # Check proper sequence activation (proximal to distal)
            power_scores = []
            ideal_sequence = ['back_knee', 'front_knee', 'torso', 'sword_arm']
            
            for i, joint in enumerate(ideal_sequence[:-1]):
                if joint in angles and ideal_sequence[i+1] in angles:
                    # Check if proximal joint moves before distal
                    score = 1.0 if angles[joint] > angles[ideal_sequence[i+1]] else 0.5
                    power_scores.append(score)
            
            return np.mean(power_scores) if power_scores else 0.0
            
        except Exception as e:
            logger.error(f"Error analyzing power chain: {str(e)}")
            return 0.0
    
    def _analyze_balance(self, angles: Dict[str, float]) -> float:
        """Analyze balance control"""
        try:
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
            
        except Exception as e:
            logger.error(f"Error analyzing balance: {str(e)}")
            return 0.0
    
    def _analyze_efficiency(self, angles: Dict[str, float]) -> float:
        """Analyze movement efficiency"""
        try:
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
            
        except Exception as e:
            logger.error(f"Error analyzing efficiency: {str(e)}")
            return 0.0
    
    def _analyze_coordination(self, angles: Dict[str, float]) -> float:
        """Analyze inter-joint coordination"""
        try:
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
            
        except Exception as e:
            logger.error(f"Error analyzing coordination: {str(e)}")
            return 0.0 