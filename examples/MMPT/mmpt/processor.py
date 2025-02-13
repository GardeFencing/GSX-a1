import os
import torch
import numpy as np
import cv2
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from transformers import CLIPProcessor, CLIPModel
import time

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class MMPTOutput:
    """Store MMPT analysis results"""
    action_segments: List[Dict[str, Any]]
    movement_scores: Dict[str, float]
    temporal_context: str
    confidence: float
    coaching_focus: Optional[str] = None
    sequence_feedback: Optional[str] = None

class MMPTProcessor:
    """Process video using MMPT for feature extraction and temporal understanding using multi-vector retrieval"""
    def __init__(self):
        try:
            # Initialize CLIP for visual understanding
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            
            # Initialize GardeRAG once
            from GardeRAG.model import GardeRAG
            self.rag = GardeRAG()
            logger.info("GardeRAG initialized")
            
            # Movement vector types for multi-vector retrieval
            self.movement_vectors = {
                'pose': [  # Pose-specific descriptors for retrieval
                    "fencer in en garde position with bent knees and balanced stance",
                    "fencer moving forward with leading foot while maintaining form",
                    "fencer moving backward with trailing foot while maintaining form",
                    "fencer executing lunge with extended arm and forward drive",
                    "fencer in defensive position with blade raised",
                    "fencer performing immediate counter-attack after defense",
                    "fencer returning to ready position after action"
                ],
                'spatial': [  # Spatial relationship descriptors
                    "fencers at long distance with defensive stances",
                    "fencers closing distance with careful footwork",
                    "fencers in close combat with blade engagement",
                    "fencers creating distance after exchange",
                    "fencer advancing while opponent retreats"
                ],
                'tactical': [  # Tactical situation descriptors
                    "fencer preparing attack with careful positioning",
                    "fencer setting up defensive trap or invitation",
                    "fencer exploiting tactical advantage with timing",
                    "fencer controlling center of strip with position",
                    "fencer using distance to control tempo of bout"
                ]
            }
            
            # Initialize analysis state with cache
            self.frame_buffer = []
            self.max_buffer_size = 15
            self.state = {
                'pose_vectors': [],      # Store pose movement embeddings
                'spatial_vectors': [],   # Store spatial relationship embeddings
                'tactical_vectors': [],  # Store tactical situation embeddings
                'raw_frames': [],        # Store original frame data
                'confidence_scores': [], # Store confidence for each vector type
                'cached_summaries': {},  # Cache for frame summaries
                'cached_embeddings': {}, # Cache for frame embeddings
                'last_query_time': 0,    # Rate limiting
                'query_interval': 0.5    # Minimum time between queries
            }
            
            logger.info("MMPT Processor initialized with multi-vector retrieval")
            
        except Exception as e:
            logger.error(f"Error initializing MMPT Processor: {str(e)}")
            raise

    def _select_keyframes(self, frames: List[np.ndarray], n_clusters: int = 10) -> List[int]:
        """Hierarchical Keyframe Selection using CLIP embeddings and clustering"""
        try:
            # Convert frames to CLIP input format
            processed_frames = [
                self.clip_processor(images=frame, return_tensors="pt")
                for frame in frames
            ]
            
            # Get frame embeddings
            frame_embeddings = []
            with torch.no_grad():
                for inputs in processed_frames:
                    features = self.clip_model.get_image_features(**inputs.to(self.device))
                    frame_embeddings.append(features.cpu().numpy())
            
            frame_embeddings = np.vstack(frame_embeddings)
            
            # Cluster similar frames
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=min(n_clusters, len(frames)), random_state=42)
            labels = kmeans.fit_predict(frame_embeddings)
            
            # Select representative frame from each cluster
            selected_indices = []
            for cluster in range(kmeans.n_clusters):
                cluster_indices = np.where(labels == cluster)[0]
                if len(cluster_indices) > 0:
                    # Select frame closest to cluster center
                    center = kmeans.cluster_centers_[cluster]
                    distances = np.linalg.norm(frame_embeddings[cluster_indices] - center, axis=1)
                    selected_indices.append(cluster_indices[np.argmin(distances)])
            
            return sorted(selected_indices)
            
        except Exception as e:
            logger.error(f"Error in keyframe selection: {str(e)}")
            return list(range(0, len(frames), max(1, len(frames) // n_clusters)))

    def analyze_movement(self, frame: np.ndarray) -> MMPTOutput:
        """Extract multi-vector features and analyze movement patterns"""
        try:
            # Convert frame to RGB if needed
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame
            
            # Process frame with CLIP for each vector type
            frame_features = {}
            movement_scores = {}
            
            # Check cache first
            frame_hash = hash(frame_rgb.tobytes())
            if frame_hash in self.state['cached_embeddings']:
                frame_features = self.state['cached_embeddings'][frame_hash]
                movement_scores = {k: v['scores'] for k, v in frame_features.items()}
            else:
                with torch.no_grad():
                    # Get base image features
                    inputs = self.clip_processor(
                        images=frame_rgb,
                        return_tensors="pt",
                        padding=True
                    ).to(self.device)
                    
                    image_features = self.clip_model.get_image_features(**inputs)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    
                    # Process each vector type
                    for vector_type, descriptors in self.movement_vectors.items():
                        # Get text features for this vector type
                        text_inputs = self.clip_processor(
                            text=descriptors,
                            return_tensors="pt",
                            padding=True
                        ).to(self.device)
                        
                        text_features = self.clip_model.get_text_features(**text_inputs)
                        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                        
                        # Calculate similarities
                        similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                        
                        # Store features and scores
                        frame_features[vector_type] = {
                            'features': image_features,
                            'scores': {
                                desc: float(score)
                                for desc, score in zip(descriptors, similarities[0])
                            }
                        }
                        movement_scores.update(frame_features[vector_type]['scores'])
                
                # Cache the results
                self.state['cached_embeddings'][frame_hash] = frame_features
            
            # Update state with new vectors
            self._update_state(frame_features, frame_rgb)
            
            # Generate temporal context using multi-vector analysis
            temporal_context = self._analyze_temporal_context()
            
            # Get action segments with multi-vector confidence
            action_segments = self._get_action_segments(frame_features)
            
            # Calculate overall confidence across vector types
            confidence = max(max(scores.values()) for scores in [f['scores'] for f in frame_features.values()])
            
            return MMPTOutput(
                action_segments=action_segments,
                movement_scores=movement_scores,
                temporal_context=temporal_context,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error in movement analysis: {str(e)}")
            return MMPTOutput([], {}, "Error in analysis", 0.0)
    
    def _update_state(self, frame_features: Dict, frame: np.ndarray):
        """Update state with new vector information"""
        try:
            # Update vector stores
            for vector_type in self.movement_vectors.keys():
                self.state[f'{vector_type}_vectors'].append(frame_features[vector_type])
                if len(self.state[f'{vector_type}_vectors']) > self.max_buffer_size:
                    self.state[f'{vector_type}_vectors'].pop(0)
            
            # Store raw frame
            self.state['raw_frames'].append(frame)
            if len(self.state['raw_frames']) > self.max_buffer_size:
                self.state['raw_frames'].pop(0)
            
            # Update confidence scores
            confidence_scores = {
                vector_type: max(features['scores'].values())
                for vector_type, features in frame_features.items()
            }
            self.state['confidence_scores'].append(confidence_scores)
            if len(self.state['confidence_scores']) > self.max_buffer_size:
                self.state['confidence_scores'].pop(0)
                
        except Exception as e:
            logger.error(f"Error updating state: {str(e)}")
    
    def _analyze_temporal_context(self) -> str:
        """Generate natural commentary using RAG based on multi-vector analysis with rate limiting"""
        if not any(self.state[f'{vtype}_vectors'] for vtype in self.movement_vectors.keys()):
            return "Starting analysis..."
            
        try:
            # Check rate limiting
            current_time = time.time()
            if current_time - self.state['last_query_time'] < self.state['query_interval']:
                # Return cached summary if available
                if self.state['cached_summaries']:
                    latest_summary = list(self.state['cached_summaries'].values())[-1]
                    return latest_summary
                return "Processing movement sequence..."
            
            # Collect recent movements with their scores
            recent_data = {vtype: [] for vtype in self.movement_vectors.keys()}
            
            # Get last 5 frames of data for each vector type
            for vtype in self.movement_vectors.keys():
                vectors = self.state[f'{vtype}_vectors'][-5:]
                for vector in vectors:
                    movements = sorted(vector['scores'].items(), key=lambda x: x[1], reverse=True)[:2]
                    for movement, score in movements:
                        if score > 0.4:  # Only include confident detections
                            recent_data[vtype].append((movement, score))
            
            # Create context for RAG
            context = {
                'movements': {
                    vtype: [{'description': m[0], 'confidence': m[1]} for m in moves]
                    for vtype, moves in recent_data.items() if moves
                },
                'sequence_length': len(self.state['raw_frames']),
                'average_confidence': {
                    vtype: np.mean([score for _, score in moves]) if moves else 0.0
                    for vtype, moves in recent_data.items()
                }
            }
            
            # Generate commentary using RAG
            prompt = """As an experienced fencing coach, describe what's happening in this sequence. Focus on:
1. The fencer's current movement and technique
2. Their tactical positioning and intent
3. Any technical feedback or coaching points

Keep the commentary natural and focused on the most important aspects."""
            
            try:
                result = self.rag.query(prompt, context)
                commentary = result.get("response", "Analyzing the sequence...")
                
                # Cache the summary
                frame_hash = hash(str(context))
                self.state['cached_summaries'][frame_hash] = commentary
                self.state['last_query_time'] = current_time
                
                return commentary
                
            except Exception as e:
                logger.error(f"Error using RAG: {str(e)}")
                return "Error generating commentary"
                
        except Exception as e:
            logger.error(f"Error in temporal analysis: {str(e)}")
            return "Analyzing movement patterns"
    
    def _get_action_segments(self, frame_features: Dict) -> List[Dict[str, Any]]:
        """Get action segments using multi-vector confidence"""
        segments = []
        threshold = 0.3
        
        try:
            # Combine high-confidence movements from all vector types
            for vector_type, features in frame_features.items():
                for movement, score in features['scores'].items():
                    if score > threshold:
                        segments.append({
                            "action": movement,
                            "vector_type": vector_type,
                            "confidence": float(score),
                            "is_key_moment": score > 0.6
                        })
            
            return sorted(segments, key=lambda x: x["confidence"], reverse=True)
            
        except Exception as e:
            logger.error(f"Error getting action segments: {str(e)}")
            return [] 