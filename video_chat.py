#!/usr/bin/env python3
"""
State-of-the-Art Fencing Bout Analysis and Video LLM
====================================================

Author: R
Date: 2025-02-03
"""

from __future__ import annotations
import os
import cv2
import torch
import numpy as np
import faiss
import logging
import base64
import time
import asyncio
import aiohttp
import json
import sys
import torch.nn.functional as F
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Dict, Optional, Any, TypeVar, Callable, AsyncIterable, Union
from PIL import Image
from tqdm import tqdm
import random
from dataclasses import dataclass
from contextlib import asynccontextmanager
from async_timeout import timeout
import requests

# Import models and estimators
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from pose_estimator import PoseEstimator, JointAngles

# ------------------------------------------------------------------------------
# RETRY AND RATE LIMITING SYSTEM
# ------------------------------------------------------------------------------

T = TypeVar('T')

@dataclass
class RetryConfig:
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    jitter_factor: float = 0.1
    timeout_seconds: int = 30
    concurrency_limit: int = 5
    rate_limit_calls: int = 10
    rate_limit_period: int = 60

class AdaptiveRetryHandler:
    """Advanced retry handler with adaptive backoff, rate limiting, and circuit breaking"""
    def __init__(self, config: RetryConfig = RetryConfig()):
        self.config = config
        self.failure_count = 0
        self.success_count = 0
        self.circuit_open = False
        self.last_failure_time = 0
        self.token_bucket = asyncio.Semaphore(config.concurrency_limit)
        
    async def _should_retry(self, attempt: int, exception: Exception) -> bool:
        if self.circuit_open:
            return False
            
        retryable_exceptions = (
            aiohttp.ClientError,
            asyncio.TimeoutError,
            ConnectionError
        )
        if not isinstance(exception, retryable_exceptions):
            return False
            
        current_time = time.time()
        if current_time - self.last_failure_time < 60:
            self.failure_count += 1
            if self.failure_count > 10:
                self.circuit_open = True
                asyncio.create_task(self._reset_circuit_breaker())
                return False
        else:
            self.failure_count = 1
            
        self.last_failure_time = current_time
        return attempt < self.config.max_attempts

    async def _reset_circuit_breaker(self) -> None:
        await asyncio.sleep(60)
        self.circuit_open = False
        self.failure_count = 0

    def _calculate_delay(self, attempt: int) -> float:
        base_delay = min(
            self.config.base_delay * (2 ** (attempt - 1)),
            self.config.max_delay
        )
        jitter = random.uniform(
            -self.config.jitter_factor * base_delay,
            self.config.jitter_factor * base_delay
        )
        return base_delay + jitter

    @asynccontextmanager
    async def rate_limit(self):
        await self.token_bucket.acquire()
        try:
            yield
        finally:
            self.token_bucket.release()

    async def execute_with_retry(
        self,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any
    ) -> T:
        attempt = 0
        
        while True:
            attempt += 1
            try:
                async with self.rate_limit():
                    async with timeout(self.config.timeout_seconds):
                        result = await func(*args, **kwargs)
                        self.success_count += 1
                        return result
                        
            except Exception as e:
                should_retry = await self._should_retry(attempt, e)
                if not should_retry:
                    raise
                    
                delay = self._calculate_delay(attempt)
                await asyncio.sleep(delay)

class BatchRequestHandler:
    """Handle multiple requests with batching and adaptive concurrency"""
    def __init__(
        self,
        retry_handler: AdaptiveRetryHandler,
        batch_size: int = 10,
        max_concurrent_batches: int = 3
    ):
        self.retry_handler = retry_handler
        self.batch_size = batch_size
        self.batch_semaphore = asyncio.Semaphore(max_concurrent_batches)

    async def process_batch(
        self,
        items: AsyncIterable[Any],
        process_func: Callable[[Any], T]
    ) -> AsyncIterable[T]:
        batch = []
        
        async for item in items:
            batch.append(item)
            
            if len(batch) >= self.batch_size:
                async with self.batch_semaphore:
                    results = await asyncio.gather(
                        *(self.retry_handler.execute_with_retry(process_func, item)
                          for item in batch)
                    )
                    for result in results:
                        yield result
                batch = []
                
        if batch:
            async with self.batch_semaphore:
                results = await asyncio.gather(
                    *(self.retry_handler.execute_with_retry(process_func, item)
                      for item in batch)
                )
                for result in results:
                    yield result

# Initialize retry handler with optimal configuration for video processing
retry_config = RetryConfig(
    max_attempts=3,
    base_delay=3.0,
    max_delay=30.0,
    concurrency_limit=5,
    rate_limit_calls=10,
    rate_limit_period=60
)
retry_handler = AdaptiveRetryHandler(retry_config)
batch_handler = BatchRequestHandler(retry_handler, batch_size=5, max_concurrent_batches=2)

# ------------------------------------------------------------------------------
# CONFIGURATION AND LOGGING SETUP
# ------------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# Global query string for caption generation (can be further tuned)
GLOBAL_QUERY = """You are a fencing coach. Give exactly TWO key points:
1. One specific technical correction
2. One improvement for better execution
Keep each point to one sentence."""

# Load models for analysis
logger.info("Loading sentence transformer for caption embeddings...")
sent_transformer = SentenceTransformer("all-MiniLM-L6-v2", device=device)

logger.info("Loading CLIP model and processor...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

# ------------------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------------------

def format_angle(angle: Optional[float]) -> str:
    """Safely format an angle value (if None, returns 'N/A')."""
    return f"{angle:.1f}" if angle is not None else "N/A"

def encode_image_base64(pil_image: Image.Image) -> str:
    """Convert a PIL Image to a base64 string (JPEG format)."""
    img_byte_arr = BytesIO()
    pil_image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    return base64.b64encode(img_byte_arr).decode('utf-8')

def safe_normalize(value: Optional[float]) -> float:
    """Normalize an angle value (assumes degrees) safely; returns 0.0 if None."""
    return (value / 180.0) if value is not None else 0.0

# ------------------------------------------------------------------------------
# VIDEO FEATURE EXTRACTOR (3D CNN based)
# ------------------------------------------------------------------------------

class VideoFeatureExtractor(torch.nn.Module):
    """A simplified video feature extractor based on 3D CNN principles."""
    def __init__(self, hidden_size: int = 512) -> None:
        super().__init__()
        self.conv3d = torch.nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool = torch.nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.flatten = torch.nn.Flatten()
        # Assuming an input shape that leads to a flattened size of 64*30*30*30
        self.fc = torch.nn.Linear(64 * 30 * 30 * 30, hidden_size)
        self.norm = torch.nn.LayerNorm(hidden_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv3d(x)
        x = torch.nn.functional.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.norm(x)
        return torch.nn.functional.normalize(x, p=2, dim=-1)

logger.info("Initializing video feature extractor...")
video_extractor = VideoFeatureExtractor().to(device)
video_extractor.eval()

# ------------------------------------------------------------------------------
# ASYNCHRONOUS RAG API CALL FUNCTIONS
# ------------------------------------------------------------------------------

async def async_get_rag_caption(
    session: aiohttp.ClientSession,
    image: Image.Image,
    angles: JointAngles,
    query: str,
    max_retries: int = 3,
    timeout: int = 30
) -> str:
    """
    Get caption from the RAG endpoint with fused pose information.
    Uses exponential backoff in case of errors.
    """
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                delay = 3 * (2 ** attempt)
                logger.info(f"[RAG] Retry attempt {attempt + 1}, waiting {delay} seconds...")
                await asyncio.sleep(delay)
            
            # Build an enhanced query with safe angle formatting
            enhanced_query = (
                f"{query}\n\nPose Analysis:"
                f"\n- Sword arm angle: {format_angle(getattr(angles, 'sword_arm_angle', None))}°"
                f"\n- Front knee angle: {format_angle(getattr(angles, 'front_knee_angle', None))}°"
                f"\n- Back knee angle: {format_angle(getattr(angles, 'back_knee_angle', None))}°"
                f"\n- Torso angle: {format_angle(getattr(angles, 'torso_angle', None))}°"
                f"\n- Stance width: {format_angle(getattr(angles, 'stance_width', None))}"
            )
            
            payload = {
                "query": enhanced_query,
                "image": encode_image_base64(image)
            }
            
            async with session.post("http://54.227.20.42/query", json=payload, timeout=timeout) as response:
                if response.status != 200:
                    logger.warning(f"[RAG] HTTP error {response.status} on attempt {attempt + 1}")
                    if attempt == max_retries - 1:
                        return f"Error: Server returned status {response.status}"
                    continue
                
                result = await response.json()
                caption = result.get("response", "No caption generated")
                
                if len(caption.split()) < 20:
                    if attempt < max_retries - 1:
                        logger.info("[RAG] Response too generic, retrying...")
                        continue
                return caption
                
        except asyncio.TimeoutError:
            logger.error("[RAG] Timeout error encountered.")
            if attempt < max_retries - 1:
                continue
            return "Analysis skipped (timeout)"
        except Exception as e:
            logger.error(f"[RAG] Error: {str(e)}")
            if attempt < max_retries - 1:
                continue
            return f"Error: {str(e)}"
    return "Coach feedback: Unable to generate feedback at this moment."

# ------------------------------------------------------------------------------
# VIDEO PROCESSING MODULE
# ------------------------------------------------------------------------------

class VideoProcessor:
    """
    Processes a video by extracting frames in chunks, applying pose estimation
    and annotation in parallel using a thread pool.
    """
    def __init__(self) -> None:
        self.pose_estimator = PoseEstimator()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.frame_cache = {}  # Cache for processed frames
        self.rag_cache = {}    # Cache for RAG API responses
        self.similarity_threshold = 0.92  # Threshold for frame similarity
        self.focus_topic = None  # Will be set from VideoChat
        self.rate_limit_delay = 12.0  # Delay in seconds to respect rate limit (5 per minute)
        
    def set_focus_topic(self, focus_topic: Optional[str]) -> None:
        """Set the focus topic for RAG queries."""
        self.focus_topic = focus_topic

    def _compute_frame_similarity(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Compute cosine similarity between CLIP embeddings of two frames."""
        with torch.no_grad():
            inputs1 = clip_processor(images=Image.fromarray(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)), 
                                   return_tensors="pt").to(device)
            inputs2 = clip_processor(images=Image.fromarray(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)), 
                                   return_tensors="pt").to(device)
            emb1 = clip_model.get_image_features(**inputs1)
            emb2 = clip_model.get_image_features(**inputs2)
            similarity = F.cosine_similarity(emb1, emb2).item()
        return similarity

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[JointAngles]]:
        """Process a single frame with caching and optimized pose estimation."""
        try:
            # Check cache first
            cache_key = hash(frame.tobytes())
            if cache_key in self.frame_cache:
                return self.frame_cache[cache_key]

            # Ensure frame is in correct format and size for MoveNet
            height, width = frame.shape[:2]
            # MoveNet expects 256x256 images
            input_size = 256
            scale_x = width / input_size
            scale_y = height / input_size
            
            # Resize frame for pose estimation
            frame_resized = cv2.resize(frame, (input_size, input_size))
            
            # Estimate pose on resized frame
            keypoints, angles = self.pose_estimator.estimate_pose(frame_resized)
            
            try:
                # Draw pose on original frame if keypoints exist
                if keypoints is not None and angles is not None:
                    # Create a copy of the frame for drawing
                    annotated_frame = frame.copy()
                    
                    # Scale keypoints back to original frame size
                    scaled_keypoints = keypoints.copy()
                    scaled_keypoints[:, 0] *= scale_x  # x coordinates
                    scaled_keypoints[:, 1] *= scale_y  # y coordinates
                    
                    # Draw pose only if all required values exist
                    if all(getattr(angles, attr, None) is not None for attr in 
                          ['sword_arm_angle', 'front_knee_angle', 'back_knee_angle', 'torso_angle']):
                        annotated_frame = self.pose_estimator.draw_pose(annotated_frame, scaled_keypoints, angles)
                    result = (annotated_frame, angles)
                else:
                    result = (frame, angles)
            except Exception as e:
                logger.error(f"[Pose] Error drawing pose: {str(e)}")
                result = (frame, angles)

            # Cache the result
            self.frame_cache[cache_key] = result
            return result

        except Exception as e:
            logger.error(f"[Pose] Error processing frame: {str(e)}")
            return frame, None

    def extract_frames(self, video_path: str, target_fps: int = 5) -> Tuple[List[Image.Image], List[Optional[JointAngles]], float]:
        """Extract frames using intelligent sampling and similarity detection."""
        logger.info(f"[VideoProcessor] Processing video with target {target_fps} FPS...")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Error opening video file")
        
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame sampling interval
        sample_interval = max(1, int(orig_fps / target_fps))
        estimated_frames = total_frames // sample_interval
        
        frames: List[Image.Image] = []
        angles_list: List[Optional[JointAngles]] = []
        last_frame = None
        
        # Process frames in batches
        batch_size = 16
        frame_buffer = []
        angles_buffer = []
        
        for frame_idx in tqdm(range(0, total_frames, sample_interval), desc="Sampling frames"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
                
            # Skip if too similar to last frame
            if last_frame is not None:
                similarity = self._compute_frame_similarity(frame, last_frame)
                if similarity > self.similarity_threshold:
                    continue
            
            frame_buffer.append(frame)
            last_frame = frame
            
            # Process batch when full
            if len(frame_buffer) >= batch_size:
                processed_frames, processed_angles = self._process_frame_batch(frame_buffer)
                frames.extend(processed_frames)
                angles_list.extend(processed_angles)
                frame_buffer = []
                angles_buffer = []
        
        # Process remaining frames
        if frame_buffer:
            processed_frames, processed_angles = self._process_frame_batch(frame_buffer)
            frames.extend(processed_frames)
            angles_list.extend(processed_angles)
        
        cap.release()
        logger.info(f"[VideoProcessor] Extracted and analyzed {len(frames)} frames")
        return frames, angles_list, target_fps

    def _process_frame_batch(self, frames: List[np.ndarray]) -> Tuple[List[Image.Image], List[Optional[JointAngles]]]:
        """Process a batch of frames in parallel."""
        futures = []
        for frame in frames:
            cache_key = hash(frame.tobytes())
            if cache_key in self.frame_cache:
                futures.append(self.executor.submit(lambda: self.frame_cache[cache_key]))
            else:
                futures.append(self.executor.submit(self.process_frame, frame))
        
        processed_frames = []
        processed_angles = []
        for future in futures:
            annotated_frame, angles = future.result()
            if isinstance(annotated_frame, np.ndarray):
                processed_frames.append(Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)))
            else:
                processed_frames.append(annotated_frame)
            processed_angles.append(angles)
        
        return processed_frames, processed_angles

    async def _process_frame_caption(self, session: aiohttp.ClientSession, features: Any, query: str) -> str:
        if features is None:
            return "Error: Could not analyze pose"
        
        frame = features['frame']
        pose_data = features['pose_data']
        keypoints, angles = pose_data
        
        # Create unique cache key
        cache_key = f"{hash(frame.tobytes())}_{hash(str(angles))}"
        if cache_key in self.rag_cache:
            return self.rag_cache[cache_key]
        
        try:
            # Extract CLIP features
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            
            # Batch process CLIP features
            clip_inputs = clip_processor(images=frame_pil, return_tensors="pt").to(device)
            with torch.no_grad():
                clip_features = clip_model.get_image_features(**clip_inputs)
                clip_features = F.normalize(clip_features, p=2, dim=1)  # [1, 512]
            
            # Normalize pose features and handle None values
            pose_features = []
            angle_dict = {}
            for attr in ['sword_arm_angle', 'front_knee_angle', 'back_knee_angle', 'torso_angle', 'stance_width']:
                val = getattr(angles, attr, None)
                # Handle None values for both pose features and angle dict
                if val is not None:
                    pose_features.append(safe_normalize(val))
                    angle_dict[attr] = float(val)
                else:
                    pose_features.append(0.0)
                    angle_dict[attr] = 0.0
            
            pose_features = np.array(pose_features)
            
            # Reshape tensors for proper multiplication
            pose_tensor = torch.tensor(pose_features, device=device).float().unsqueeze(0)  # [1, 5]
            
            # Project pose features to match CLIP dimension
            pose_projection = torch.nn.Linear(5, 512).to(device)
            with torch.no_grad():  # Disable gradient tracking
                projected_pose = pose_projection(pose_tensor)  # [1, 512]
                
                # Compute attention weights
                attention_weights = torch.matmul(clip_features, projected_pose.t())  # [1, 1]
                attention_weights = torch.nn.functional.softmax(attention_weights / np.sqrt(512), dim=1)
                
                # Fuse features
                fused_features = torch.cat([
                    clip_features,
                    projected_pose
                ], dim=1)  # [1, 1024]
                
                # Apply attention
                attended_features = attention_weights * fused_features  # [1, 1024]
            
            # Convert to list for API - use detach() before numpy()
            fused_features_list = attended_features.detach().cpu().numpy().tolist()
            clip_features_list = clip_features.detach().cpu().numpy().tolist()
            
            # Enhanced context with fused features
            context = {
                "pose_data": {
                    "angles": angle_dict,  # Use preprocessed angle dictionary
                    "confidence": 1.0 if all(v != 0.0 for v in angle_dict.values()) else 0.5
                },
                "visual_features": {
                    "clip": clip_features_list[0],  # Remove batch dimension
                    "pose": pose_features.tolist(),
                    "fused": fused_features_list[0]  # Remove batch dimension
                },
                "focus_topic": self.focus_topic if self.focus_topic else "fencing technique"
            }
            
            # Make RAG API call with retries
            for attempt in range(3):
                try:
                    async with session.post(
                        "http://54.227.20.42/query",
                        json={
                            "query": query,
                            "context": context,
                            "max_tokens": 75,  # Allow for two focused points
                            "temperature": 0.3,  # Keep focused
                            "response_format": {
                                "type": "text",
                                "max_length": 150  # characters
                            }
                        },
                        timeout=10
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            caption = result.get("response", "")
                            if caption:
                                # Format the response
                                def format_feedback(text):
                                    # Remove common verbose patterns
                                    text = text.replace("I notice that ", "")
                                    text = text.replace("I can see that ", "")
                                    text = text.replace("You should ", "")
                                    text = text.replace("Try to ", "")
                                    text = text.replace("Make sure to ", "")
                                    
                                    # Split into points if not already formatted
                                    if not text.startswith("1."):
                                        points = text.split(". ")
                                        if len(points) >= 2:
                                            text = f"1. {points[0]}\n2. {points[1]}"
                                    
                                    return text
                                
                                caption = format_feedback(caption)
                                self.rag_cache[cache_key] = caption
                                return caption
                        await asyncio.sleep(1)
                except Exception as e:
                    if attempt == 2:
                        logger.error(f"[RAG] Error: {str(e)}")
                        return "Error analyzing pose"
                    await asyncio.sleep(1)
            
            return "Coach feedback: Unable to generate feedback at this moment."
            
        except Exception as e:
            logger.error(f"[Fusion] Error: {str(e)}")
            return "Error analyzing pose"

    async def _process_frame_batch_captions(self, session: aiohttp.ClientSession, features_batch: List[Any], query: str) -> List[str]:
        """Process a batch of frames with rate limiting."""
        results = []
        for features in features_batch:
            try:
                caption = await self._process_frame_caption(session, features, query)
                results.append(caption)
                # Add delay to respect rate limit (5 requests per minute)
                if len(features_batch) > 1:  # Only sleep if there are more frames to process
                    await asyncio.sleep(self.rate_limit_delay)
            except Exception as e:
                logger.error(f"[RAG] Error processing frame: {str(e)}")
                results.append("Error: Caption generation failed")
        return results

    def _generate_chunk_captions(self, chunk_features: List[Any]) -> List[str]:
        """Generate captions for a chunk of frames using improved clustering."""
        try:
            # Filter out None features and prepare for clustering
            valid_features = [(i, f) for i, f in enumerate(chunk_features) if f is not None]
            if not valid_features:
                return ["Error: Could not analyze pose"] * len(chunk_features)
            
            valid_indices, features = zip(*valid_features)
            
            # Extract CLIP embeddings and pose features for clustering
            embeddings_list = []
            pose_features_list = []
            
            for f in features:
                frame = f['frame']
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                
                # Get CLIP embeddings
                clip_inputs = clip_processor(images=frame_pil, return_tensors="pt").to(device)
                with torch.no_grad():
                    emb = clip_model.get_image_features(**clip_inputs)
                    emb = F.normalize(emb, p=2, dim=1).cpu().numpy().flatten()
                embeddings_list.append(emb)
                
                # Get pose features
                _, angles = f['pose_data']
                pose_features = []
                for attr in ['sword_arm_angle', 'front_knee_angle', 'back_knee_angle', 'torso_angle']:
                    val = getattr(angles, attr, None)
                    pose_features.append(safe_normalize(val) if val is not None else 0.0)
                pose_features_list.append(np.array(pose_features))
            
            embeddings = np.stack(embeddings_list)
            pose_features = np.stack(pose_features_list)
            
            # Combine visual and pose features with weights
            combined_features = np.concatenate([
                embeddings * 0.7,  # Visual features weight
                pose_features * 0.3  # Pose features weight
            ], axis=1)
            
            # Improved clustering with higher threshold and pose-aware similarity
            clusters = []
            cluster_map = {}
            threshold = 0.85  # Increased threshold for more distinct clusters
            
            for j, feat in enumerate(combined_features):
                assigned = False
                for cid, rep_idx in enumerate(clusters):
                    # Compare combined similarity
                    sim = np.dot(feat, combined_features[rep_idx])
                    
                    # Get pose angles for detailed comparison
                    _, rep_angles = features[rep_idx]['pose_data']
                    _, cur_angles = features[j]['pose_data']
                    
                    # Check if poses are significantly different
                    pose_diff = False
                    for attr in ['sword_arm_angle', 'front_knee_angle', 'back_knee_angle', 'torso_angle']:
                        rep_val = getattr(rep_angles, attr, None)
                        cur_val = getattr(cur_angles, attr, None)
                        if rep_val is not None and cur_val is not None:
                            if abs(rep_val - cur_val) > 20:  # Increased angle threshold
                                pose_diff = True
                                break
                    
                    if sim >= threshold and not pose_diff:
                        cluster_map[j] = cid
                        assigned = True
                        break
                
                if not assigned:
                    cid = len(clusters)
                    clusters.append(j)
                    cluster_map[j] = cid
            
            # Get representative frames ensuring unique poses
            cluster_reps = []
            used_poses = set()
            
            for cid in range(len(clusters)):
                members = [j for j, c in cluster_map.items() if c == cid]
                
                # Find the best representative frame
                best_rep = None
                best_score = -1
                
                for member in members:
                    _, angles = features[member]['pose_data']
                    pose_key = tuple(getattr(angles, attr, 0) for attr in 
                                   ['sword_arm_angle', 'front_knee_angle', 'back_knee_angle', 'torso_angle'])
                    
                    if pose_key not in used_poses:
                        # Score based on pose confidence and frame quality
                        score = sum(1 for x in pose_key if x is not None)
                        if score > best_score:
                            best_score = score
                            best_rep = member
                
                if best_rep is not None:
                    cluster_reps.append(features[best_rep])
                    _, angles = features[best_rep]['pose_data']
                    pose_key = tuple(getattr(angles, attr, 0) for attr in 
                                   ['sword_arm_angle', 'front_knee_angle', 'back_knee_angle', 'torso_angle'])
                    used_poses.add(pose_key)
                else:
                    # Fallback to first member if no unique pose found
                    cluster_reps.append(features[members[0]])
            
            # Process representatives with rate limiting
            async def process_representatives() -> List[str]:
                async with aiohttp.ClientSession() as session:
                    batch_size = 4
                    all_captions = []
                    
                    with tqdm(total=len(cluster_reps), desc="Generating captions", unit="frame") as pbar:
                        for i in range(0, len(cluster_reps), batch_size):
                            batch = cluster_reps[i:i + batch_size]
                            captions = await self._process_frame_batch_captions(session, batch, GLOBAL_QUERY)
                            all_captions.extend(captions)
                            pbar.update(len(batch))
                    
                    return all_captions
            
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                rep_captions = loop.run_until_complete(process_representatives())
                loop.close()
            except Exception as e:
                logger.error(f"[Async] Error: {str(e)}")
                return ["Error: Caption generation failed"] * len(chunk_features)
            
            # Map captions back to original frames with improved uniqueness
            results = ["Error: Could not analyze pose"] * len(chunk_features)
            for i, idx in enumerate(valid_indices):
                cid = cluster_map[i]
                if cid < len(rep_captions):
                    results[idx] = rep_captions[cid]
            
            return results
            
        except Exception as e:
            logger.error(f"[Clustering] Error: {str(e)}")
            return ["Error: Caption generation failed"] * len(chunk_features)

    def process_video(self, video_path: str):
        """Process video file and extract frames with captions."""
        try:
            logger.info(f"Processing video: {video_path}")
            
            # Initialize video capture with explicit backend
            cap = cv2.VideoCapture(video_path, cv2.CAP_ANY)
            if not cap.isOpened():
                raise ValueError("Failed to open video file")
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            target_fps = 5  # Target 5 FPS for analysis
            frame_interval = max(1, int(fps / target_fps))
            
            logger.info(f"[VideoProcessor] Processing video with target {target_fps} FPS...")
            
            # Initialize lists for frames and captions
            self.frames = []
            self.captions = []
            self.frame_poses = []
            self.frame_angles = []
            
            frame_idx = 0
            processed_count = 0
            
            # Process frames
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_idx % frame_interval == 0:
                    try:
                        # Convert BGR to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Get pose estimation
                        keypoints, angles = self.pose_estimator.estimate_pose(frame_rgb)
                        
                        # Draw pose on frame if keypoints exist
                        if keypoints is not None and angles is not None:
                            # Create a copy of the frame for drawing
                            annotated_frame = frame_rgb.copy()
                            annotated_frame = self.pose_estimator.draw_pose(annotated_frame, keypoints, angles)
                        else:
                            annotated_frame = frame_rgb
                        
                        # Store processed frame and data
                        self.frames.append(annotated_frame)
                        self.frame_poses.append(keypoints)
                        self.frame_angles.append(angles)
                        processed_count += 1
                        
                    except Exception as e:
                        logger.error(f"Error processing frame {frame_idx}: {str(e)}")
                        # Skip problematic frame but continue processing
                        continue
                        
                frame_idx += 1
                
            cap.release()
            
            logger.info(f"[VideoProcessor] Extracted and analyzed {processed_count} frames")
            
            # Generate captions in chunks
            chunk_size = 5
            for i in range(0, len(self.frames), chunk_size):
                chunk_frames = self.frames[i:i+chunk_size]
                chunk_features = []
                for j, frame in enumerate(chunk_frames):
                    chunk_features.append({
                        'frame': frame,
                        'pose_data': (self.frame_poses[i+j], self.frame_angles[i+j])
                    })
                chunk_captions = self._generate_chunk_captions(chunk_features)
                self.captions.extend(chunk_captions)
                
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            raise

# ------------------------------------------------------------------------------
# VIDEO CHAT MODULE (Interactive Coaching)
# ------------------------------------------------------------------------------

class VideoChat:
    """
    High-level class for performing video analysis, caption generation,
    and interactive chat with the Fencing Coach AI.
    """
    def __init__(self):
        """Initialize the VideoChat system with necessary components."""
        self.video_processor = VideoProcessor()
        self.frames = []
        self.captions = []
        self.frame_poses = []
        self.frame_angles = []
        self.motion_analyzer = MotionAnalyzer()
        self.sequence_memory = []  # Store sequence analysis history
        
    def process_video(self, video_path: str) -> None:
        """Process a video file and prepare it for chat interaction."""
        try:
            logger.info(f"Processing video: {video_path}")
            
            # Process video using video processor
            self.video_processor.process_video(video_path)
            
            # Get the processed data
            self.frames = self.video_processor.frames
            self.captions = self.video_processor.captions
            self.frame_poses = self.video_processor.frame_poses
            self.frame_angles = self.video_processor.frame_angles
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            raise

    def chat(self, query: str) -> str:
        """Process a chat query about the video."""
        try:
            if not self.frames or not self.captions:
                return "Please process a video first before chatting."
            
            # Create context from frames, captions, poses, and angles
            context = self._create_chat_context()
            
            # Generate response using the context and query
            response = self._generate_response(query, context)
            
            return response
            
        except Exception as e:
            logger.error(f"[Chat] Error processing chat query: {str(e)}")
            return "I apologize, but I encountered an error processing your query. Please try again."

    def _create_chat_context(self) -> str:
        """Create a rich context string with temporal and technical analysis."""
        if not self.frame_angles:
            return "No pose data available"
            
        # Get motion analysis for recent frames
        analysis = self.motion_analyzer.analyze_sequence(self.frame_angles[-10:])
        self.sequence_memory.append(analysis)
        
        # Get current frame angles
        current_angles = self.frame_angles[-1]
        
        # Create detailed context
        context = f"""Movement Analysis:
{analysis['temporal_context']}
Current Phase: {analysis['sequence']} - {analysis['phase']} (Confidence: {analysis['confidence']:.2f})

Technical Measurements:
- Sword Arm: {current_angles.sword_arm_angle:.0f}° ({'Extended' if current_angles.sword_arm_angle > 160 else 'Bent'})
- Front Knee: {current_angles.front_knee_angle:.0f}° ({'Deep' if current_angles.front_knee_angle < 100 else 'Shallow'})
- Back Knee: {current_angles.back_knee_angle:.0f}° ({'Straight' if current_angles.back_knee_angle > 160 else 'Bent'})

Movement Quality:
- Form: {analysis['technical_quality']['form']:.2f}
- Balance: {analysis['technical_quality']['balance']:.2f}
- Efficiency: {analysis['technical_quality']['efficiency']:.2f}

Progression Metrics:
- Smoothness: {analysis['progression']['smoothness']:.2f}
- Speed: {analysis['progression']['speed']:.2f}
- Control: {analysis['progression']['control']:.2f}

Previous Feedback History:
{self._get_feedback_history()}"""
        
        return context
        
    def _get_feedback_history(self) -> str:
        """Get relevant feedback history for context."""
        if not self.sequence_memory:
            return "No previous feedback"
            
        # Get last 3 unique phases
        unique_phases = []
        seen = set()
        for analysis in reversed(self.sequence_memory[-10:]):
            phase = f"{analysis['sequence']} - {analysis['phase']}"
            if phase not in seen and len(unique_phases) < 3:
                seen.add(phase)
                unique_phases.append(f"- {phase}: Form={analysis['technical_quality']['form']:.2f}, "
                                   f"Efficiency={analysis['technical_quality']['efficiency']:.2f}")
        
        return "Recent Movement History:\n" + "\n".join(unique_phases)

    def _generate_response(self, query: str, context: str) -> str:
        """Generate a response to the query using the context."""
        try:
            # Create focused prompt for sequence-aware feedback
            prompt = f"""You are an expert fencing coach analyzing a video sequence.
Based on the detailed movement analysis provided, give comprehensive feedback in three parts:

1. Movement Sequence Analysis:
   - Comment on the detected sequence and phase
   - Evaluate the progression quality
   - Note any issues in timing or transitions

2. Technical Execution:
   - Analyze the current position's form
   - Comment on balance and efficiency scores
   - Suggest specific angle corrections

3. Improvement Strategy:
   - Provide actionable corrections
   - Reference the movement history
   - Suggest drills or focus areas

{context}

Focus on the most critical aspects of the {query} and provide specific, actionable feedback."""
            
            try:
                response = requests.post(
                    "http://54.227.20.42/query",
                    json={
                        "query": prompt,
                        "context": context,
                        "max_tokens": 300,  # Allow for comprehensive response
                        "temperature": 0.3,  # Keep focused
                        "response_format": {
                            "type": "text",
                            "max_length": 800  # characters
                        }
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    feedback = result.get("response", "")
                    return feedback
                else:
                    logger.error(f"RAG API error: {response.status_code}")
                    return "Error analyzing video"
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"RAG API request error: {str(e)}")
                return "Network error"
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "Error analyzing video"

# ------------------------------------------------------------------------------
# CAPTION INDEXING & RETRIEVAL (for additional search functionality)
# ------------------------------------------------------------------------------

def build_caption_index(captions: List[str]) -> Tuple[faiss.IndexFlatIP, np.ndarray]:
    """
    Embed captions with a SentenceTransformer and build a FAISS index.
    """
    logger.info("[Index] Embedding captions and building FAISS index...")
    caption_embeddings = sent_transformer.encode(captions, convert_to_numpy=True, normalize_embeddings=True)
    d = caption_embeddings.shape[1]
    index = faiss.IndexFlatIP(d)  # inner product on normalized vectors
    index.add(caption_embeddings)
    return index, caption_embeddings

def query_captions(query: str, index: faiss.IndexFlatIP, captions: List[str], top_k: int = 3) -> List[Tuple[str, float]]:
    """
    Embed the query and retrieve the top_k matching captions.
    Returns a list of (caption, similarity score) tuples.
    """
    query_embedding = sent_transformer.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, idxs = index.search(query_embedding, top_k)
    results = []
    for j, i in enumerate(idxs[0]):
        results.append((captions[i], scores[0][j]))
    return results

# ------------------------------------------------------------------------------
# FRAME VIEWER (Interactive Visualization)
# ------------------------------------------------------------------------------

class FrameViewer:
    """
    A simple GUI using OpenCV to browse frames and view generated captions.
    Use keyboard/mouse controls to navigate the analysis.
    """
    def __init__(self, frames: List[Union[np.ndarray, Image.Image]], captions: List[str]) -> None:
        # Convert PIL Images to numpy arrays if needed
        self.frames = []
        for frame in frames:
            if isinstance(frame, Image.Image):
                # Convert PIL Image to numpy array and BGR format for OpenCV
                frame_np = np.array(frame)
                if len(frame_np.shape) == 3 and frame_np.shape[2] == 3:
                    frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                self.frames.append(frame_np)
            else:
                self.frames.append(frame)
        
        self.captions = captions
        self.current_frame_idx = 0
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.is_playing = False
        
        cv2.namedWindow('Fencing Analysis')
        cv2.createTrackbar('Frame', 'Fencing Analysis', 0, len(self.frames) - 1, self._on_trackbar)
        cv2.setMouseCallback('Fencing Analysis', self._on_mouse)
        
        self.help_text = [
            'Controls:',
            'A/Left Arrow - Previous frame',
            'D/Right Arrow - Next frame',
            'Space - Play/Pause',
            'H - Show/Hide help',
            'Q - Quit'
        ]
        
    def show_help(self) -> None:
        """Display a help window with control instructions."""
        help_img = np.zeros((len(self.help_text) * 30 + 20, 400, 3), dtype=np.uint8)
        for i, text in enumerate(self.help_text):
            cv2.putText(help_img, text, (10, 30 + i * 30), self.font, 0.75, (255, 255, 255), 1)
        cv2.imshow('Help', help_img)
    
    def _on_trackbar(self, value: int) -> None:
        self.current_frame_idx = value
        self._update_display()
    
    def _on_mouse(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDBLCLK:
            self.current_frame_idx = cv2.getTrackbarPos('Frame', 'Fencing Analysis')
            self._update_display()
    
    def _update_display(self) -> None:
        try:
            frame_idx = min(max(0, self.current_frame_idx), len(self.frames) - 1)
            frame = self.frames[frame_idx].copy()
            caption = self.captions[frame_idx] if frame_idx < len(self.captions) else "Processing caption..."
            
            caption_height = 200
            caption_img = np.zeros((caption_height, frame.shape[1], 3), dtype=np.uint8)
            
            sections = caption.split("\n\n") if caption else ["No data available"]
            y_offset = 30
            if sections:
                for line in sections[0].split("\n"):
                    cv2.putText(caption_img, line, (10, y_offset), self.font, 0.75, (255, 255, 255), 1)
                    y_offset += 25
            
            if len(sections) > 1:
                y_offset += 10
                feedback_text = sections[1]
                max_width = frame.shape[1] - 20
                words = feedback_text.split()
                current_line = []
                for word in words:
                    current_line.append(word)
                    text = ' '.join(current_line)
                    (text_width, _), _ = cv2.getTextSize(text, self.font, 0.75, 1)
                    if text_width > max_width:
                        if len(current_line) > 1:
                            current_line.pop()
                            cv2.putText(caption_img, ' '.join(current_line), (10, y_offset), self.font, 0.75, (255, 255, 255), 1)
                            y_offset += 25
                            current_line = [word]
                        else:
                            cv2.putText(caption_img, word, (10, y_offset), self.font, 0.75, (255, 255, 255), 1)
                            y_offset += 25
                            current_line = []
                if current_line:
                    cv2.putText(caption_img, ' '.join(current_line), (10, y_offset), self.font, 0.75, (255, 255, 255), 1)
            
            display_height = frame.shape[0] + caption_height
            display_img = np.zeros((display_height, frame.shape[1], 3), dtype=np.uint8)
            display_img[:frame.shape[0]] = frame
            display_img[frame.shape[0]:] = caption_img
            
            cv2.imshow('Fencing Analysis', display_img)
            cv2.setTrackbarPos('Frame', 'Fencing Analysis', frame_idx)
        except Exception as e:
            logger.error(f"[Viewer] Error updating display: {str(e)}")
    
    def show(self) -> None:
        try:
            self._update_display()
            self.show_help()
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    cv2.destroyAllWindows()
                    break
                elif key == ord('h') or key == ord('H'):
                    self.show_help()
                elif key in [ord('d'), ord('D'), 83]:
                    self.current_frame_idx = (self.current_frame_idx + 1) % len(self.frames)
                    self._update_display()
                elif key in [ord('a'), ord('A'), 81]:
                    self.current_frame_idx = (self.current_frame_idx - 1) % len(self.frames)
                    self._update_display()
                elif key == ord(' '):
                    self.is_playing = not self.is_playing
                    while self.is_playing:
                        self.current_frame_idx = (self.current_frame_idx + 1) % len(self.frames)
                        self._update_display()
                        if cv2.waitKey(50) != -1:
                            self.is_playing = False
                            break
        except Exception as e:
            logger.error(f"[Viewer] Error in frame viewer: {str(e)}")
            cv2.destroyAllWindows()

# ------------------------------------------------------------------------------
# MAIN FUNCTION
# ------------------------------------------------------------------------------

def main() -> None:
    try:
        video_path = "En-Garde+Lunge - Made with Clipchamp.mp4"
        logger.info("\n[Main] Initializing Fencing Coach AI...")
        
        if not os.path.exists(video_path):
            logger.error(f"[Main] Video file not found: {video_path}")
            return
        
        chat = VideoChat()
        chat.process_video(video_path)
        logger.info("[Main] VideoChat instance created successfully")
        
        chat.frame_viewer.show()
        logger.info("\n[Main] Fencing Coach AI ready!")
        
    except Exception as e:
        logger.error(f"[Main] Error in main: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()

class MotionAnalyzer:
    """Advanced motion analysis system using dynamic pattern recognition and biomechanical understanding."""
    def __init__(self, memory_window: int = 10):
        self.memory_window = memory_window
        self.motion_history = []
        self.phase_transitions = []
        
        # Dynamic movement patterns learned from motion sequences
        self.movement_dynamics = {
            'lunge': {
                'phases': ['preparation', 'extension', 'recovery'],
                'key_relationships': {
                    'preparation': {
                        'spatial': ['sword_arm_to_hip_alignment', 'weight_distribution'],
                        'temporal': ['stability_period', 'tension_buildup']
                    },
                    'extension': {
                        'spatial': ['forward_momentum', 'power_chain'],
                        'temporal': ['acceleration', 'coordination_timing']
                    },
                    'recovery': {
                        'spatial': ['balance_restoration', 'stance_reset'],
                        'temporal': ['deceleration', 'recentering']
                    }
                }
            }
        }
        
        # Biomechanical chains for power and movement analysis
        self.kinetic_chains = {
            'power_generation': ['back_leg', 'hip_rotation', 'torso', 'sword_arm'],
            'balance_control': ['stance_width', 'hip_alignment', 'torso_angle'],
            'movement_efficiency': ['joint_coordination', 'momentum_transfer', 'energy_conservation']
        }
        
    def analyze_sequence(self, angles_history: List[JointAngles]) -> Dict[str, Any]:
        """Analyze motion sequence using advanced biomechanical understanding."""
        if not angles_history:
            return {"sequence": "Unknown", "phase": "Unknown", "confidence": 0.0}
            
        # Update motion history
        self.motion_history = angles_history[-self.memory_window:]
        
        # Extract rich motion features
        motion_features = self._extract_motion_features()
        
        # Analyze kinetic chain activation
        kinetic_analysis = self._analyze_kinetic_chains(motion_features)
        
        # Detect movement phase using dynamic pattern recognition
        phase_analysis = self._analyze_movement_phase(motion_features, kinetic_analysis)
        
        # Analyze movement quality through biomechanical lens
        quality_analysis = self._analyze_movement_quality(motion_features, kinetic_analysis)
        
        return {
            "sequence": phase_analysis['movement'],
            "phase": phase_analysis['phase'],
            "confidence": phase_analysis['confidence'],
            "kinetic_chain": kinetic_analysis,
            "biomechanics": quality_analysis,
            "temporal_context": self._get_temporal_context()
        }
    
    def _extract_motion_features(self) -> Dict[str, Any]:
        """Extract rich motion features including velocities, accelerations, and coordination patterns."""
        features = {
            'kinematics': self._analyze_kinematics(),
            'coordination': self._analyze_coordination(),
            'energy': self._analyze_energy_flow()
        }
        return features
        
    def _analyze_kinematics(self) -> Dict[str, np.ndarray]:
        """Analyze movement kinematics including velocities and accelerations."""
        kinematics = {}
        
        # Calculate velocities and accelerations for key joints
        for joint in ['sword_arm', 'front_knee', 'back_knee', 'torso']:
            angles = []
            for frame in self.motion_history:
                angle = getattr(frame, f"{joint}_angle", None)
                angles.append(angle if angle is not None else 0.0)
            
            angles = np.array(angles)
            velocities = np.gradient(angles)
            accelerations = np.gradient(velocities)
            
            kinematics[joint] = {
                'position': angles,
                'velocity': velocities,
                'acceleration': accelerations
            }
        
        return kinematics
        
    def _analyze_coordination(self) -> Dict[str, float]:
        """Analyze inter-joint coordination and movement patterns."""
        if len(self.motion_history) < 2:
            return {'coordination_score': 0.0}
            
        # Analyze coordination between joints
        coordination_patterns = []
        key_joints = ['sword_arm', 'front_knee', 'back_knee']
        
        for i in range(len(key_joints)):
            for j in range(i+1, len(key_joints)):
                joint1, joint2 = key_joints[i], key_joints[j]
                correlation = self._calculate_joint_coordination(joint1, joint2)
                coordination_patterns.append(correlation)
        
        return {
            'coordination_score': np.mean(coordination_patterns) if coordination_patterns else 0.0,
            'patterns': coordination_patterns
        }
        
    def _analyze_energy_flow(self) -> Dict[str, float]:
        """Analyze energy flow through the kinetic chain."""
        if len(self.motion_history) < 2:
            return {'energy_efficiency': 0.0}
            
        # Calculate energy transfer through joints
        energy_transfer = []
        for i in range(1, len(self.motion_history)):
            prev = self.motion_history[i-1]
            curr = self.motion_history[i]
            
            # Calculate kinetic energy changes
            energy_change = 0
            for joint in ['back_knee', 'front_knee', 'torso', 'sword_arm']:
                prev_angle = getattr(prev, f"{joint}_angle", None)
                curr_angle = getattr(curr, f"{joint}_angle", None)
                if prev_angle and curr_angle:
                    energy_change += abs(curr_angle - prev_angle)
            
            energy_transfer.append(energy_change)
        
        efficiency = 1.0 - min(1.0, np.std(energy_transfer) / 30.0)
        return {
            'energy_efficiency': efficiency,
            'transfer_smoothness': 1.0 - min(1.0, np.mean(energy_transfer) / 45.0)
        }
        
    def _analyze_kinetic_chains(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Analyze activation and efficiency of kinetic chains."""
        kinematics = features['kinematics']
        
        # Analyze power generation chain
        power_scores = []
        for joint in self.kinetic_chains['power_generation']:
            if joint in kinematics:
                # Look for proper sequence of activation
                acceleration = kinematics[joint]['acceleration']
                power_scores.append(np.max(np.abs(acceleration)))
        
        # Analyze balance control
        balance_scores = []
        for joint in self.kinetic_chains['balance_control']:
            if joint in kinematics:
                # Check stability of joint positions
                positions = kinematics[joint]['position']
                balance_scores.append(1.0 - min(1.0, np.std(positions) / 20.0))
        
        return {
            'power_generation': np.mean(power_scores) if power_scores else 0.0,
            'balance_control': np.mean(balance_scores) if balance_scores else 0.0,
            'chain_activation': self._calculate_chain_activation(kinematics)
        }
        
    def _calculate_chain_activation(self, kinematics: Dict[str, Any]) -> float:
        """Calculate the quality of kinetic chain activation sequence."""
        if not kinematics:
            return 0.0
            
        # Analyze timing of peak accelerations
        peak_timings = {}
        for joint in self.kinetic_chains['power_generation']:
            if joint in kinematics:
                acceleration = kinematics[joint]['acceleration']
                peak_idx = np.argmax(np.abs(acceleration))
                peak_timings[joint] = peak_idx
        
        # Check if activation sequence is correct (proximal to distal)
        correct_sequence = list(peak_timings.values())
        sorted_sequence = sorted(peak_timings.values())
        
        # Calculate sequence accuracy
        sequence_accuracy = 1.0 - np.mean([
            abs(a - b) / len(kinematics[joint]['acceleration'])
            for a, b in zip(correct_sequence, sorted_sequence)
        ])
        
        return sequence_accuracy
        
    def _calculate_joint_coordination(self, joint1: str, joint2: str) -> float:
        """Calculate coordination between two joints using phase analysis."""
        angles1 = []
        angles2 = []
        
        for frame in self.motion_history:
            angle1 = getattr(frame, f"{joint1}_angle", None)
            angle2 = getattr(frame, f"{joint2}_angle", None)
            if angle1 is not None and angle2 is not None:
                angles1.append(angle1)
                angles2.append(angle2)
        
        if not angles1 or not angles2:
            return 0.0
            
        # Calculate phase correlation
        correlation = np.corrcoef(angles1, angles2)[0, 1]
        return max(0.0, correlation)  # Return positive correlation only
        
    def _analyze_movement_phase(self, features: Dict[str, Any], kinetic_analysis: Dict[str, float]) -> Dict[str, Any]:
        """Analyze movement phase using dynamic pattern recognition."""
        # Identify movement type and phase based on feature patterns
        phase_probabilities = self._calculate_phase_probabilities(features, kinetic_analysis)
        
        # Get highest probability phase
        best_movement = max(phase_probabilities.items(), key=lambda x: max(x[1].values()))
        movement_type = best_movement[0]
        phase_probs = best_movement[1]
        current_phase = max(phase_probs.items(), key=lambda x: x[1])
        
        return {
            'movement': movement_type,
            'phase': current_phase[0],
            'confidence': current_phase[1]
        }
        
    def _calculate_phase_probabilities(self, features: Dict[str, Any], kinetic_analysis: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Calculate probabilities for each movement phase based on features."""
        probabilities = {}
        
        for movement, dynamics in self.movement_dynamics.items():
            phase_probs = {}
            for phase, relationships in dynamics['key_relationships'].items():
                # Calculate probability based on spatial and temporal relationships
                spatial_score = self._evaluate_spatial_relationships(
                    relationships['spatial'],
                    features,
                    kinetic_analysis
                )
                temporal_score = self._evaluate_temporal_relationships(
                    relationships['temporal'],
                    features
                )
                phase_probs[phase] = (spatial_score + temporal_score) / 2
            
            probabilities[movement] = phase_probs
        
        return probabilities
        
    def _evaluate_spatial_relationships(self, relationships: List[str], features: Dict[str, Any], 
                                     kinetic_analysis: Dict[str, float]) -> float:
        """Evaluate spatial relationships in movement."""
        scores = []
        for relationship in relationships:
            if relationship == 'power_chain':
                scores.append(kinetic_analysis['chain_activation'])
            elif relationship == 'balance_restoration':
                scores.append(kinetic_analysis['balance_control'])
            # Add more relationship evaluations as needed
        
        return np.mean(scores) if scores else 0.0
        
    def _evaluate_temporal_relationships(self, relationships: List[str], features: Dict[str, Any]) -> float:
        """Evaluate temporal relationships in movement."""
        scores = []
        for relationship in relationships:
            if relationship == 'coordination_timing':
                scores.append(features['coordination']['coordination_score'])
            elif relationship == 'energy_conservation':
                scores.append(features['energy']['energy_efficiency'])
            # Add more relationship evaluations as needed
        
        return np.mean(scores) if scores else 0.0
        
    def _analyze_movement_quality(self, features: Dict[str, Any], kinetic_analysis: Dict[str, float]) -> Dict[str, float]:
        """Analyze overall movement quality through biomechanical lens."""
        return {
            'efficiency': features['energy']['energy_efficiency'],
            'coordination': features['coordination']['coordination_score'],
            'power_transfer': kinetic_analysis['power_generation'],
            'balance': kinetic_analysis['balance_control'],
            'technical_mastery': np.mean([
                features['energy']['energy_efficiency'],
                features['coordination']['coordination_score'],
                kinetic_analysis['power_generation'],
                kinetic_analysis['balance_control']
            ])
        }
        
    def _get_temporal_context(self) -> str:
        """Generate temporal context description."""
        if len(self.phase_transitions) < 2:
            return "Initial position"
            
        transitions = " -> ".join(self.phase_transitions[-3:])
        return f"Movement sequence: {transitions}"
