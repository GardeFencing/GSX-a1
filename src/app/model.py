import os
import logging
import torch
from transformers import AutoModel, AutoTokenizer
from typing import Tuple, Optional, List
from PIL import Image
from dotenv import load_dotenv
from decord import VideoReader, cpu

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_NAME = "openbmb/MiniCPM-V-2_6"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NUM_FRAMES = 64  # Reduce if CUDA OOM
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")

def load_model_and_tokenizer() -> Tuple[AutoModel, AutoTokenizer]:
    """
    Load MiniCPM-V model and tokenizer with multi-image support.
    
    Returns:
        Tuple[AutoModel, AutoTokenizer]: Loaded model and tokenizer
    """
    try:
        logger.info(f"Loading model {MODEL_NAME} on {DEVICE}...")
        
        # Load model with correct settings
        model = AutoModel.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            attn_implementation='sdpa',
            torch_dtype=torch.bfloat16
        )
        model = model.eval().to(DEVICE)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True
        )
        
        logger.info("Model and tokenizer loaded successfully")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def encode_video(video_path: str) -> List[Image.Image]:
    """
    Extract frames from video file.
    """
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        sample_fps = round(vr.get_avg_fps() / 1)  # 1 FPS
        frame_idx = [i for i in range(0, len(vr), sample_fps)]
        
        if len(frame_idx) > MAX_NUM_FRAMES:
            frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
            
        frames = vr.get_batch(frame_idx).asnumpy()
        frames = [Image.fromarray(v.astype('uint8')) for v in frames]
        logger.info(f'Extracted {len(frames)} frames from video')
        return frames
        
    except Exception as e:
        logger.error(f"Error encoding video: {str(e)}")
        raise

def generate_response(model: AutoModel, 
                     tokenizer: AutoTokenizer,
                     frames: List[Image.Image],
                     question: str,
                     is_video: bool = False) -> Optional[str]:
    """
    Generate response for video/image questions using MiniCPM-V.
    """
    try:
        # For video, we need special parameters
        params = {}
        if is_video:
            params["use_image_id"] = False
            params["max_slice_nums"] = 2  # use 1 if CUDA OOM and video resolution > 448*448
            
            # For video, append question after frames
            msgs = [{'role': 'user', 'content': frames + [question]}]
        else:
            # For single/multi image, include images and question in content
            msgs = [{'role': 'user', 'content': [*frames, question]}]
        
        # Generate response
        response = model.chat(
            image=None,
            msgs=msgs,
            tokenizer=tokenizer,
            **params
        )
        
        return response if response else None
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return None 