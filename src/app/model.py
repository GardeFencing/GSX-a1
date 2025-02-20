import os
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_NAME = "openbmb/MiniCPM-V-2_6-gguf"  # Using GGUF version for better CPU performance
DEVICE = "cpu"  # Force CPU since we're using GGUF format
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")

def load_model_and_tokenizer() -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load MiniCPM-V model and tokenizer.
    
    Returns:
        Tuple[AutoModelForCausalLM, AutoTokenizer]: Loaded model and tokenizer
    """
    try:
        logger.info(f"Loading model {MODEL_NAME} on {DEVICE}...")
        
        if not ACCESS_TOKEN:
            raise ValueError("ACCESS_TOKEN not found in environment variables")
        
        # Load model with CPU optimizations
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,  # Use float32 for CPU
            device_map=None,  # No device mapping needed for CPU
            token=ACCESS_TOKEN,
            low_cpu_mem_usage=True  # Enable CPU memory optimizations
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=ACCESS_TOKEN)
        
        logger.info("Model and tokenizer loaded successfully")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def generate_response(model: AutoModelForCausalLM, 
                     tokenizer: AutoTokenizer,
                     frames: list,
                     question: str) -> Optional[str]:
    """
    Generate response for a video question using MiniCPM-V.
    
    Args:
        model (AutoModelForCausalLM): The loaded model
        tokenizer (AutoTokenizer): The loaded tokenizer
        frames (list): List of video frames
        question (str): Question about the video
        
    Returns:
        Optional[str]: Generated response or None if error occurs
    """
    try:
        # Prepare input
        msgs = [{"role": "user", "content": frames + [question]}]
        
        # Generate answer with CPU-friendly settings
        params = {
            "use_image_id": False,
            "max_slice_nums": 1,
            "max_new_tokens": 256,  # Reduced for CPU
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 50,  # Reduced for CPU
            "num_beams": 1  # Use greedy decoding on CPU
        }
        
        answer = model.chat(
            image=None,
            msgs=msgs,
            tokenizer=tokenizer,
            **params
        )
        
        return answer if answer else None
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return None 