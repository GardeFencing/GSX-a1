import os
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, Optional, List
from PIL import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_NAME = "openbmb/MiniCPM-V-2_6"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")

def load_model_and_tokenizer() -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load MiniCPM-V model and tokenizer with multi-image support.
    
    Returns:
        Tuple[AutoModelForCausalLM, AutoTokenizer]: Loaded model and tokenizer
    """
    try:
        logger.info(f"Loading model {MODEL_NAME} on {DEVICE}...")
        
        if not ACCESS_TOKEN:
            raise ValueError("ACCESS_TOKEN not found in environment variables")
        
        # Load model with optimizations
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            device_map="auto" if DEVICE == "cuda" else None,
            token=ACCESS_TOKEN,
            trust_remote_code=True
        )
        
        # Load tokenizer with trust_remote_code for multi-image support
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, 
            token=ACCESS_TOKEN,
            trust_remote_code=True
        )
        
        logger.info("Model and tokenizer loaded successfully")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def generate_response(model: AutoModelForCausalLM, 
                     tokenizer: AutoTokenizer,
                     frames: List[Image.Image],
                     question: str,
                     few_shot_examples: Optional[List[dict]] = None) -> Optional[str]:
    """
    Generate response for a video question using MiniCPM-V with multi-image and few-shot support.
    
    Args:
        model (AutoModelForCausalLM): The loaded model
        tokenizer (AutoTokenizer): The loaded tokenizer
        frames (List[Image.Image]): List of video frames
        question (str): Question about the video
        few_shot_examples (Optional[List[dict]]): List of few-shot examples
        
    Returns:
        Optional[str]: Generated response or None if error occurs
    """
    try:
        # Prepare image tags for multi-image input
        image_tags = "".join(["(<image>./</image>)" for _ in frames])
        
        # Prepare few-shot examples if provided
        few_shot_prompt = ""
        if few_shot_examples:
            for example in few_shot_examples:
                few_shot_prompt += f"User: {example['question']}\nAssistant: {example['answer']}\n\n"
        
        # Construct messages with multi-image support
        messages = [{
            "role": "user",
            "content": f"{image_tags}\n{few_shot_prompt}Question: {question}"
        }]
        
        # Apply chat template
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Prepare inputs with multi-modal data
        inputs = {
            "prompt": prompt,
            "multi_modal_data": {
                "image": frames if len(frames) > 1 else frames[0]
            }
        }
        
        # Define stop tokens
        stop_tokens = ['<|im_end|>', '<|endoftext|>']
        stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
        
        # Generate response with optimized parameters
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=stop_token_ids,
            use_cache=True,
            num_beams=3 if DEVICE == "cuda" else 1
        )
        
        # Decode and clean response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(prompt, "").strip()
        
        return response if response else None
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return None 