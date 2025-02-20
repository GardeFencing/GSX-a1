import os
import logging
import torch
from typing import Tuple, Optional, List
from PIL import Image
import subprocess
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MiniCPMVProcessor:
    """Handle MiniCPM-V model integration for video captioning and chat"""
    def __init__(self, model_path: str = "MiniCPM-V-2_6-gguf/ggml-model-Q4_K_M.gguf",
                 mmproj_path: str = "MiniCPM-V-2_6-gguf/mmproj-model-f16.gguf"):
        self.model_path = model_path
        self.mmproj_path = mmproj_path
        self.cli_path = "/home/ubuntu/GSX-a1/llama.cpp/build/bin/llama-minicpmv-cli"
        
        # Optimized parameters
        self.temp = 0.1
        self.top_p = 0.7
        self.top_k = 40
        self.repeat_penalty = 1.1
        self.context_size = 2048
        self.n_threads = max(1, os.cpu_count() // 2)
        
        # Create temp directory
        self.temp_dir = os.path.join(os.getcwd(), 'temp_frames')
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir, exist_ok=True)
        
        # Ensure model files exist
        if not os.path.exists(model_path):
            raise ValueError(f"Model file not found: {model_path}")
        if not os.path.exists(mmproj_path):
            raise ValueError(f"MMPROJ file not found: {mmproj_path}")
        if not os.path.exists(self.cli_path):
            raise ValueError(f"CLI executable not found: {self.cli_path}")
    
    def process_frames(self, frames: List[Image.Image], system_prompt: str = None, query: str = "Describe these frames in detail.") -> str:
        """Process multiple frames with MiniCPM-V"""
        temp_paths = []
        try:
            logger.info(f"\nMiniCPMV processing {len(frames)} frames")
            logger.info(f"CLI path: {self.cli_path}")
            logger.info(f"Model path: {self.model_path}")
            logger.info(f"MMPROJ path: {self.mmproj_path}")
            
            # Save frames as temporary images
            for i, frame in enumerate(frames):
                temp_path = os.path.join(self.temp_dir, f"temp_frame_{int(time.time() * 1000)}_{i}.jpg")
                logger.info(f"Saving frame {i} to {temp_path}")
                frame.save(temp_path, "JPEG", quality=95)
                
                if not os.path.exists(temp_path):
                    logger.error(f"Failed to save frame {i}")
                    continue
                    
                temp_paths.append(temp_path)
            
            if not temp_paths:
                logger.error("No frames were saved successfully")
                return "Error: Failed to save frames"
            
            logger.info(f"Successfully saved {len(temp_paths)} frames")
            
            # Construct command with multi-image support
            cmd = [
                self.cli_path,
                "-m", os.path.abspath(self.model_path),
                "--mmproj", os.path.abspath(self.mmproj_path),
                "-t", str(self.n_threads),
                "-n", "256",  # Increased token limit for multi-image
                "--temp", str(self.temp),
                "--ctx-size", "8192",  # Increased context for multi-image
                "--top-p", str(self.top_p),
                "--top-k", str(self.top_k)
            ]
            
            # Add system prompt if provided
            if system_prompt:
                cmd.extend(["--system", system_prompt])
            
            # Add images
            for temp_path in temp_paths:
                cmd.extend(["--image", os.path.abspath(temp_path)])
            
            # Add query
            cmd.extend(["-p", query])
            
            logger.info("\nExecuting command:")
            logger.info(" ".join(cmd))
            
            # Run the command with retry mechanism
            max_retries = 3
            retry_count = 0
            while retry_count < max_retries:
                try:
                    logger.info(f"\nAttempt {retry_count + 1}/{max_retries}")
                    result = subprocess.run(
                        cmd, 
                        capture_output=True, 
                        text=True, 
                        timeout=60,  # Increased timeout for multi-image
                        cwd=os.getcwd()
                    )
                    
                    logger.info(f"Return code: {result.returncode}")
                    logger.info("Stdout: " + (result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout))
                    logger.info("Stderr: " + (result.stderr[:500] + "..." if len(result.stderr) > 500 else result.stderr))
                    
                    # Process output
                    if result.returncode == 0 and result.stdout:
                        response = result.stdout.strip()
                        # Clean up the response
                        response = ' '.join(
                            line.strip() for line in response.split('\n') 
                            if line.strip() 
                            and not any(x in line.lower() for x in 
                                      ['loading', 'llama_print', 'log start', 'clip_model', 
                                       'user:', 'assistant:', '<user>', '<assistant>'])
                        )
                        logger.info(f"\nProcessed response: {response[:200]}...")
                        return response if response else "Error: No caption generated"
                    else:
                        retry_count += 1
                        if retry_count < max_retries:
                            logger.warning(f"Command failed, retrying in 1 second...")
                            time.sleep(1)  # Wait before retrying
                            continue
                        return f"Error: Command failed with code {result.returncode}"
                        
                except subprocess.TimeoutExpired:
                    logger.error("Command timed out")
                    retry_count += 1
                    if retry_count < max_retries:
                        logger.warning(f"Timeout, retrying in 1 second...")
                        time.sleep(1)  # Wait before retrying
                        continue
                    return "Error: Processing timeout after retries"
                    
        except Exception as e:
            logger.error(f"Error in process_frames: {str(e)}")
            return f"Error: {str(e)}"
            
        finally:
            # Clean up temp files
            logger.info("\nCleaning up temporary files...")
            for temp_path in temp_paths:
                try:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                        logger.info(f"Removed {temp_path}")
                except Exception as e:
                    logger.error(f"Failed to remove {temp_path}: {str(e)}")
    
    def process_frame(self, frame: Image.Image, query: str = "Describe this image briefly.") -> str:
        """Process a single frame with MiniCPM-V (legacy method)"""
        return self.process_frames([frame], query=query) 