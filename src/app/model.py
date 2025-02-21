import os
import logging
import subprocess
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoLLM:
    """Wrapper for MiniCPM-V CLI interface"""
    def __init__(self):
        self.base_dir = os.path.abspath(os.getcwd())
        self.model_path = os.path.join(self.base_dir, "MiniCPM-V-2_6-gguf/ggml-model-Q4_K_M.gguf")
        self.mmproj_path = os.path.join(self.base_dir, "MiniCPM-V-2_6-gguf/mmproj-model-f16.gguf")
        self.cli_path = os.path.join(self.base_dir, "llama.cpp/build/bin/llama-minicpmv-cli")
        
        # Verify paths exist
        if not os.path.exists(self.model_path):
            raise ValueError(f"Model file not found: {self.model_path}")
        if not os.path.exists(self.mmproj_path):
            raise ValueError(f"MMPROJ file not found: {self.mmproj_path}")
        if not os.path.exists(self.cli_path):
            raise ValueError(f"CLI executable not found: {self.cli_path}")
            
        logger.info(f"Initialized with:")
        logger.info(f"Model: {self.model_path}")
        logger.info(f"MMPROJ: {self.mmproj_path}")
        logger.info(f"CLI: {self.cli_path}")

    def generate_response(self, video_path: str, question: str) -> Optional[str]:
        """
        Generate response using llama-minicpmv-cli
        
        Args:
            video_path (str): Path to video file
            question (str): Question about the video
            
        Returns:
            Optional[str]: Generated response or None if error occurs
        """
        try:
            # Construct command
            cmd = [
                self.cli_path,
                "-m", self.model_path,
                "--mmproj", self.mmproj_path,
                "-c", "4096",
                "--temp", "0.7",
                "--top-p", "0.8",
                "--top-k", "100",
                "--repeat-penalty", "1.05",
                "-v", os.path.abspath(video_path),  # Use -v for video input
                "-p", question
            ]
            
            logger.info(f"Running command: {' '.join(cmd)}")
            
            # Run command with retries
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=300,  # 5 minutes timeout
                        check=False
                    )
                    
                    # Process output
                    if result.stdout:
                        lines = result.stdout.split('\n')
                        response_lines = []
                        capture_mode = False
                        
                        for line in lines:
                            line = line.strip()
                            # Start capturing after assistant tag
                            if '<assistant>' in line.lower():
                                capture_mode = True
                                continue
                            # Stop capturing at these markers
                            if any(x in line.lower() for x in ['<user>', 'llama_print', '<end>']):
                                capture_mode = False
                                continue
                            # Skip initialization lines
                            if any(x in line.lower() for x in [
                                'clip_model_load', 'loading', 'system_info:', 
                                'ggml', 'llama', '<start>', '<|im_start|', '<|im_end|'
                            ]):
                                continue
                            # Capture actual response content
                            if capture_mode and line:
                                response_lines.append(line)
                        
                        response = " ".join(response_lines).strip()
                        if response:
                            logger.info(f"Generated response: {response[:100]}...")
                            return response
                    
                    # If we get here, retry
                    retry_count += 1
                    if retry_count < max_retries:
                        logger.warning(f"Retrying... (attempt {retry_count + 1})")
                        continue
                    return None
                    
                except subprocess.TimeoutExpired:
                    logger.error(f"Timeout (attempt {retry_count + 1})")
                    retry_count += 1
                    if retry_count < max_retries:
                        continue
                    return None
                    
                except Exception as e:
                    logger.error(f"Error: {str(e)}")
                    retry_count += 1
                    if retry_count < max_retries:
                        continue
                    return None
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return None

# Initialize global model
model = VideoLLM() 