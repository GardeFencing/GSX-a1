#!/usr/bin/env python3
"""
Script to test if the LoRA fine-tuned model can handle video analysis.
This helps verify compatibility before deploying to production.
"""

import os
import sys
import logging
import subprocess
import argparse
from pathlib import Path
import dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
dotenv.load_dotenv()

def test_video_inference(video_path, lora_model_path, base_model_path=None, test_prompt="What is happening in this video?"):
    """
    Test if a model can perform inference on a video.
    
    Args:
        video_path: Path to a test video file
        lora_model_path: Path to the LoRA fine-tuned model directory
        base_model_path: Path to the base model directory (for fallback testing)
        test_prompt: Prompt to use for testing
    
    Returns:
        dict: Test results including success status and details
    """
    if not os.path.exists(video_path):
        return {
            "success": False,
            "error": f"Test video not found at {video_path}",
            "recommendations": ["Make sure to provide a valid video file path"]
        }
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    llama_cpp_path = os.path.join(script_dir, "llama.cpp")
    
    if not os.path.exists(llama_cpp_path):
        return {
            "success": False,
            "error": f"llama.cpp directory not found at {llama_cpp_path}",
            "recommendations": ["Clone and build llama.cpp repository"]
        }
    
    # Check for llama-minicpmv-cli
    minicpmv_cli_path = os.path.join(llama_cpp_path, "llama-minicpmv-cli")
    if not os.path.exists(minicpmv_cli_path):
        return {
            "success": False,
            "error": f"llama-minicpmv-cli not found at {minicpmv_cli_path}",
            "recommendations": [
                "Rebuild llama.cpp with video support (LLAMA_FFMPEG=1)",
                "Ensure the llama-minicpmv-cli binary was successfully compiled"
            ]
        }
    
    # Verify video support in binary
    try:
        help_output = subprocess.check_output(
            [minicpmv_cli_path, "--help"],
            stderr=subprocess.STDOUT,
            cwd=llama_cpp_path,
            universal_newlines=True
        )
        
        if "--video" not in help_output:
            return {
                "success": False,
                "error": "llama-minicpmv-cli does not appear to support video (--video flag not found in help output)",
                "recommendations": [
                    "Rebuild llama.cpp with LLAMA_FFMPEG=1",
                    "Make sure ffmpeg development libraries are installed"
                ]
            }
    except subprocess.CalledProcessError as e:
        return {
            "success": False,
            "error": f"Error checking llama-minicpmv-cli: {e}",
            "output": e.output if hasattr(e, "output") else None,
            "recommendations": ["Check if the binary is working correctly"]
        }
    
    # Find LoRA model files
    model_path = None
    mmproj_path = None
    
    # Look for model files with common extensions
    model_candidates = [
        "model.bin", 
        "pytorch_model.bin", 
        "model.safetensors", 
        "consolidated.00.pth",
        "pytorch_model.bin.index.json"
    ]
    
    # Check for model file
    for candidate in model_candidates:
        candidate_path = os.path.join(lora_model_path, candidate)
        if os.path.exists(candidate_path):
            model_path = candidate_path
            break
    
    # Check for mmproj file
    mmproj_candidates = [
        "mmproj-model-f16.gguf",
        "mm_projector.bin",
        "mm_projector/model.safetensors"
    ]
    
    for candidate in mmproj_candidates:
        candidate_path = os.path.join(lora_model_path, candidate)
        if os.path.exists(candidate_path):
            mmproj_path = candidate_path
            break
    
    if not model_path:
        return {
            "success": False,
            "error": f"No model file found in {lora_model_path}",
            "recommendations": [
                "Check that the LoRA model has been properly merged",
                "Ensure the model files exist in the specified directory"
            ]
        }
    
    if not mmproj_path:
        return {
            "success": False,
            "error": f"No mmproj file found in {lora_model_path}",
            "recommendations": [
                "Check that the mmproj file was copied during model merging",
                "Copy mmproj-model-f16.gguf from the base model directory"
            ]
        }
    
    # Test LoRA model
    logger.info(f"Testing LoRA model with video: {video_path}")
    try:
        cmd = [
            minicpmv_cli_path,
            "-m", model_path,
            "--mmproj", mmproj_path,
            "-c", "4096",
            "--temp", "0.7",
            "--top-p", "0.8",
            "--top-k", "100",
            "--repeat-penalty", "1.05",
            "--video", os.path.abspath(video_path),
            "-p", test_prompt,
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        output = subprocess.check_output(
            cmd,
            stderr=subprocess.STDOUT,
            cwd=llama_cpp_path,
            universal_newlines=True,
            timeout=300  # 5 minute timeout
        )
        
        return {
            "success": True,
            "model": "lora",
            "output": output,
            "message": "LoRA model successfully processed video"
        }
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running LoRA model: {e.output if hasattr(e, 'output') else e}")
        
        # If base model path is provided, test with base model too
        if base_model_path:
            base_results = test_base_model(video_path, base_model_path, test_prompt, minicpmv_cli_path, llama_cpp_path)
            if base_results["success"]:
                return {
                    "success": False,
                    "error": f"LoRA model failed, but base model succeeded",
                    "lora_error": e.output if hasattr(e, "output") else str(e),
                    "base_model_success": True,
                    "recommendations": [
                        "The LoRA model may not support video inference",
                        "Consider using the base model for video analysis",
                        "Or retrain the LoRA model with video examples"
                    ]
                }
            else:
                return {
                    "success": False,
                    "error": "Both LoRA and base models failed with video",
                    "lora_error": e.output if hasattr(e, "output") else str(e),
                    "base_error": base_results.get("error"),
                    "recommendations": [
                        "Check if ffmpeg is properly installed",
                        "Verify the video file is in a supported format",
                        "Rebuild llama.cpp with proper video support"
                    ]
                }
        
        return {
            "success": False,
            "error": f"LoRA model failed with video",
            "output": e.output if hasattr(e, "output") else str(e),
            "recommendations": [
                "The LoRA model may not support video inference",
                "Try testing with the base model to see if it's a general issue"
            ]
        }
    except Exception as e:
        logger.exception(f"Unexpected error testing LoRA model: {e}")
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}",
            "recommendations": ["Check the logs for more details"]
        }

def test_base_model(video_path, base_model_path, test_prompt, minicpmv_cli_path, llama_cpp_path):
    """Test the base model with video if LoRA model fails."""
    logger.info(f"Testing base model with video: {video_path}")
    
    # Check for required files
    model_path = os.path.join(base_model_path, os.environ.get("MODEL", "ggml-model-Q4_K_M.gguf"))
    mmproj_path = os.path.join(base_model_path, "mmproj-model-f16.gguf")
    
    if not os.path.exists(model_path):
        return {
            "success": False,
            "error": f"Base model file not found at {model_path}"
        }
    
    if not os.path.exists(mmproj_path):
        return {
            "success": False,
            "error": f"Base mmproj file not found at {mmproj_path}"
        }
    
    try:
        cmd = [
            minicpmv_cli_path,
            "-m", model_path,
            "--mmproj", mmproj_path,
            "-c", "4096",
            "--temp", "0.7",
            "--top-p", "0.8",
            "--top-k", "100",
            "--repeat-penalty", "1.05",
            "--video", os.path.abspath(video_path),
            "-p", test_prompt,
        ]
        
        logger.info(f"Running base model command: {' '.join(cmd)}")
        output = subprocess.check_output(
            cmd,
            stderr=subprocess.STDOUT,
            cwd=llama_cpp_path,
            universal_newlines=True,
            timeout=300  # 5 minute timeout
        )
        
        return {
            "success": True,
            "model": "base",
            "output": output,
            "message": "Base model successfully processed video"
        }
    except subprocess.CalledProcessError as e:
        return {
            "success": False,
            "error": f"Base model failed: {e.output if hasattr(e, 'output') else str(e)}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error with base model: {str(e)}"
        }

def update_env_file(use_lora):
    """Update the .env file based on test results."""
    try:
        # Read existing .env file
        env_path = '.env'
        if not os.path.exists(env_path):
            logger.warning(f"{env_path} not found, creating new file")
            with open(env_path, 'w') as f:
                f.write(f"USE_LORA={str(use_lora).lower()}\n")
            return True
        
        # Read existing content
        with open(env_path, 'r') as f:
            lines = f.readlines()
        
        # Update or add USE_LORA line
        use_lora_found = False
        updated_lines = []
        
        for line in lines:
            if line.strip().startswith('USE_LORA='):
                updated_lines.append(f"USE_LORA={str(use_lora).lower()}\n")
                use_lora_found = True
            else:
                updated_lines.append(line)
        
        if not use_lora_found:
            updated_lines.append(f"USE_LORA={str(use_lora).lower()}\n")
        
        # Write updated content
        with open(env_path, 'w') as f:
            f.writelines(updated_lines)
        
        return True
    except Exception as e:
        logger.error(f"Error updating .env file: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Test LoRA model compatibility with video analysis')
    parser.add_argument('--video', type=str, default='fencing_trimmed.mp4',
                       help='Path to test video file')
    parser.add_argument('--lora_path', type=str, 
                        default='MiniCPM-V/finetune/output/output_fencing_lora/merged_model',
                        help='Path to LoRA fine-tuned model directory')
    parser.add_argument('--base_path', type=str,
                        default='models/MiniCPM-V-2_6-gguf',
                        help='Path to base model directory (for fallback testing)')
    parser.add_argument('--prompt', type=str, 
                        default='What fencing techniques are shown in this video?',
                        help='Test prompt to use')
    parser.add_argument('--update_env', action='store_true',
                        help='Update .env file based on test results')
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.video):
        logger.error(f"Test video not found: {args.video}")
        sys.exit(1)
    
    if not os.path.exists(args.lora_path):
        logger.error(f"LoRA model directory not found: {args.lora_path}")
        sys.exit(1)
    
    # Run the test
    result = test_video_inference(
        args.video,
        args.lora_path,
        args.base_path if os.path.exists(args.base_path) else None,
        args.prompt
    )
    
    # Print results
    if result["success"]:
        logger.info("✅ LoRA model works with video!")
        logger.info(f"Output: {result.get('output', '')[:200]}...")
        
        if args.update_env:
            if update_env_file(True):
                logger.info("Updated .env file to USE_LORA=true")
    else:
        logger.error(f"❌ LoRA model test failed: {result.get('error', 'Unknown error')}")
        
        if "recommendations" in result:
            logger.info("Recommendations:")
            for rec in result["recommendations"]:
                logger.info(f"  - {rec}")
        
        if result.get("base_model_success"):
            logger.info("✅ Base model works with video - falling back to it")
            if args.update_env:
                if update_env_file(False):
                    logger.info("Updated .env file to USE_LORA=false")
        
        # Exit with error code
        sys.exit(1)

if __name__ == "__main__":
    main() 