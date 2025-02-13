import os
from huggingface_hub import hf_hub_download
import shutil

def download_llama_vision():
    # Define model info
    repo_id = "leafspark/Llama-3.2-11B-Vision-Instruct-GGUF"
    model_path = os.path.expanduser("~/.llama/checkpoints/Llama3.2-11B-Vision")
    
    # Create directories if they don't exist
    os.makedirs(model_path, exist_ok=True)
    
    # Files to download
    files = [
        "Llama-3.2-11B-Vision-Instruct.Q4_K_M.gguf",  # Quantized model (smaller size)
        "Llama-3.2-11B-Vision-Instruct-mmproj.f16.gguf"  # Vision projection model
    ]
    
    print("Downloading Llama 3.2 Vision model files...")
    
    for file in files:
        print(f"\nDownloading {file}...")
        try:
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=file,
                local_dir=model_path
            )
            print(f"Successfully downloaded {file} to {downloaded_path}")
        except Exception as e:
            print(f"Error downloading {file}: {str(e)}")
            raise

if __name__ == "__main__":
    download_llama_vision() 