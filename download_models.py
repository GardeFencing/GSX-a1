import os
from huggingface_hub import hf_hub_download
import sys

def download_models():
    # Set up token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = "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"
    
    # Create models directory
    models_dir = os.path.join("llama.cpp", "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Model files to download
    files_to_download = [
        {
            "repo_id": "openbmb/MiniCPM-V-2_6-gguf",
            "filename": "ggml-model-Q2_K.gguf",  # 2-bit quantized model
            "local_name": "ggml-model-Q2_K.gguf"
        },
        {
            "repo_id": "openbmb/MiniCPM-V-2_6-gguf",
            "filename": "mmproj-model-f16.gguf",
            "local_name": "mmproj-model-f16.gguf"
        }
    ]
    
    # Download each file
    for file_info in files_to_download:
        print(f"Downloading {file_info['filename']}...")
        try:
            downloaded_path = hf_hub_download(
                repo_id=file_info["repo_id"],
                filename=file_info["filename"],
                local_dir=models_dir,
                token=os.environ["HUGGING_FACE_HUB_TOKEN"]
            )
            
            # Rename the file if needed
            local_path = os.path.join(models_dir, file_info["local_name"])
            if downloaded_path != local_path:
                os.rename(downloaded_path, local_path)
            
            print(f"Successfully downloaded and saved to {local_path}")
            # Print file size
            size_mb = os.path.getsize(local_path) / (1024 * 1024)
            print(f"File size: {size_mb:.2f} MB")
        except Exception as e:
            print(f"Error downloading {file_info['filename']}: {str(e)}", file=sys.stderr)
            sys.exit(1)
    
    print("All files downloaded successfully!")

if __name__ == "__main__":
    download_models() 