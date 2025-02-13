import torch
import requests
import json
import sys
import time

def main():
    # Print CUDA availability and device info
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Test prompt
    prompt = "Hello! Can you help me understand what capabilities you have?"
    
    try:
        print("Testing Ollama API...")
        print(f"Sending prompt: {prompt}")
        
        # Prepare the request
        url = "http://localhost:11434/api/generate"
        data = {
            "model": "llama3.2-vision",
            "prompt": prompt,
            "stream": False
        }
        
        # Send request with a longer timeout
        print("\nWaiting for response (this may take a minute or two)...")
        response = requests.post(url, json=data, timeout=180)  # 3 minute timeout
        
        if response.status_code == 200:
            result = response.json()
            print("\nModel response:")
            print(result.get('response', 'No response in result'))
        else:
            print(f"\nError: Received status code {response.status_code}")
            print(response.text)
            
    except requests.exceptions.Timeout:
        print("\nError: Request timed out after 3 minutes")
    except requests.exceptions.ConnectionError:
        print("\nError: Could not connect to Ollama service. Make sure it's running.")
    except Exception as e:
        print(f"\nError occurred: {str(e)}")

if __name__ == "__main__":
    main() 