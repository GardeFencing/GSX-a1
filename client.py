import requests
import os
import json
from typing import Optional, Dict, Any

class VidCLIPClient:
    """Client for interacting with VidCLIP backend"""
    def __init__(self, base_url: str = "http://your-ec2-ip"):
        self.base_url = base_url.rstrip('/')
        
    def upload_video(self, video_path: str) -> Dict[str, Any]:
        """
        Upload a video file to the backend
        
        Args:
            video_path: Path to the video file
            
        Returns:
            dict: Response from the server
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        # Prepare the file for upload
        with open(video_path, 'rb') as f:
            files = {'video': (os.path.basename(video_path), f, 'video/mp4')}
            
            # Send POST request to upload endpoint
            response = requests.post(
                f"{self.base_url}/upload",
                files=files,
                timeout=300  # 5 minutes timeout
            )
            
            response.raise_for_status()
            return response.json()
    
    def ask_question(self, query: str, frame_index: Optional[int] = None) -> Dict[str, Any]:
        """
        Ask a question about the uploaded video
        
        Args:
            query: Question about the video
            frame_index: Optional specific frame to ask about
            
        Returns:
            dict: Response from the server
        """
        data = {
            "query": query
        }
        if frame_index is not None:
            data["current_frame"] = frame_index
            
        # Send POST request to chat endpoint
        response = requests.post(
            f"{self.base_url}/chat",
            json=data,
            timeout=60  # 1 minute timeout
        )
        
        response.raise_for_status()
        return response.json()
    
    def get_frame(self, frame_idx: int) -> Dict[str, Any]:
        """
        Get a specific frame with its caption
        
        Args:
            frame_idx: Index of the frame to retrieve
            
        Returns:
            dict: Frame data and caption
        """
        response = requests.get(
            f"{self.base_url}/get_frame/{frame_idx}",
            timeout=30
        )
        
        response.raise_for_status()
        return response.json()

def main():
    # Replace with your EC2 instance's public IP or domain
    client = VidCLIPClient("http://your-ec2-ip")
    
    # Example usage
    video_path = "path/to/your/video.mp4"
    
    try:
        # Upload video
        print("Uploading video...")
        upload_result = client.upload_video(video_path)
        print(f"Upload result: {json.dumps(upload_result, indent=2)}")
        
        # Ask a question about the video
        print("\nAsking question...")
        question = "What specific fencing position or technique is being demonstrated?"
        chat_result = client.ask_question(question)
        print(f"Answer: {json.dumps(chat_result, indent=2)}")
        
        # Get a specific frame
        print("\nGetting frame 0...")
        frame_result = client.get_frame(0)
        print(f"Frame caption: {frame_result.get('caption', 'No caption available')}")
        
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with server: {str(e)}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 