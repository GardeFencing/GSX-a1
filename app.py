import warnings
warnings.filterwarnings("ignore")

import os
import logging
import gradio as gr
from src.app.response import describe_video

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def gradio_interface(video_file, question):
    """Gradio interface function"""
    try:
        if not video_file or not question:
            return "Please provide both a video and a question."
            
        response = describe_video(video_file.name, question)
        return response
    
    except Exception as e:
        logger.error(f"Error in interface: {str(e)}")
        return f"Error: {str(e)}"

# Create Gradio interface
title = "Video Question Answering with MiniCPM-V"
description = """
This application uses MiniCPM-V 2.6 to answer questions about videos.
Upload a video and ask a question about it - the AI will analyze the video and provide an answer.

Model: MiniCPM-V 2.6 (8B parameters)
Features:
- Multi-image and video understanding
- Multilingual capabilities
- Efficient token generation
"""

# Example questions
examples = [
    ["path/to/example1.mp4", "What is happening in this video?"],
    ["path/to/example2.mp4", "How many people are in the video?"],
    ["path/to/example3.mp4", "Describe the main action in this scene."]
]

iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Video(label="Upload Video"),
        gr.Textbox(label="Question", placeholder="Ask a question about the video...")
    ],
    outputs=gr.Textbox(label="Answer"),
    title=title,
    description=description,
    examples=examples,
    theme="gradio/soft"
)

if __name__ == "__main__":
    iface.launch(debug=True) 