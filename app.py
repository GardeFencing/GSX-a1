import warnings
warnings.filterwarnings("ignore")

import os
import logging
import gradio as gr
from src.app.response import describe_video

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoChat:
    def __init__(self):
        self.current_video = None
        self.chat_history = []

    def analyze_video(self, video_file, question):
        """Process video and generate initial analysis"""
        try:
            if not video_file:
                return [], "Please provide a video file."
                
            self.current_video = video_file
            response = describe_video(video_file, question)
            self.chat_history = [(question, response)]
            return self.chat_history, ""

        except Exception as e:
            logger.error(f"Error in video analysis: {str(e)}")
            return [], f"Error: {str(e)}"

    def chat(self, message, history):
        """Handle ongoing chat conversation"""
        try:
            if not self.current_video:
                return history + [(message, "Please upload a video first.")]

            response = describe_video(self.current_video, message)
            return history + [(message, response)]

        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            return history + [(message, f"Error: {str(e)}")]

def create_demo():
    video_chat = VideoChat()
    
    with gr.Blocks(title="Fencing Video Analysis Chatbot", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🤺 Fencing Video Analysis Chatbot")
        gr.Markdown("Upload a video of fencing techniques and get detailed feedback and analysis.")
        
        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(
                    label="Upload Fencing Video",
                    sources=["upload"],
                    height=400
                )
                initial_question = gr.Textbox(
                    label="Initial Analysis Question",
                    placeholder="What would you like to know about this video?",
                    value="Please analyze this fencing video and provide detailed feedback on technique, form, and areas for improvement.",
                    lines=3
                )
                with gr.Row():
                    analyze_btn = gr.Button("🔍 Analyze Video", variant="primary")
                    clear_btn = gr.ClearButton(
                        components=[video_input, initial_question],
                        value="🔄 Reset"
                    )
            
            with gr.Column(scale=1):
                error_box = gr.Textbox(
                    label="Status",
                    visible=True,
                    interactive=False
                )
                chatbot = gr.Chatbot(
                    label="Analysis & Conversation",
                    height=500,
                    bubble_full_width=False,
                    show_copy_button=True
                )
                msg = gr.Textbox(
                    label="Ask a follow-up question",
                    placeholder="Ask about specific techniques, movements, or suggestions for improvement...",
                    lines=2
                )
                clear_chat = gr.ClearButton([msg, chatbot], value="🗑️ Clear Chat")
        
        # Example questions
        gr.Examples(
            examples=[
                ["How is my lunge technique and what can I improve?"],
                ["Analyze my footwork and balance during the movements."],
                ["What's the quality of my blade control and point accuracy?"],
                ["Evaluate my defensive position and guard stance."],
                ["Are my movements efficient and well-timed?"]
            ],
            inputs=msg,
            label="Example Questions"
        )
        
        # Set up event handlers
        analyze_btn.click(
            fn=video_chat.analyze_video,
            inputs=[video_input, initial_question],
            outputs=[chatbot, error_box]
        )
        
        msg.submit(
            fn=video_chat.chat,
            inputs=[msg, chatbot],
            outputs=chatbot
        )
        
    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=5000,
        share=True,
        show_error=True
    ) 