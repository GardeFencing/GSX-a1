import os
import logging
import subprocess
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class VideoAnalyzer:
    def __init__(self):
        self.base_dir = os.path.abspath(os.getcwd())
        self.llama_cpp_path = os.path.expanduser("~/GSX-a1/llama.cpp")
        self.model_path = os.path.expanduser("~/GSX-a1/MiniCPM-V-2_6-gguf/ggml-model-Q4_K_M.gguf")
        self.mmproj_path = os.path.expanduser("~/GSX-a1/MiniCPM-V-2_6-gguf/mmproj-model-f16.gguf")
        self.current_video = None
        self._verify_paths()

    def _verify_paths(self):
        """Verify that all required paths exist"""
        if not os.path.exists(self.llama_cpp_path):
            raise RuntimeError(f"llama.cpp directory not found at {self.llama_cpp_path}")
        if not os.path.exists(self.model_path):
            raise RuntimeError(f"Model file not found at {self.model_path}")
        if not os.path.exists(self.mmproj_path):
            raise RuntimeError(f"MM project model not found at {self.mmproj_path}")

    def analyze_video(self, video_path, query):
        """Analyze video using MiniCPM-V"""
        try:
            # Convert video path to absolute path relative to base directory
            abs_video_path = os.path.abspath(video_path)
            if not os.path.exists(abs_video_path):
                raise RuntimeError(f"Video file not found at {abs_video_path}")

            logger.info(f"Analyzing video: {abs_video_path}")
            
            cmd = [
                "./llama-minicpmv-cli",
                "-m", self.model_path,
                "--mmproj", self.mmproj_path,
                "-c", "8192",
                "--temp", "0.7",
                "--top-p", "0.8",
                "--top-k", "100",
                "--repeat-penalty", "1.05",
                "--video", abs_video_path,
                "-p", query
            ]
            
            logger.info(f"Running command: {' '.join(cmd)}")
            
            # Run the command from the llama.cpp directory
            result = subprocess.run(
                cmd,
                cwd=self.llama_cpp_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Check if output is empty or contains error messages
            output = result.stdout.strip()
            if not output or "error" in output.lower():
                raise RuntimeError(f"Failed to analyze video. Output: {output}")
            
            return output
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running video analysis: {e.stderr}")
            raise RuntimeError(f"Video analysis failed: {e.stderr}")
        except Exception as e:
            logger.error(f"Unexpected error during video analysis: {e}")
            raise

# Initialize the video analyzer
video_analyzer = VideoAnalyzer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No video file selected'}), 400
    
    try:
        # Log upload attempt
        logger.info(f"Processing video upload: {video_file.filename}")
        
        # Save the uploaded video
        filename = secure_filename(video_file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video_file.save(video_path)
        
        # Log successful save
        logger.info(f"Video saved to: {video_path}")
        
        # Store the current video path
        video_analyzer.current_video = video_path
        
        # Verify file exists and is readable
        if not os.path.exists(video_path):
            raise RuntimeError(f"Failed to save video file at {video_path}")
        
        # Log file size
        file_size = os.path.getsize(video_path)
        logger.info(f"Video file size: {file_size} bytes")
        
        # Perform initial analysis
        logger.info("Starting initial video analysis...")
        initial_analysis = video_analyzer.analyze_video(
            video_path,
            "Please describe what is happening in this video in detail."
        )
        logger.info("Initial analysis completed successfully")
        
        return jsonify({
            'success': True,
            'analysis': initial_analysis
        })
        
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    if not video_analyzer.current_video:
        return jsonify({'error': 'No video has been uploaded yet'}), 400
    
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({'error': 'No query provided'}), 400
    
    try:
        logger.info(f"Processing chat query for video: {video_analyzer.current_video}")
        logger.info(f"Query: {data['query']}")
        
        response = video_analyzer.analyze_video(
            video_analyzer.current_video,
            data['query']
        )
        
        logger.info("Chat response generated successfully")
        return jsonify({'response': response})
        
    except Exception as e:
        logger.error(f"Error processing chat query: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 
