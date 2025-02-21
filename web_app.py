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
        
        # Verify llama-minicpmv-cli exists
        cli_path = os.path.join(self.llama_cpp_path, "llama-minicpmv-cli")
        if not os.path.exists(cli_path):
            raise RuntimeError(f"llama-minicpmv-cli not found at {cli_path}")

    def analyze_video(self, video_path, query):
        """Analyze video using MiniCPM-V"""
        try:
            # Convert video path to absolute path relative to base directory
            abs_video_path = os.path.abspath(video_path)
            if not os.path.exists(abs_video_path):
                raise RuntimeError(f"Video file not found at {abs_video_path}")

            logger.info(f"Analyzing video: {abs_video_path}")
            logger.info(f"Video file size: {os.path.getsize(abs_video_path)} bytes")
            
            # Use original parameters that are known to work
            cmd = [
                "./llama-minicpmv-cli",
                "-m", self.model_path,
                "--mmproj", self.mmproj_path,
                "-c", "8192",        # Original context size
                "--temp", "0.7",     # Original temperature
                "--top-p", "0.8",
                "--top-k", "100",    # Original top_k
                "--repeat-penalty", "1.05",
                "--video", abs_video_path,  # Use --video flag
                "-p", query
            ]
            
            logger.info(f"Running command: {' '.join(cmd)}")
            
            # Run the command from the llama.cpp directory with timeout
            result = subprocess.run(
                cmd,
                cwd=self.llama_cpp_path,
                capture_output=True,
                text=True,
                check=True,
                timeout=300  # 5 minute timeout
            )
            
            # Check if output is empty or contains error messages
            output = result.stdout.strip()
            if not output:
                raise RuntimeError("No output generated from video analysis")
            if "error" in output.lower():
                raise RuntimeError(f"Failed to analyze video. Output: {output}")
            
            # Log success
            logger.info("Video analysis completed successfully")
            logger.info(f"Output length: {len(output)} characters")
            
            return output
            
        except subprocess.TimeoutExpired:
            logger.error("Video analysis timed out after 5 minutes")
            raise RuntimeError("Analysis is taking too long. Please try again with a shorter video.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running video analysis: {e.stderr}")
            raise RuntimeError(f"Video analysis failed: {e.stderr}")
        except Exception as e:
            logger.error(f"Unexpected error during video analysis: {e}")
            raise

# Initialize the video analyzer
video_analyzer = VideoAnalyzer()

# Configure Flask app
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['PERMANENT_SESSION_LIFETIME'] = 600  # 10 minutes
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

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
        
        # Generate initial coach's report
        logger.info("Generating initial coach's report...")
        initial_report = video_analyzer.analyze_video(
            video_path,
            """As a fencing coach, give me a detailed but conversational analysis of this video. Focus on the fencer's technique, what they're doing well, and specific areas where they could improve. Include clear, actionable feedback that would help them enhance their performance."""
        )
        logger.info("Initial report generated successfully")
        
        return jsonify({
            'success': True,
            'report': initial_report
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
        
        # Enhance the query with coaching context
        coaching_query = f"""As a fencing coach watching this video, {data['query']} Give specific, practical advice based on what you observe."""
        
        response = video_analyzer.analyze_video(
            video_analyzer.current_video,
            coaching_query
        )
        
        logger.info("Chat response generated successfully")
        return jsonify({'response': response})
        
    except Exception as e:
        logger.error(f"Error processing chat query: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 
