from flask import Flask, request, jsonify, render_template, send_file
import os
import logging
import cv2
import base64
import base32_lib
import numpy as np
from PIL import Image
import io
from pathlib import Path
import time
import threading
from functools import wraps

# Import GardeRAG components
from GardeRAG.model import GardeRAG
from GardeRAG.frame import Frame, FrameStorage

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Global instances
garde_rag = None
frame_storage = None

def init_garde_rag():
    """Initialize GardeRAG instance."""
    global garde_rag, frame_storage
    if garde_rag is None:
        garde_rag = GardeRAG()
        frame_storage = FrameStorage()
        logger.info("GardeRAG initialized successfully")

init_garde_rag()

# ------------------------------------------------------------------------------
# ROUTES
# ------------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    """Handle video upload and processing."""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file uploaded'}), 400
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        # Save video file
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
        video_file.save(video_path)
        
        try:
            # Process video
            result = garde_rag.process_video(video_path)
            return jsonify({
                'message': 'Video processed successfully',
                'total_frames': result['num_frames'],
                'frame_indices': result['frame_indices'],
                'analyses': [
                    {
                        'frame_index': i,
                        'analysis': analysis.get('response', 'No analysis available')
                    }
                    for i, analysis in enumerate(result['frame_analyses'])
                ]
            })
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            return jsonify({'error': 'Error processing video'}), 500
    except Exception as e:
        logger.error(f"Error in upload: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat queries about the video."""
    try:
        if not garde_rag:
            return jsonify({'error': 'Please upload a video first'}), 400
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'No query provided'}), 400
            
        query = data['query']
        frame_index = data.get('frame_index')
        
        # Generate response
        response = garde_rag.chat(query, frame_index)
        return jsonify({'response': response})
            
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/frame/<int:frame_idx>')
def get_frame(frame_idx):
    try:
        if not garde_rag or not garde_rag.frame_storage.has_frames():
            return jsonify({'error': 'No video loaded'}), 400
            
        frame = garde_rag.frame_storage.get_frame(frame_idx)
        if frame is None:
            return jsonify({'error': 'Invalid frame index'}), 400
            
        # Read and convert frame
        frame_image = cv2.imread(str(frame.path))
        frame_rgb = cv2.cvtColor(frame_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Convert to base64
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='JPEG', quality=95)
        img_byte_arr = img_byte_arr.getvalue()
        frame_b64 = base64.b64encode(img_byte_arr).decode('utf-8')
        
        return jsonify({
            'frame': frame_b64,
            'frame_number': frame.number,
            'timestamp': frame.timestamp,
            'total_frames': len(garde_rag.frame_storage.frames)
        })
    except Exception as e:
        logger.error(f"Error getting frame: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/view_video')
def view_video():
    if not garde_rag or not garde_rag.frame_storage.has_frames():
        return render_template('index.html', error="Please upload a video first")
    return render_template('viewer.html', total_frames=len(garde_rag.frame_storage.frames))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
