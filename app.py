from flask import Flask, request, jsonify, render_template, send_file
import os
from video_chat import VideoChat
import logging
import cv2
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
video_chat = None  # Global VideoChat instance

# Ensure upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global video_chat
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file uploaded'}), 400
            
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        # Save the uploaded file
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
        video_file.save(video_path)
        
        # Initialize VideoChat and process video
        video_chat = VideoChat()  # Create instance without arguments
        video_chat.process_video(video_path)  # Process video separately
        
        return jsonify({
            'message': 'Video processed successfully',
            'total_frames': len(video_chat.frames)
        })
        
    except Exception as e:
        logging.error(f"Error processing upload: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    global video_chat
    try:
        if not video_chat:
            return jsonify({'error': 'Please upload a video first'}), 400
            
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'No query provided'}), 400
            
        query = data['query']
        response = video_chat.chat(query)
        
        return jsonify({
            'response': response
        })
        
    except Exception as e:
        logging.error(f"Error in chat: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_frame/<int:frame_idx>')
def get_frame(frame_idx):
    global video_chat
    try:
        if not video_chat or frame_idx >= len(video_chat.frames):
            return jsonify({'error': 'Invalid frame index'}), 400
            
        frame = video_chat.frames[frame_idx]
        caption = video_chat.captions[frame_idx] if frame_idx < len(video_chat.captions) else ""
        
        # Convert frame to base64
        _, buffer = cv2.imencode('.jpg', frame)
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'frame': frame_b64,
            'caption': caption
        })
        
    except Exception as e:
        logging.error(f"Error getting frame: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 