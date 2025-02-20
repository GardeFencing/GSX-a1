from gevent import monkey
monkey.patch_all()

from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

from flask import Flask
from web_app import app
from config import Config

if __name__ == '__main__':
    # Initialize app with config
    Config.init_app(app)
    
    # Run with Flask's development server if not running with Gunicorn
    app.run(host='0.0.0.0', port=5000) 