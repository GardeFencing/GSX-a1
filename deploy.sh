#!/bin/bash

# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install system dependencies
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-setuptools \
    python3-venv \
    nginx \
    supervisor

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Setup Nginx
sudo cp nginx.conf /etc/nginx/sites-available/vidclip
sudo ln -s /etc/nginx/sites-available/vidclip /etc/nginx/sites-enabled
sudo rm /etc/nginx/sites-enabled/default
sudo systemctl restart nginx

# Setup Supervisor
sudo cp supervisor.conf /etc/supervisor/conf.d/vidclip.conf
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl restart vidclip

# Create necessary directories
mkdir -p uploads temp_frames logs

# Set permissions
sudo chown -R ubuntu:ubuntu .
sudo chmod -R 755 .

echo "Deployment completed!" 