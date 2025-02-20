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

# Create necessary directories with proper permissions
sudo mkdir -p /var/log/supervisor
sudo chown -R ubuntu:ubuntu /var/log/supervisor
sudo mkdir -p uploads temp_frames

# Setup Nginx
sudo cp nginx.conf /etc/nginx/sites-available/vidclip
sudo ln -sf /etc/nginx/sites-available/vidclip /etc/nginx/sites-enabled
sudo rm -f /etc/nginx/sites-enabled/default
sudo systemctl restart nginx

# Setup Supervisor
sudo cp supervisor.conf /etc/supervisor/conf.d/vidclip.conf
sudo supervisorctl reread
sudo supervisorctl update

# Wait for supervisor to initialize
sleep 5

# Start the application
sudo supervisorctl restart vidclip

# Set permissions
sudo chown -R ubuntu:ubuntu .
sudo chmod -R 755 .

# Check status
echo "Checking service status..."
sudo supervisorctl status vidclip
curl -I http://localhost:5000

echo "Deployment completed!" 