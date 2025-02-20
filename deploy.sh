#!/bin/bash

# Exit on error
set -e

echo "Starting deployment..."

# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install system dependencies
echo "Installing system dependencies..."
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-setuptools \
    python3-venv \
    nginx \
    supervisor \
    cmake \
    pkg-config \
    protobuf-compiler \
    libprotobuf-dev

# Create and activate virtual environment
echo "Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "Installing Python packages..."
pip install --upgrade pip
pip install wheel setuptools
pip install --no-cache-dir -r requirements.txt

# Create necessary directories with proper permissions
echo "Setting up directories..."
sudo mkdir -p /var/log/supervisor
sudo chown -R ubuntu:ubuntu /var/log/supervisor
sudo mkdir -p uploads temp_frames

# Setup Nginx
echo "Configuring Nginx..."
sudo cp nginx.conf /etc/nginx/sites-available/vidclip
sudo ln -sf /etc/nginx/sites-available/vidclip /etc/nginx/sites-enabled
sudo rm -f /etc/nginx/sites-enabled/default
sudo systemctl restart nginx

# Setup Supervisor
echo "Configuring Supervisor..."
sudo cp supervisor.conf /etc/supervisor/conf.d/vidclip.conf
sudo supervisorctl reread
sudo supervisorctl update

# Wait for supervisor to initialize
sleep 5

# Start the application
echo "Starting the application..."
sudo supervisorctl restart vidclip
sleep 5  # Wait for the application to start

# Set permissions
sudo chown -R ubuntu:ubuntu .
sudo chmod -R 755 .

# Check status
echo "Checking service status..."
sudo supervisorctl status vidclip

# Test the server
echo "Testing server connection..."
for i in {1..6}; do
    if curl -s -I http://localhost:5000 > /dev/null; then
        echo "Server is running!"
        break
    else
        echo "Waiting for server to start... (attempt $i/6)"
        sleep 5
    fi
done

# Check logs if server didn't start
if ! curl -s -I http://localhost:5000 > /dev/null; then
    echo "Server failed to start. Checking logs..."
    echo "Supervisor logs:"
    sudo tail -n 50 /var/log/supervisor/vidclip.err.log
    sudo tail -n 50 /var/log/supervisor/vidclip.out.log
    exit 1
fi

echo "Deployment completed successfully!" 