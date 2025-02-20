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
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-setuptools \
    nginx \
    supervisor \
    cmake \
    pkg-config \
    protobuf-compiler \
    libprotobuf-dev

# Stop services and cleanup
echo "Cleaning up previous installation..."
sudo supervisorctl stop vidclip || true
sudo rm -f /etc/supervisor/conf.d/vidclip.conf
sudo rm -f /etc/nginx/sites-enabled/vidclip
sudo rm -f /etc/nginx/sites-available/vidclip

# Deactivate any existing virtual environment
deactivate 2>/dev/null || true

# Remove existing virtual environment if it exists
echo "Setting up fresh virtual environment..."
rm -rf venv_py310

# Create and activate virtual environment with Python 3.10
echo "Creating Python 3.10 virtual environment..."
python3.10 -m venv venv_py310
source venv_py310/bin/activate

# Verify Python version
echo "Python version:"
python --version

# Install Python dependencies
echo "Installing Python packages..."
python -m pip install --upgrade pip
python -m pip install --no-cache-dir wheel setuptools
python -m pip install --no-cache-dir -r requirements.txt

# Create necessary directories with proper permissions
echo "Setting up directories..."
sudo mkdir -p /var/log/supervisor
sudo chown -R ubuntu:ubuntu /var/log/supervisor
sudo mkdir -p uploads temp_frames
sudo chown -R ubuntu:ubuntu uploads temp_frames

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
echo "Waiting for supervisor to initialize..."
sleep 5

# Start the application
echo "Starting the application..."
sudo supervisorctl restart vidclip
sleep 10  # Increased wait time

# Set permissions
sudo chown -R ubuntu:ubuntu .
sudo chmod -R 755 .

# Check status
echo "Checking service status..."
sudo supervisorctl status vidclip

# Test the server with increased timeout and attempts
echo "Testing server connection..."
max_attempts=12
attempt=1
while [ $attempt -le $max_attempts ]; do
    if curl -s -I http://localhost:5000 > /dev/null; then
        echo "Server is running!"
        break
    else
        echo "Waiting for server to start... (attempt $attempt/$max_attempts)"
        sleep 10
        attempt=$((attempt + 1))
    fi
done

# Check logs if server didn't start
if ! curl -s -I http://localhost:5000 > /dev/null; then
    echo "Server failed to start. Checking logs..."
    echo "Supervisor error log:"
    sudo tail -n 50 /var/log/supervisor/vidclip.err.log
    echo -e "\nSupervisor output log:"
    sudo tail -n 50 /var/log/supervisor/vidclip.out.log
    exit 1
fi

echo "Deployment completed successfully!" 